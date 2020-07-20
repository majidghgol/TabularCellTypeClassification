from models import ClassificationModel, CEModel, FeatEnc
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
import time
import gzip
import logging
import os
import numpy as np
import sys
import argparse
import json
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def main(spec):
    input_file = spec['cl']['input_file']
    folds_path = spec['cl']['folds']
    mode = spec['cl']['mode']
    device = spec['device']
    out_path = spec['cl']['model_path']
    ce_path = spec['cl']['ce_model']
    fe_path = spec['cl']['fe_model']
    ce_dim = spec['ce']['encdim']
    senc_dim = spec['senc_dim']
    window = spec['ce']['window']
    f_dim = spec['fe']['fdim']
    fenc_dim = spec['fe']['encdim']
    num_epochs = spec['cl']['epochs']
    lr = spec['cl']['lr']
    train_size = spec['cl']['train_size']
    dev_size = spec['cl']['cv_size']
    n_classes = spec['cl']['num_classes']
    infersent_model = spec['infersent_model']
    w2v_path = spec['w2v_path']
    vocab_size = spec['vocab_size']
    half_precision = False
    if device != 'cpu': torch.cuda.set_device(device)

    senc = SentEnc(infersent_model, w2v_path, 
                   vocab_size, device=device, hp=half_precision)
    prep = Preprocess()
    with gzip.open(input_file) as infile:
        tables = np.array([json.loads(line) for li, line in enumerate(infile) if li < (train_size+dev_size)])
    for i in range(len(tables)): 
        tables[i]['table_array'] = np.array(prep.clean_table_array(tables[i]['table_array']))
    folds = json.load(open(folds_path))
    ## initialize the sentence encodings
    pbar = tqdm(total=len(tables))
    pbar.set_description('initialize sent encodings:')
    sentences = set()
    for t in tables:
        for row in t['table_array']:
            for c in row:
                sentences.add(c)
        pbar.update(1)
    senc.cache_sentences(list(sentences))

    for fi, fold in enumerate(folds):
        ce_model = CEModel(senc_dim, ce_dim//2, window*4)
        ce_model.load_state_dict(torch.load(ce_path, map_location=device))
        ce_model = ce_model.to(device)
        
        fe_model = FeatEnc(f_dim, fenc_dim)
        fe_model.load_state_dict(torch.load(fe_path, map_location=device))
        fe_model = fe_model.to(device)

        cl_model = ClassificationModel(ce_dim+fenc_dim, n_classes).to(device)
        optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=lr)
        optimizer_ce = torch.optim.Adam(ce_model.parameters(), lr=lr/100)
        optimizer_fe = torch.optim.Adam(fe_model.parameters(), lr=lr/100)

        print(f'fold {fi} started ...')
        best_dev_loss = np.inf
        train_tables, dev_tables, test_tables = split_train_test(tables, fold, dev_size)
        
        class_weights = get_class_weights(train_tables)
        class_weights = torch.from_numpy(class_weights).float().to(device)
        loss_func = nn.NLLLoss(weight=class_weights, reduction='mean').to(device)
        
        pbar = tqdm(total=num_epochs*(len(train_tables)+len(dev_tables)))
        pbar.set_description('tr:{:.3f}, dev:{:.3f}'.format(np.NaN, best_dev_loss))
        for e in range(num_epochs):
            eloss_train = 0
            eloss_dev = 0
            for ti, t in enumerate(train_tables):
                tarr = np.array(t['table_array'])
                feature_array = np.array(t['feature_array'])
                ann_array = t['annotations']
                n,m = tarr.shape
                
                fevtarr = get_fevectarr(feature_array, n, m, fe_model, device)
                cevtarr = get_cevectarr(tarr, ce_model, senc, device, ce_model.num_context//4, senc_dim=4096)
                labels, targets_i, targets_j = get_annotations(ann_array, n, m)
                labels = torch.LongTensor(labels).to(device)
                fevtarr = torch.from_numpy(fevtarr).float()
                cevtarr = torch.from_numpy(cevtarr).float()
                features = torch.cat([cevtarr, fevtarr], dim=-1).to(device)
                pred = cl_model(features)

                loss = loss_func(pred[(targets_i, targets_j)], labels)
                eloss_train += loss.item()

                cl_model.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_ce.step()
                optimizer_fe.step()
                pbar.update(1)
                pbar.set_description('tr:{:.3f}, dev:{:.3f}'.format(eloss_train/(ti+1), best_dev_loss))
            with torch.no_grad():
                for t in dev_tables:
                    tarr = np.array(t['table_array'])
                    feature_array = np.array(t['feature_array'])
                    ann_array = t['annotations']
                    n,m = tarr.shape
                    
                    fevtarr = get_fevectarr(feature_array, n, m, fe_model, device)
                    cevtarr = get_cevectarr(tarr, ce_model, senc, device, ce_model.num_context//4, senc_dim=4096)
                    labels, targets_i, targets_j = get_annotations(ann_array, n, m)
                    labels = torch.LongTensor(labels).to(device)
                    fevtarr = torch.from_numpy(fevtarr).float()
                    cevtarr = torch.from_numpy(cevtarr).float()
                    features = torch.cat([cevtarr, fevtarr], dim=-1).to(device)
                    pred = cl_model(features)

                    loss = loss_func(pred[(targets_i, targets_j)], labels)
                    eloss_dev += loss.item()

                    pbar.update(1)
            eloss_train = eloss_train / len(train_tables)
            eloss_dev = eloss_dev / len(dev_tables)
            if eloss_dev < best_dev_loss:
                best_dev_loss = eloss_dev
                torch.save(cl_model.state_dict(), out_path+f'/cl_fold{fi}.model')
                torch.save(fe_model.state_dict(), out_path+f'/fe_fold{fi}.model')
                torch.save(ce_model.state_dict(), out_path+f'/ce_fold{fi}.model')
            pbar.set_description('tr:{:.3f}, dev:{:.3f}'.format(eloss_train, best_dev_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_path', type=str)
    parser.add_argument('--infersent_source', type=str)
    
    FLAGS, unparsed = parser.parse_known_args()
    spec = json.load(open(FLAGS.spec_path))

    np.random.seed(spec['seed'])
    torch.manual_seed(spec['seed'])
    
    nthreads = spec['threads']
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)

    sys.path.append(FLAGS.infersent_source)
    from InferSent.models import InferSent
    from helpers import (CellDatasetInMemory, TableCellSample, Preprocess, SentEnc,
            label2ind, split_train_test, get_nonempty_cells,
            get_annotations, get_cevectarr, get_fevectarr,
            get_class_weights)
    main(spec)