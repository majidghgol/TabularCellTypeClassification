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
from InferSent.models import InferSent
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd
from functools import reduce
import re
import tabulate

def predict_labels(t, cl_model, ce_model, fe_model, senc, mode='ce+f', device='cpu'):
    if 'ce' in mode: ce_dim = ce_model.encdim*2
    if 'f' in mode: fenc_dim = fe_model.encdim
    if mode == 'ce+f':
        cl_input_dim = ce_dim+fenc_dim 
        runce = runfe = True
    elif mode == 'ce':
        cl_input_dim = ce_dim
        runfe = False
        runce = True
    elif mode == 'fe':
        cl_input_dim = fenc_dim
        runfe = True
        runce = False
    with torch.no_grad():
        tarr = np.array(t['table_array'])
        feature_array = np.array(t['feature_array'])
        n,m = tarr.shape
        
        if runfe: fevtarr = get_fevectarr(feature_array, n, m, fe_model, device)
        if runce: cevtarr = get_cevectarr(tarr, ce_model, senc, device, ce_model.num_context//4, senc_dim=4096)
        if runfe: fevtarr = torch.from_numpy(fevtarr).float()
        if runce: cevtarr = torch.from_numpy(cevtarr).float()
        if mode == 'ce+f':
            features = torch.cat([cevtarr, fevtarr], dim=-1).to(device)  
        elif mode == 'ce':
            features = cevtarr.to(device)
        elif mode == 'fe':
            features = fevtarr.to(device)
        pred = cl_model(features).detach().cpu().numpy()
        pred_labels = np.argmax(pred, axis=-1)
        pred_probs = np.max(pred, axis=-1)
        
    return pred_labels, pred_probs

def predict(test_tables, cl_model, ce_model, fe_model, senc, label2ind, mode='ce+f', device='cpu'):
    if 'ce' in mode: ce_dim = ce_model.encdim*2
    if 'f' in mode: fenc_dim = fe_model.encdim
    if mode == 'ce+f':
        cl_input_dim = ce_dim+fenc_dim 
        runce = runfe = True
    elif mode == 'ce':
        cl_input_dim = ce_dim
        runfe = False
        runce = True
    elif mode == 'fe':
        cl_input_dim = fenc_dim
        runfe = True
        runce = False
    with torch.no_grad():
        test_gt = []
        test_pred = []
        test_pred_proba = []
        for t in test_tables:
            tarr = np.array(t['table_array'])
            feature_array = np.array(t['feature_array'])
            ann_array = t['annotations']
            n,m = tarr.shape
            
            if runfe: fevtarr = get_fevectarr(feature_array, n, m, fe_model, device)
            if runce: cevtarr = get_cevectarr(tarr, ce_model, senc, device, ce_model.num_context//4, senc_dim=4096)
            labels, targets_i, targets_j = get_annotations(ann_array, n, m)
            if runfe: fevtarr = torch.from_numpy(fevtarr).float()
            if runce: cevtarr = torch.from_numpy(cevtarr).float()
            if mode == 'ce+f':
                features = torch.cat([cevtarr, fevtarr], dim=-1).to(device)  
            elif mode == 'ce':
                features = cevtarr.to(device)
            elif mode == 'fe':
                features = fevtarr.to(device)
            pred = cl_model(features).detach().cpu().numpy()
            pred = pred[(targets_i, targets_j)]
            pred_labels = np.argmax(pred, axis=1)
            pred_proba = [pred[i, l] for i, l in enumerate(pred_labels)]
            test_gt += labels
            test_pred += pred_labels.tolist()
            test_pred_proba += pred_proba
    f1macro = f1_score(test_gt, test_pred, average='macro') 
    report = classification_report([label2ind[x] for x in test_gt], 
                                    [label2ind[x] for x in test_pred]) 
    return f1macro, report, test_gt, test_pred, test_pred_proba

def main(spec):
    np.random.seed(spec['seed'])
    torch.manual_seed(spec['seed'])
    
    nthreads = spec['threads']
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)

    input_file = spec['cl']['input_file']
    folds_path = spec['cl']['folds']
    mode = spec['cl']['mode']
    device = spec['device']
    models_path = spec['cl']['model_path']
    ce_dim = spec['ce']['encdim']
    senc_dim = spec['senc_dim']
    window = spec['ce']['window']
    f_dim = spec['fe']['fdim']
    fenc_dim = spec['fe']['enc_dim']
    n_classes = spec['cl']['num_classes']
    infersent_model = spec['infersent_model']
    w2v_path = spec['w2v_path']
    vocab_size = spec['vocab_size']
    half_precision = False
    if device != 'cpu': torch.cuda.set_device(device)

    senc = SentEnc(infersent_model, w2v_path, 
                   vocab_size, device=device, hp=False)
    prep = Preprocess()
    with gzip.open(input_file) as infile:
        tables = np.array([json.loads(line) for li, line in enumerate(infile)])
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
    senc.cache_sentences(list(sentences))
    reports = []
    for fi, fold in enumerate(folds):
        train_tables, dev_tables, test_tables = split_train_test(tables, fold, 1)

        ce_model = CEModel(senc_dim, ce_dim//2, window*4)
        ce_model = ce_model.to(device)
        fe_model = FeatEnc(f_dim, fenc_dim)
        fe_model = fe_model.to(device)
        cl_model = ClassificationModel(ce_dim+fenc_dim, n_classes).to(device)
        
        ce_model.load_state_dict(torch.load(models_path+f'/ce_fold{fi}.model', map_location=device))
        fe_model.load_state_dict(torch.load(models_path+f'/fe_fold{fi}.model', map_location=device))
        cl_model.load_state_dict(torch.load(models_path+f'/cl_fold{fi}.model', map_location=device))
        f1macro, report, _, _, _ = predict(test_tables, cl_model, ce_model, fe_model, senc, label2ind, device=device)
        reports.append(report)
        print(f'fold {fi} test f1-macro = {f1macro}')
    dfs = [get_df(r) for r in reports]
    mean_res = reduce(lambda x, y: x.add(y, fill_value=0), dfs)/len(dfs)
    std_res = [(x-mean_res) ** 2 for x in dfs]
    std_res = reduce(lambda x, y: x.add(y, fill_value=0), dfs)
    std_res = std_res.pow(1./2)/len(dfs)
    print('mean:')
    print(tabulate.tabulate(mean_res, headers='keys', tablefmt='psql'))
    print('STD:')
    print(tabulate.tabulate(std_res, headers='keys', tablefmt='psql'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_path', type=str)
    parser.add_argument('--infersent_source', type=str)
    
    FLAGS, unparsed = parser.parse_known_args()
    spec = json.load(open(FLAGS.spec_path))

    sys.path.append(FLAGS.infersent_source)
    from InferSent.models import InferSent
    from helpers import (CellDatasetInMemory, TableCellSample, Preprocess, SentEnc,
            label2ind, split_train_test, get_nonempty_cells,
            get_annotations, get_cevectarr, get_fevectarr,
            get_class_weights, get_df)
    main(spec)
    