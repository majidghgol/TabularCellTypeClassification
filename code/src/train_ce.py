from models import CEModel
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
import logging

def main(spec):
    np.random.seed(spec['seed'])
    torch.manual_seed(spec['seed'])

    device = spec['device']
    input_file = spec['ce']['input_file']
    out_path = spec['ce']['model_path']
    d = spec['ce']['encdim'] // 2
    num_epochs = spec['ce']['epochs']
    lr = spec['ce']['lr']
    loss = spec['ce']['loss']
    train_size = spec['ce']['train_size']
    dev_size = spec['ce']['cv_size']
    target_p = spec['ce']['target_p']
    min_row = spec['ce']['min_row']
    min_col = spec['ce']['min_col']
    window = spec['ce']['window']
    infersent_model = spec['infersent_model']
    w2v_path = spec['w2v_path']
    vocab_size = spec['vocab_size']
    half_precision = False

    torch.cuda.set_device(device)

    cell_sampler = TableCellSample(target_p, min_row, min_col, window)
    senc = SentEnc(infersent_model, w2v_path, 
                   vocab_size, device=device, hp=half_precision)
    
    with gzip.open(input_file) as infile:
        tables = np.array([json.loads(line) for li, line in enumerate(infile) if li < (train_size+dev_size)])

    inds = np.random.permutation(np.arange(len(tables)))
    dev_tables = tables[inds[:dev_size]]
    train_tables = tables[inds[dev_size:]]

    train_cells = [cell_sampler.sample(t['table_array'], None, None) for t in tqdm(train_tables)]
    train_cells = [x for xx in train_cells for x in xx]
    dev_cells = [cell_sampler.sample(t['table_array'], None, None) for t in tqdm(dev_tables)]
    dev_cells = [x for xx in dev_cells for x in xx]

    ## initialize the sentence encodings
    pbar = tqdm(total=len(train_cells)+len(dev_cells))
    pbar.set_description('initialize sent encodings:')
    sentences = set()
    for c in train_cells:
        sentences.add(c['target'])
        [sentences.add(x) for x in c['context']]
        pbar.update(1)
    for c in dev_cells:
        sentences.add(c['target'])
        [sentences.add(x) for x in c['context']]
        pbar.update(1)
    senc.cache_sentences(list(sentences))

    train_set = CellDatasetInMemory(train_cells, senc)
    dev_set = CellDatasetInMemory(dev_cells, senc)

    dataloader_train = DataLoader(train_set, batch_size=200, num_workers=2)
    dataloader_dev = DataLoader(dev_set, batch_size=200, num_workers=2) 

    ce_model = CEModel(4096, d, window*4).to(device)
    if half_precision:
        ce_model = ce_model.half()

    logging.basicConfig(filename=out_path+'/train.log',level=logging.DEBUG)
    best_dev_loss = np.inf
    for mi, (m, train_loss, dev_loss) in enumerate(ce_fit_iterative(ce_model, lr, loss, 
                                                          dataloader_train, dataloader_dev, 
                                                          num_epochs, device, hp=half_precision)):
        logging.info(f'epoch {mi+1}, train_loss: {train_loss}, dev_loss: {dev_loss}\n')
        if dev_loss < best_dev_loss:
            torch.save(m.state_dict(), out_path+'/CE.model')
            best_dev_loss = dev_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_path', type=str)
    parser.add_argument('--infersent_source', type=str)

    FLAGS, unparsed = parser.parse_known_args()

    spec = json.load(open(FLAGS.spec_path))

    nthreads = spec['threads']
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)

    sys.path.append(FLAGS.infersent_source)
    from InferSent.models import InferSent
    from helpers import CellDatasetInMemory, TableCellSample, SentEnc, ce_fit_iterative

    main(spec)
    

