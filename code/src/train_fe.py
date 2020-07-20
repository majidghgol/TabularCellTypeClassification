from models import FeatEnc
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
    d = spec['fe']['encdim']
    num_epochs = spec['ce']['epochs']
    lr = spec['ce']['lr']
    loss = spec['ce']['loss']
    train_size = spec['ce']['train_size']
    dev_size = spec['ce']['cv_size']
    target_p = spec['ce']['target_p']
    min_row = spec['ce']['min_row']
    min_col = spec['ce']['min_col']
    window = spec['ce']['window']
    fdim = spec['fe']['fdim']
    half_precision = False

    torch.cuda.set_device(device)
    cell_sampler = TableCellSample(target_p, min_row, min_col, window)
    
    with gzip.open(input_file) as infile:
        tables = np.array([json.loads(line) for li, line in enumerate(infile) if li < (train_size+dev_size)])

    inds = np.random.permutation(np.arange(len(tables)))
    # dev_tables = tables[inds[:dev_size]]
    train_tables = tables[inds[dev_size:]]

    train_cells = [cell_sampler.sample(t['table_array'], t['feature_array'], None) for t in tqdm(train_tables)]
    train_cells = [x for xx in train_cells for x in xx]
    # dev_cells = [cell_sampler.sample(t['table_array'], t['feature_array']) for t in tqdm(dev_tables)]
    # dev_cells = [x for xx in dev_cells for x in xx]

    ## initialize the sentence encodings
    train_set = CellDatasetInMemory(train_cells, None)
    # dev_set = CellDatasetInMemory(dev_cells, None)

    dataloader_train = DataLoader(train_set, batch_size=1000, num_workers=1)
    # dataloader_dev = DataLoader(dev_set, batch_size=1000, num_workers=10) 
    dataloader_dev = None

    fe_model = FeatEnc(fdim, d).to(device)
    if half_precision:
        fe_model = fe_model.half()
    logging.basicConfig(filename=out_path+'/train.log',level=logging.DEBUG)
    for mi, (m, train_loss, dev_loss) in enumerate(fe_fit_iterative(fe_model, lr, loss, 
                                                          dataloader_train, dataloader_dev, 
                                                          num_epochs, device, hp=half_precision)):
        logging.info(f'epoch {mi+1}, train_loss: {train_loss}, dev_loss: {dev_loss}\n')
    torch.save(m.state_dict(), out_path+f'/FE.model')
    

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
    from helpers import CellDatasetInMemory, TableCellSample, fe_fit_iterative

    main(spec)