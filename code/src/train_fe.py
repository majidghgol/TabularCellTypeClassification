from models import FeatEnc
from helpers import CellDatasetInMemory, TableCellSample, fe_fit_iterative
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
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train_size', type=int, default=5000000)
    # parser.add_argument('--cv_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--encdim', type=int, default=512)
    parser.add_argument('--fdim', type=int, default=43)
    

    FLAGS, unparsed = parser.parse_known_args()

    np.random.seed(12345)
    torch.manual_seed(12345)

    os.environ["OMP_NUM_THREADS"] = str(FLAGS.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(FLAGS.threads)
    os.environ["MKL_NUM_THREADS"] = str(FLAGS.threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(FLAGS.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(FLAGS.threads)

    device = FLAGS.device
    out_path = FLAGS.out_path
    d = FLAGS.encdim
    fdim = FLAGS.fdim
    num_epochs = FLAGS.epochs
    lr = FLAGS.lr
    loss = FLAGS.loss
    train_size = FLAGS.train_size
    # dev_size = FLAGS.cv_size
    dev_size = 0
    target_p = 1.0
    min_row = 3
    min_col = 2
    window = 0
    half_precision = False

    torch.cuda.set_device(device)

    cell_sampler = TableCellSample(target_p, min_row, min_col, window)
    
    with gzip.open(FLAGS.input_file) as infile:
        tables = np.array([json.loads(line) for li, line in enumerate(infile) if li < (train_size+dev_size)])

    inds = np.random.permutation(np.arange(len(tables)))
    # dev_tables = tables[inds[:dev_size]]
    train_tables = tables[inds[dev_size:]]

    train_cells = [cell_sampler.sample(t['table_array'], t['feature_array']) for t in tqdm(train_tables)]
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
        fe_model = ce_model.half()
    logging.basicConfig(filename=out_path+'/train.log',level=logging.DEBUG)
    for mi, (m, train_loss, dev_loss) in enumerate(fe_fit_iterative(fe_model, lr, loss, 
                                                          dataloader_train, dataloader_dev, 
                                                          num_epochs, device, hp=half_precision)):
        logging.info(f'epoch {mi+1}, train_loss: {train_loss}, dev_loss: {dev_loss}\n')
        torch.save(m.state_dict(), out_path+f'/CE_epoch{mi}.model')
    

