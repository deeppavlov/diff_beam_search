from __future__ import print_function

import re

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import time
import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import chain, islice
import argparse, os, sys

from util import read_corpus, data_iter, batch_slice
from vocab import Vocab, VocabEntry
from process_samples import generate_hamming_distance_payoff_distribution
import math
from expected_bleu.modules.expectedMultiBleu import bleu, bleu_with_bp
from expected_bleu.TF_GOOGLE_NMT import compute_bleu
from expected_bleu.modules.utils import bleu_score, reinforce_bleu
from os import listdir
from os.path import isfile, join
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
from nmt import NMT
from multiprocessing import Process, Manager
import torch.multiprocessing as mp
import pickle

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

# def get_trailing_number(s):
#     m = re.search(r'\d+$', s)
#     return int(m.group()) if m else None

def get_trailing_number(file_name):
    parse = re.search(r'.*iter([0-9]*)\.', file_name)
    try:
        return int(parse.group(1))
    except:
        print('trailing number computation fails', file_name)
        return None

def get_gpu_number(file_name):
    parse = re.search(r'.*_([0-9]*)\.iter.*', file_name)
    try:
        return int(parse.group(1))
    except:
        print(file_name)
        print('gpu number computation fails', file_name)
        return None

def decode(args, model, data, verbose=True):
    """
    decode the dataset and compute sentence level acc. and BLEU.
    """
    hypotheses = []
    begin_time = time.time()

    if type(data[0]) is tuple:
        for src_sent, tgt_sent in data:
            hyps = model.translate(src_sent, beam_size=args.beam_size, args_new=args)
            hypotheses.append(hyps)

            if verbose:
                print('*' * 50)
                print('Source: ', ' '.join(src_sent))
                print('Target: ', ' '.join(tgt_sent))
                print('Top Hypothesis: ', ' '.join(hyps[0]))
    else:
        for src_sent in data:
            hyps = model.translate(src_sent, beam_size=args.beam_size, args_new=args)
            hypotheses.append(hyps)

            if verbose:
                print('*' * 50)
                print('Source: ', ' '.join(src_sent))
                print('Top Hypothesis: ', ' '.join(hyps[0]))

    elapsed = time.time() - begin_time

    print('decoded %d examples, took %d s' % (len(data), elapsed), file=sys.stderr)

    return hypotheses

def model_validation(args, model_path, step, train_data, dev_data, return_dict):
    print('-' * 10 + str(step) + '-' * 10)
    print('loading model')
    # import torch
    params = torch.load(model_path, map_location=lambda storage, loc: storage)
    vocab = params['vocab']
    saved_args = params['args']
    state_dict = params['state_dict']

    model = NMT(saved_args, vocab)
    model.load_state_dict(state_dict)
    model.train(mode=False)
    if args.cuda:
        model = model.cuda()
    def _validate(model, data):
        print('started decoding...')
        hyps = decode(args, model, data, verbose=False)
        hyps = [h[0] for h in hyps]
        print('computing bleu...')
        metric = compute_bleu([[tgt] for src, tgt in data], hyps)[0]
        return metric
    train_metric = None#_validate(model, train_data)
    dev_metric = _validate(model, dev_data)
    return_dict[step] = (train_metric, dev_metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_src', type=str, default="data/train.de-en.de.wmixerprep", help='path to the training source file')
    parser.add_argument('--train_tgt', type=str, default="data/train.de-en.en.wmixerprep", help='path to the training target file')
    parser.add_argument('--dev_src', type=str,default="data/valid.de-en.de", help='path to the dev source file')
    parser.add_argument('--dev_tgt', type=str, default="data/valid.de-en.en", help='path to the dev target file')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    # parser.add_argument('--test_src', type=str, help='path to the test source file')
    # parser.add_argument('--test_tgt', type=str, help='path to the test target file')
    parser.add_argument('--model_dir', default=None, type=str, help='path to models')
    parser.add_argument('--cuda', action='store_true', default=True, help='use gpu')
    parser.add_argument('--decode_max_time_step', default=200, type=int, help='maximum number of time steps used '
                                                                              'in decoding and sampling')
    parser.add_argument('--exp_name', type=str, default="", help='name of experiment')
    parser.add_argument('--bucket_size', default=None, type=int, help='bucket size of processes')
    parser.add_argument('--gpu_id', default=None, type=int, help='gpu_id_to_parse_in_filename')

    args = parser.parse_args()

    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    # model
    manager = Manager()
    return_dict = manager.dict()
    model_dir = args.model_dir
    models_pathes = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]
    processes = []
    def _joiner(processes):
        for p in processes:
            p.join()
    for i in models_pathes:
        step = get_trailing_number(i)
        if step is None:
            continue
        if get_gpu_number(i) != args.gpu_id:
            continue
        p = mp.Process(target=model_validation, args=(args,\
                                    join(model_dir, i), step,\
                                    train_data, dev_data, return_dict))
        processes.append(p)
        p.start()
        if args.bucket_size and len(processes) % args.bucket_size == 0:
            _joiner(processes)
            processes = []
    _joiner(processes)
    print('dumping plotting data')
    print(return_dict)
    print('...')
    casted_return_dict = {key: val for key, val in return_dict.items()}
    with open(join(model_dir, args.exp_name +\
                    ("gpu_id" + str(args.gpu_id) if args.gpu_id else "") +\
                                     "plotting_data.pickle"), 'wb') as f:
        pickle.dump(casted_return_dict, f)
