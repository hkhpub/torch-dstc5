# -*- coding: utf-8 -*-
from __future__ import print_function
from pprint import pprint
from sklearn import preprocessing
import data_helpers
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import argparse, sys, time, json
import dataset_walker

from slu_model import SluConvNet

np.random.seed(0)
torch.manual_seed(0)

def main(argv):
    parser = argparse.ArgumentParser(description='CNN baseline for DSTC5 SAP Task')
    parser.add_argument('--trainset', dest='trainset', action='store', metavar='TRAINSET', required=True, help='')
    parser.add_argument('--testset', dest='testset', action='store', metavar='TESTSET', required=True, help='')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True, metavar='PATH',  help='')
    parser.add_argument('--roletype', dest='roletype', action='store', choices=['guide',  'tourist'], required=True,  help='speaker')

    args = parser.parse_args()

    train_utters = []
    trainset = dataset_walker.dataset_walker(args.trainset, dataroot=args.dataroot, labels=True, translations=True)
    sys.stderr.write('Loading training instances ... ')

    last_speaker = args.roletype
    last_sa_label_str = None
    total = 0
    same = 0
    multilabel_utter_cnt = 0
    utter_cnt = 0

    for call in trainset:
        for (log_utter, translations, label_utter) in call:
            if log_utter['speaker'].lower() != args.roletype:
                last_sa_label_str = None
                pass
            else:
                transcript = data_helpers.tokenize_and_lower(log_utter['transcript'])
                speech_act = label_utter['speech_act']
                sa_label_list = []
                for sa in speech_act:
                    sa_label_list += ['%s_%s' % (sa['act'], attr) for attr in sa['attributes']]

                if len(sa_label_list) > 1:
                    multilabel_utter_cnt += 1
                utter_cnt += 1

                sa_label_str = '|'.join(sa_label_list)
                if log_utter['speaker'] == last_speaker:
                    total += 1
                    if last_sa_label_str is None or sa_label_str == last_sa_label_str:
                        same += 1
                    else:
                        # print("")
                        pass
                # sa_label_list = sorted(set(sa_label_list))
                # train_utters += [(transcript, log_utter['speaker'], sa_label_list)]

                last_sa_label_str = sa_label_str
            last_speaker = log_utter['speaker']
    sys.stderr.write('Done\n')

    print("same/total=ratio: %d/%d=%.4f" % (same, total, 1.0*same/total))
    print("multi_label/total=ratio: %d/%d=%.4f" % (multilabel_utter_cnt, utter_cnt, (1.0*multilabel_utter_cnt/utter_cnt)))

    test_utters = []
    testset = dataset_walker.dataset_walker(args.testset, dataroot=args.dataroot, labels=True, translations=True)
    sys.stderr.write('Loading testing instances ... ')
    for call in testset:
        for (log_utter, translations, label_utter) in call:
            if log_utter['speaker'].lower() != args.roletype:
                continue
            try:
                translation = data_helpers.tokenize_and_lower(translations['translated'][0]['hyp'])
            except:
                translation = ''

            speech_act = label_utter['speech_act']
            sa_label_list = []
            for sa in speech_act:
                sa_label_list += ['%s_%s' % (sa['act'], attr) for attr in sa['attributes']]
            sa_label_list = sorted(set(sa_label_list))
            test_utters += [(translation, log_utter['speaker'], sa_label_list)]

    pprint(train_utters[:2])
    pprint(test_utters[:2])



if __name__ == "__main__":
    main(sys.argv)
