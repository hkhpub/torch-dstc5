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
    for call in trainset:
        for (log_utter, translations, label_utter) in call:
            if log_utter['speaker'].lower() != args.roletype:
                continue
            transcript = data_helpers.tokenize_and_lower(log_utter['transcript'])

            speech_act = label_utter['speech_act']
            sa_label_list = []
            for sa in speech_act:
                sa_label_list += ['%s_%s' % (sa['act'], attr) for attr in sa['attributes']]
            sa_label_list = sorted(set(sa_label_list))
            train_utters += [(transcript, log_utter['speaker'], sa_label_list)]
    sys.stderr.write('Done\n')

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

    # load parameters
    params = data_helpers.load_params("parameters/cnn.txt")
    pprint(params)
    num_epochs = int(params['num_epochs'])
    validation_split = float(params['validation_split'])
    batch_size = int(params['batch_size'])
    multilabel = params['multilabel']=="true"

    # build vocabulary
    sents = [utter[0].split(' ') for utter in train_utters]
    max_sent_len = int(params['max_sent_len'])
    pad_sents = data_helpers.pad_sentences(sents, max_sent_len)
    vocabulary, inv_vocabulary = data_helpers.build_vocab(pad_sents)
    print("vocabulary size: %d" % len(vocabulary))
    # params['max_sent_len'] = max_sent_len

    # build inputs
    train_inputs = data_helpers.build_input_data(pad_sents, vocabulary)

    test_sents = [utter[0].split(' ') for utter in test_utters]
    test_pad_sents = data_helpers.pad_sentences(test_sents, max_sent_len)
    test_inputs = data_helpers.build_input_data(test_pad_sents, vocabulary)

    # build labels
    sa_train_labels = [utter[2] for utter in train_utters]
    sa_test_labels = [utter[2] for utter in test_utters]
    label_binarizer = preprocessing.MultiLabelBinarizer()
    label_binarizer.fit(sa_train_labels+sa_test_labels)

    train_labels = label_binarizer.transform(sa_train_labels)
    test_labels = label_binarizer.transform(sa_test_labels)

    # split and shuffle data
    indices = np.arange(train_inputs.shape[0])
    np.random.shuffle(indices)
    train_inputs = train_inputs[indices]
    train_labels = train_labels[indices]
    num_validation = int(validation_split * train_inputs.shape[0])

    # x_train = train_inputs[:-num_validation]
    # y_train = train_labels[:-num_validation]
    # x_val = train_inputs[-num_validation:]
    # y_val = train_labels[-num_validation:]
    x_train = train_inputs
    y_train = train_labels

    x_test = test_inputs
    y_test = test_labels

    # construct a pytorch data_loader
    x_train = torch.from_numpy(x_train).long()
    y_train = torch.from_numpy(y_train).float()
    dataset_tensor = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True, num_workers=4,
                                         pin_memory=False)

    x_test = torch.from_numpy(x_test).long()
    y_test = torch.from_numpy(y_test).long()
    dataset_tensor = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(dataset_tensor, batch_size=batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False)


    # load pre-trained word embeddings
    embedding_dim = int(params['embedding_dim'])
    embedding_matrix = data_helpers.load_embedding(vocabulary, embedding_dim=embedding_dim, embedding=params['embedding'])

    # load model
    model = SluConvNet(params, embedding_matrix, len(vocabulary), y_train.shape[1])

    if torch.cuda.is_available():
        model = model.cuda()
    learning_rate = float(params['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()   # set the model to training mode (apply dropout etc)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = autograd.Variable(inputs), autograd.Variable(labels)
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            preds = model(inputs)
            if torch.cuda.is_available():
                preds = preds.cuda()

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("current loss: %.4f" % loss)

        model.eval()        # set the model to evaluation mode

        true_acts, pred_acts, metrics = evaluate(model, label_binarizer, test_loader, y_test, multilabel)
        print("Precision: %.4f\tRecall: %.4f\tF1-score: %.4f\n" % (metrics[0], metrics[1], metrics[2]))

    # end of training
    true_acts, pred_acts, metrics = evaluate(model, label_binarizer, test_loader, y_test, multilabel)
    print("Precision: %.4f\tRecall: %.4f\tF1-score: %.4f\n" % (metrics[0], metrics[1], metrics[2]))

    with open(("pred_result_%s.txt" % args.roletype), "w") as f:
        for pred_act, true_act in zip(pred_acts, true_acts):
            f.write("pred: %s\ntrue: %s\n\n" % (', '.join(pred_act), ', '.join(true_act)))

# end of main

def evaluate(model, label_binarizer, test_loader, y_test, multilabel=False):
    preds = None
    for i, (x_batch, _) in enumerate(test_loader):
        inputs = autograd.Variable(x_batch)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        preds_batch = model(inputs)

        preds_batch = preds_batch.cpu().data.numpy()
        if preds is None:
            preds = preds_batch
        else:
            preds = np.concatenate((preds, preds_batch), axis=0)    # merge along batch axis

    if multilabel:
        pred_labels = predict_multilabel(preds)
    else:
        pred_labels = predict_onelabel(preds)      # multiclass

    pred_acts = label_binarizer.inverse_transform(pred_labels)
    true_acts = label_binarizer.inverse_transform(y_test)

    # calculate F1-measure
    pred_cnt = pred_correct_cnt = answer_cnt = 0
    for pred_act, true_act in zip(pred_acts, true_acts):
        pred_cnt += len(pred_act)
        answer_cnt += len(true_act)
        pred_correct_cnt += len([act for act in pred_act if act in true_act])

    P = pred_correct_cnt * 1.0 / pred_cnt
    R = pred_correct_cnt * 1.0 / answer_cnt
    F = 2*P*R / (P+R)
    metrics = (P, R, F)

    return true_acts, pred_acts, metrics

def predict_onelabel(preds):
    pred_labels = np.zeros(preds.shape)
    preds = np.argmax(preds, axis=1)
    for i, label_index in enumerate(preds):
        pred_labels[i][label_index] = 1

    return pred_labels

def predict_multilabel(preds):
    threshold = 0.2
    pred_labels = np.zeros(preds.shape)
    for i, pred in enumerate(preds):
        vec = np.array([1 if p > threshold else 0 for p in pred])
        pred_labels[i] = vec

    return pred_labels

if __name__ == "__main__":
    main(sys.argv)
