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

from slu_sequence_model import SluCnnLstm

np.random.seed(0)
torch.manual_seed(0)

def dump_corpus(train_utters, filenm):
    print("dumping corpus to file %s..." % filenm)
    with open(filenm, "w") as f:
        for train_utter in train_utters:
            utter_text = train_utter[0]
            speaker = train_utter[2]
            label_str = ' | '.join(train_utter[3])
            utter_index = train_utter[4]

            f.write("[%d] (%s)\n" % (utter_index, speaker))
            f.write("%s\n" % utter_text)
            f.write("%s\n\n" % label_str)
    print("Done!")
    pass

def run_slu_sequence_task(embedding_matrix, vocabulary, label_binarizer,
                          train_inputs, train_labels,
                          test_inputs, test_labels):

    # load parameters
    params = data_helpers.load_params("parameters/cnn.txt")
    pprint(params)
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    multilabel = params['multilabel']=="true"

    x_train = torch.from_numpy(train_inputs).long()
    y_train = torch.from_numpy(train_labels).float()
    train_tensor = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False)

    x_test = torch.from_numpy(test_inputs).long()
    y_test = torch.from_numpy(test_labels).float()
    test_tensor = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test_tensor, batch_size=batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False)

    # load model
    model = SluCnnLstm(params, embedding_matrix, len(vocabulary), y_train.shape[1])
    if torch.cuda.is_available():
        model = model.cuda()
    learning_rate = float(params['learning_rate'])
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    optimizer = optim.Adadelta(model.parameters())

    loss_fn = nn.MultiLabelSoftMarginLoss()

    for epoch in range(num_epochs):
        model.train()   # set the model to training mode (apply dropout etc)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = autograd.Variable(inputs)
            labels = autograd.Variable(labels)

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            model.hidden = model.init_hidden(inputs.size()[0])

            preds = model(inputs)
            if torch.cuda.is_available():
                preds = preds.cuda()

            loss = loss_fn(preds, labels)

            # if i % 100 == 0:
            #     print("current loss: %.4f" % loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()        # set the model to evaluation mode

        true_acts, pred_acts, metrics, preds = evaluate(model, label_binarizer, test_loader, y_test, multilabel)
        print("Precision: %.4f\tRecall: %.4f\tF1-score: %.4f\n" % (metrics[0], metrics[1], metrics[2]))
    pass

def evaluate(model, label_binarizer, test_loader, y_test, multilabel=False):
    preds = None
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = autograd.Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        model.zero_grad()
        model.hidden = model.init_hidden(inputs.size()[0])

        preds_batch = model(inputs)

        preds_batch = preds_batch.cpu().data.numpy()
        if preds is None:
            preds = preds_batch
        else:
            preds = np.concatenate((preds, preds_batch), axis=0)            # merge along batch axis

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

    return true_acts, pred_acts, metrics, preds

def predict_onelabel(preds):
    pred_labels = np.zeros(preds.shape)
    preds = np.argmax(preds, axis=1)
    for i, label_index in enumerate(preds):
        pred_labels[i][label_index] = 1

    return pred_labels

def predict_multilabel(preds):
    threshold = 0.3
    pred_labels = np.zeros(preds.shape)
    for i, pred in enumerate(preds):
        vec = np.array([1 if p > threshold else 0 for p in pred])
        pred_labels[i] = vec

    return pred_labels

def main(argv):
    parser = argparse.ArgumentParser(description='CNN baseline for DSTC5 SAP Task')
    parser.add_argument('--trainset', dest='trainset', action='store', metavar='TRAINSET', required=True, help='')
    parser.add_argument('--devset', dest='devset', action='store', metavar='DEVSET', required=True, help='')
    parser.add_argument('--testset', dest='testset', action='store', metavar='TESTSET', required=True, help='')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True, metavar='PATH',  help='')

    args = parser.parse_args()

    # load parameters
    params = data_helpers.load_params("parameters/cnn.txt")
    pprint(params)
    ctx_len = int(params['context_length'])

    trainset = dataset_walker.dataset_walker(args.trainset, dataroot=args.dataroot, labels=True, translations=True)
    devset = dataset_walker.dataset_walker(args.devset, dataroot=args.dataroot, labels=True, translations=True)
    testset = dataset_walker.dataset_walker(args.testset, dataroot=args.dataroot, labels=True, translations=True)
    train_utters, dev_utters, test_utters = data_helpers.load_dstc5_dataset(trainset, devset, testset)

    train_utters += dev_utters

    # pprint(train_utters[:2])
    # pprint(test_utters[:2])

    # dump_corpus(train_utters, "dstc5_train.txt")
    # dump_corpus(test_utters, "dstc5_test.txt")

    # build vocabulary
    utters = [[char for char in utter[0]] for utter in train_utters]
    max_sent_len = int(params['max_sent_len'])
    pad_utters = data_helpers.pad_sentences(utters, max_sent_len)

    vocabulary, inv_vocabulary = data_helpers.build_vocab(pad_utters)
    print("vocabulary size: %d" % len(vocabulary))

    # build input
    train_inputs = data_helpers.build_input_data(pad_utters, vocabulary)

    utters = [[char for char in utter[0]] for utter in test_utters]
    pad_utters = data_helpers.pad_sentences(utters, max_sent_len)
    test_inputs = data_helpers.build_input_data(pad_utters, vocabulary)

    # make windowed input data as context
    train_inputs = data_helpers.build_windowed_input(train_inputs, ctx_len)
    test_inputs = data_helpers.build_windowed_input(test_inputs, ctx_len)

    # build labels
    sa_train_labels = [utter[2] for utter in train_utters]
    sa_test_labels = [utter[2] for utter in test_utters]
    label_binarizer = preprocessing.MultiLabelBinarizer()
    label_binarizer.fit(sa_train_labels + sa_test_labels)

    train_labels = label_binarizer.transform(sa_train_labels)
    test_labels = label_binarizer.transform(sa_test_labels)

    # split speakers into two sets
    tourist_train_indices = [i for i, utter in enumerate(train_utters) if utter[1].lower() == 'tourist']
    guide_train_indices = [i for i, utter in enumerate(train_utters) if utter[1].lower() == 'guide']

    tourist_test_indices = [i for i, utter in enumerate(test_utters) if utter[1].lower() == 'tourist']
    guide_test_indices = [i for i, utter in enumerate(test_utters) if utter[1].lower() == 'guide']

    np.random.shuffle(tourist_train_indices)
    np.random.shuffle(guide_train_indices)
    # np.random.shuffle(tourist_test_indices)
    # np.random.shuffle(guide_test_indices)

    tourist_train_inputs = train_inputs[tourist_train_indices]
    tourist_train_labels = train_labels[tourist_train_indices]

    guide_train_inputs = train_inputs[guide_train_indices]
    guide_train_labels = train_labels[guide_train_indices]

    tourist_test_inputs = test_inputs[tourist_test_indices]
    tourist_test_labels = test_labels[tourist_test_indices]

    guide_test_inputs = test_inputs[guide_test_indices]
    guide_test_labels = test_labels[guide_test_indices]

    # load pre-trained word embeddings
    embedding_dim = int(params['embedding_dim'])
    embedding_matrix = data_helpers.load_embedding(vocabulary, embedding_dim=embedding_dim,
                                                   embedding=params['embedding'])

    run_slu_sequence_task(embedding_matrix, vocabulary, label_binarizer,
                          tourist_train_inputs, tourist_train_labels,
                          tourist_test_inputs, tourist_test_labels)

    run_slu_sequence_task(embedding_matrix, vocabulary, label_binarizer,
                          guide_train_inputs, guide_train_labels,
                          guide_test_inputs, guide_test_labels)

if __name__ == "__main__":
    main(sys.argv)
