# -*- coding: utf-8 -*-
"""
현재 화자의 발화를 분석 시
상대방의 이전 발화의 화행정보(전부/최근 한개)를 입력정보로 사용함
dynamic 방식은 아니며 사전에 입력데이터를 구성한다 (성능향상에 도움이 되는지 실험하기 위함)
"""
from __future__ import print_function
from pprint import pprint
from sklearn import preprocessing
import data_helpers
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from dataset import TensorMultiInputDataset
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import argparse, sys, time, json
import dataset_walker

from slu_model import SluCtxConvNet
from slu_model import SluConvNet
from slu_model import SluCtxLabelConvNet

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

def run_slu_task(embedding_matrix, vocabulary, label_binarizer,
                 train_inputs, train_ctx_inputs, train_labels, train_ctx_labels,
                 test_inputs, test_ctx_inputs, test_labels, test_ctx_labels):

    # load parameters
    params = data_helpers.load_params("parameters/cnn.txt")
    pprint(params)
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    multilabel = params['multilabel']=="true"

    x_train = torch.from_numpy(train_inputs).long()
    x_ctx_train = torch.from_numpy(train_ctx_labels).float()
    y_train = torch.from_numpy(train_labels).float()
    train_tensor = TensorMultiInputDataset((x_train, x_ctx_train), y_train)
    train_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False)

    x_test = torch.from_numpy(test_inputs).long()
    x_ctx_test = torch.from_numpy(test_ctx_labels).float()
    y_test = torch.from_numpy(test_labels).float()
    test_tensor = TensorMultiInputDataset((x_test, x_ctx_test), y_test)
    test_loader = data_utils.DataLoader(test_tensor, batch_size=batch_size, shuffle=False, num_workers=4,
                                         pin_memory=False)

    # load model
    # model = SluCtxConvNet(params, embedding_matrix, len(vocabulary), y_train.shape[1])
    # model = SluConvNet(params, embedding_matrix, len(vocabulary), y_train.shape[1])
    model = SluCtxLabelConvNet(params, embedding_matrix, len(vocabulary), y_train.shape[1])

    if torch.cuda.is_available():
        model = model.cuda()
    learning_rate = float(params['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.MultiLabelSoftMarginLoss()

    for epoch in range(num_epochs):
        model.train()   # set the model to training mode (apply dropout etc)
        for i, (inputs_tuple, labels) in enumerate(train_loader):
            inputs = autograd.Variable(inputs_tuple[0])
            ctx_inputs = autograd.Variable(inputs_tuple[1])
            labels = autograd.Variable(labels)

            if torch.cuda.is_available():
                inputs, ctx_inputs, labels = inputs.cuda(), ctx_inputs.cuda(), labels.cuda()

            preds = model(inputs, ctx_inputs)
            if torch.cuda.is_available():
                preds = preds.cuda()

            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("current loss: %.4f" % loss)

        model.eval()        # set the model to evaluation mode

        true_acts, pred_acts, metrics, preds = evaluate(model, label_binarizer, test_loader, y_test, multilabel)
        print("Precision: %.4f\tRecall: %.4f\tF1-score: %.4f\n" % (metrics[0], metrics[1], metrics[2]))
    pass

def evaluate(model, label_binarizer, test_loader, y_test, multilabel=False):
    preds = None
    for i, (inputs_tuple, labels) in enumerate(test_loader):
        inputs = autograd.Variable(inputs_tuple[0])
        ctx_inputs = autograd.Variable(inputs_tuple[1])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            ctx_inputs = ctx_inputs.cuda()
        preds_batch = model(inputs, ctx_inputs)

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
    threshold = 0.2
    pred_labels = np.zeros(preds.shape)
    for i, pred in enumerate(preds):
        vec = np.array([1 if p > threshold else 0 for p in pred])
        pred_labels[i] = vec

    return pred_labels

def main(argv):
    parser = argparse.ArgumentParser(description='CNN baseline for DSTC5 SAP Task')
    parser.add_argument('--trainset', dest='trainset', action='store', metavar='TRAINSET', required=True, help='')
    parser.add_argument('--testset', dest='testset', action='store', metavar='TESTSET', required=True, help='')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True, metavar='PATH',  help='')

    args = parser.parse_args()

    train_utters = []
    trainset = dataset_walker.dataset_walker(args.trainset, dataroot=args.dataroot, labels=True, translations=True)
    sys.stderr.write('Loading training instances ... ')
    for call in trainset:
        context_utters = []
        context_utter_str = '<PAD/>'
        context_labels = []
        context_label = ['INI_OPENING']
        last_speaker = None
        for (log_utter, translations, label_utter) in call:
            transcript = data_helpers.tokenize_and_lower(log_utter['transcript'])
            speech_act = label_utter['speech_act']
            sa_label_list = []
            for sa in speech_act:
                sa_label_list += ['%s_%s' % (sa['act'], attr) for attr in sa['attributes']]
            sa_label_list = sorted(set(sa_label_list))

            if last_speaker is not None and log_utter['speaker'] != last_speaker:
                if len(context_utters) > 0:
                    context_utter_str = ' <pause> '.join(context_utters)
                    context_label = context_labels[-1]
                else:
                    context_utter_str = '<PAD/>'
                    context_label = ['INI_OPENING']

                context_utters = []
                context_labels = []
                last_speaker = None

            if last_speaker is None or log_utter['speaker'] == last_speaker:
                context_utters += [transcript]  # cumulate context utters
                context_labels += [sa_label_list]

            last_speaker = log_utter['speaker']
            train_utters += [(transcript, context_utter_str, log_utter['speaker'], sa_label_list, log_utter['utter_index'], context_label)]
            # train_utters += [(transcript, context_utter_str, log_utter['speaker'], sa_label_list, log_utter['utter_index'], sa_label_list)]

    sys.stderr.write('Done\n')

    test_utters = []
    testset = dataset_walker.dataset_walker(args.testset, dataroot=args.dataroot, labels=True, translations=True)
    sys.stderr.write('Loading testing instances ... ')
    for call in testset:
        context_utters = []
        context_utter_str = '<PAD/>'
        context_labels = []
        context_label = ['INI_OPENING']
        last_speaker = None
        for (log_utter, translations, label_utter) in call:
            try:
                translation = data_helpers.tokenize_and_lower(translations['translated'][0]['hyp'])
            except:
                translation = ''

            speech_act = label_utter['speech_act']
            sa_label_list = []
            for sa in speech_act:
                sa_label_list += ['%s_%s' % (sa['act'], attr) for attr in sa['attributes']]
            sa_label_list = sorted(set(sa_label_list))

            if last_speaker is not None and log_utter['speaker'] != last_speaker:
                if len(context_utters) > 0:
                    context_utter_str = ' <pause> '.join(context_utters)
                    context_label = context_labels[-1]
                else:
                    context_utter_str = ''
                    context_label = ['INI_OPENING']

                context_utters = []
                context_labels = []
                last_speaker = None

            if last_speaker is None or log_utter['speaker'] == last_speaker:
                context_utters += [translation]  # cumulate context utters
                context_labels += [sa_label_list]

            last_speaker = log_utter['speaker']

            test_utters += [(translation, context_utter_str, log_utter['speaker'], sa_label_list, log_utter['utter_index'], context_label)]
            # test_utters += [(translation, context_utter_str, log_utter['speaker'], sa_label_list, log_utter['utter_index'], sa_label_list)]

    # pprint(train_utters[:2])
    # pprint(test_utters[:2])

    # dump_corpus(train_utters, "dstc5_train.txt")
    # dump_corpus(test_utters, "dstc5_test.txt")

    # load parameters
    params = data_helpers.load_params("parameters/cnn.txt")
    pprint(params)

    # build vocabulary
    utters = [utter[0].split(' ') for utter in train_utters]
    ctx_utters = [utter[1].split(' ') for utter in train_utters]
    print("max context utter length: %d " % max([len(ctx_utter) for ctx_utter in ctx_utters]))
    max_sent_len = int(params['max_sent_len'])
    pad_utters = data_helpers.pad_sentences(utters, max_sent_len)
    pad_ctx_utters = data_helpers.pad_sentences(ctx_utters, max_sent_len)

    vocabulary, inv_vocabulary = data_helpers.build_vocab(pad_ctx_utters)
    print("vocabulary size: %d" % len(vocabulary))

    # build input
    train_inputs = data_helpers.build_input_data(pad_utters, vocabulary)
    train_ctx_inputs = data_helpers.build_input_data(pad_ctx_utters, vocabulary)

    utters = [utter[0].split(' ') for utter in test_utters]
    ctx_utters = [utter[1].split(' ') for utter in test_utters]
    pad_utters = data_helpers.pad_sentences(utters, max_sent_len)
    pad_ctx_utters = data_helpers.pad_sentences(ctx_utters, max_sent_len)
    test_inputs = data_helpers.build_input_data(pad_utters, vocabulary)
    test_ctx_inputs = data_helpers.build_input_data(pad_ctx_utters, vocabulary)

    # build labels
    sa_train_labels = [utter[3] for utter in train_utters]
    sa_test_labels = [utter[3] for utter in test_utters]
    sa_train_ctx_labels = [utter[5] for utter in train_utters]
    sa_test_ctx_labels = [utter[5] for utter in test_utters]

    label_binarizer = preprocessing.MultiLabelBinarizer()
    label_binarizer.fit(sa_train_labels + sa_test_labels)

    train_labels = label_binarizer.transform(sa_train_labels)
    test_labels = label_binarizer.transform(sa_test_labels)
    train_ctx_labels = label_binarizer.transform(sa_train_ctx_labels)
    test_ctx_labels = label_binarizer.transform(sa_test_ctx_labels)

    # split speakers into two sets
    tourist_train_indices = [i for i, utter in enumerate(train_utters) if utter[2].lower() == 'tourist']
    guide_train_indices = [i for i, utter in enumerate(train_utters) if utter[2].lower() == 'guide']

    tourist_test_indices = [i for i, utter in enumerate(test_utters) if utter[2].lower() == 'tourist']
    guide_test_indices = [i for i, utter in enumerate(test_utters) if utter[2].lower() == 'guide']

    np.random.shuffle(tourist_train_indices)
    np.random.shuffle(guide_train_indices)

    tourist_train_inputs = train_inputs[tourist_train_indices]
    tourist_train_ctx_inputs = train_ctx_inputs[tourist_train_indices]
    tourist_train_labels = train_labels[tourist_train_indices]
    tourist_train_ctx_labels = train_ctx_labels[tourist_train_indices]

    guide_train_inputs = train_inputs[guide_train_indices]
    guide_train_ctx_inputs = train_ctx_inputs[guide_train_indices]
    guide_train_labels = train_labels[guide_train_indices]
    guide_train_ctx_labels = train_ctx_labels[guide_train_indices]

    tourist_test_inputs = test_inputs[tourist_test_indices]
    tourist_test_ctx_inputs = test_ctx_inputs[tourist_test_indices]
    tourist_test_labels = test_labels[tourist_test_indices]
    tourist_test_ctx_labels = test_ctx_labels[tourist_test_indices]

    guide_test_inputs = test_inputs[guide_test_indices]
    guide_test_ctx_inputs = test_ctx_inputs[guide_test_indices]
    guide_test_labels = test_labels[guide_test_indices]
    guide_test_ctx_labels = test_ctx_labels[guide_test_indices]

    # load pre-trained word embeddings
    embedding_dim = int(params['embedding_dim'])
    embedding_matrix = data_helpers.load_embedding(vocabulary, embedding_dim=embedding_dim,
                                                   embedding=params['embedding'])

    run_slu_task(embedding_matrix, vocabulary, label_binarizer,
                 tourist_train_inputs, tourist_train_ctx_inputs, tourist_train_labels, tourist_train_ctx_labels,
                 tourist_test_inputs, tourist_test_ctx_inputs, tourist_test_labels, tourist_test_ctx_labels)

    run_slu_task(embedding_matrix, vocabulary, label_binarizer,
                 guide_train_inputs, guide_train_ctx_inputs, guide_train_labels, guide_train_ctx_labels,
                 guide_test_inputs, guide_test_ctx_inputs, guide_test_labels, guide_test_ctx_labels)

    print("")

if __name__ == "__main__":
    main(sys.argv)
