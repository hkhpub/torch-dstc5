import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
import ast

np.random.seed(0)
torch.manual_seed(0)

class SluCnnLstm(nn.Module):

    def __init__(self, params, embedding_matrix, vocab_size, num_classes):
        super(SluCnnLstm, self).__init__()

        filter_sizes = ast.literal_eval(params['filter_sizes'])
        num_filters = int(params['num_filters'])
        embedding_dim = int(params['embedding_dim'])
        max_sent_len = int(params['max_sent_len'])
        batch_size = int(params['batch_size'])
        self.hidden_dim = int(params['hidden_dim'])
        self.dropout = float(params['dropout'])

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        # self.word_embeddings.weight = nn.Parameter(torch.from_numpy(embedding_matrix))
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # self.word_embeddings.weight.requires_grad = True

        conv_blocks = []
        for filter_size in filter_sizes:
            maxpool_size = max_sent_len - filter_size + 1
            conv1 = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=num_filters,
                              kernel_size=filter_size)
            component = nn.Sequential(conv1, nn.ReLU(), nn.MaxPool1d(maxpool_size))
            # conv1_bn = nn.BatchNorm1d(num_features=num_filters)
            # # apply batch norm
            # component = nn.Sequential(conv1, conv1_bn, nn.ReLU(), nn.MaxPool1d(maxpool_size))
            if torch.cuda.is_available():
                component = component.cuda()

            conv_blocks += [component]
        # end of for loop
        self.conv_blocks = nn.ModuleList(conv_blocks)
        # self.lstm = nn.LSTM(num_filters * len(filter_sizes), self.hidden_dim)
        self.lstm = nn.LSTM(num_filters * len(filter_sizes), self.hidden_dim, dropout=self.dropout)
        self.hidden = self.init_hidden(batch_size)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def init_hidden(self, batch_size):
        # (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda())

    def forward(self, x):
        x = x.transpose(0, 1)   # (batch, ctx_len, sent_len) --> (ctx_len, batch, sent_len)
        sent_vectors = []
        for i in range(x.size()[0]):
            sent_x = x[i]
            sent_x = self.word_embeddings(sent_x)
            # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
            sent_x = sent_x.transpose(1, 2)
            x_list = [conv_block(sent_x) for conv_block in self.conv_blocks]

            # x_list.shape: [(batch, num_filters, filter_size_3), (batch, num_filters, filter_size_4), ...]
            sent_out = torch.cat(x_list, 2)

            # out.shape: (batch, num_filters * len(filter_sizes))
            sent_out = sent_out.view(sent_out.size(0), -1)

            # fc_out.shape: (batch, num_classes)
            sent_vectors += [sent_out]

        sent_vectors = torch.stack(sent_vectors)        # this is sentence vectors which size is context_length

        # use lstm here
        lstm_out, self.hidden = self.lstm(sent_vectors, self.hidden)
        out = F.dropout(lstm_out[-1], p=self.dropout, training=self.training)
        out = self.fc(out)
        result = F.softmax(out, dim=1)

        return result

class SluMultitaskCnnLstm(nn.Module):

    def __init__(self, params, embedding_matrix, vocab_size, y_shapes):
        super(SluMultitaskCnnLstm, self).__init__()

        filter_sizes = ast.literal_eval(params['filter_sizes'])
        num_filters = int(params['num_filters'])
        embedding_dim = int(params['embedding_dim'])
        max_sent_len = int(params['max_sent_len'])
        batch_size = int(params['batch_size'])
        self.hidden_dim = int(params['hidden_dim'])
        self.dropout = float(params['dropout'])

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        # self.word_embeddings.weight = nn.Parameter(torch.from_numpy(embedding_matrix))
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # self.word_embeddings.weight.requires_grad = True

        conv_blocks = []
        for filter_size in filter_sizes:
            maxpool_size = max_sent_len - filter_size + 1
            conv1 = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=num_filters,
                              kernel_size=filter_size)
            component = nn.Sequential(conv1, nn.ReLU(), nn.MaxPool1d(maxpool_size))
            # conv1_bn = nn.BatchNorm1d(num_features=num_filters)
            # # apply batch norm
            # component = nn.Sequential(conv1, conv1_bn, nn.ReLU(), nn.MaxPool1d(maxpool_size))
            if torch.cuda.is_available():
                component = component.cuda()

            conv_blocks += [component]
        # end of for loop
        self.conv_blocks = nn.ModuleList(conv_blocks)
        # self.lstm = nn.LSTM(num_filters * len(filter_sizes), self.hidden_dim)
        self.lstm = nn.LSTM(num_filters * len(filter_sizes), self.hidden_dim, dropout=self.dropout)
        self.hidden = self.init_hidden(batch_size)

        self.fc_category = nn.Linear(self.hidden_dim, y_shapes[0])
        self.fc_attr = nn.Linear(self.hidden_dim, y_shapes[1])
        self.fc_sa = nn.Linear(self.hidden_dim, y_shapes[2])

    def init_hidden(self, batch_size):
        # (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda())

    def forward(self, x):
        x = x.transpose(0, 1)   # (batch, ctx_len, sent_len) --> (ctx_len, batch, sent_len)
        sent_vectors = []
        for i in range(x.size()[0]):
            sent_x = x[i]
            sent_x = self.word_embeddings(sent_x)
            # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
            sent_x = sent_x.transpose(1, 2)
            x_list = [conv_block(sent_x) for conv_block in self.conv_blocks]

            # x_list.shape: [(batch, num_filters, filter_size_3), (batch, num_filters, filter_size_4), ...]
            sent_out = torch.cat(x_list, 2)

            # out.shape: (batch, num_filters * len(filter_sizes))
            sent_out = sent_out.view(sent_out.size(0), -1)

            # fc_out.shape: (batch, num_classes)
            sent_vectors += [sent_out]

        sent_vectors = torch.stack(sent_vectors)        # this is sentence vectors which size is context_length

        # use lstm here
        lstm_out, self.hidden = self.lstm(sent_vectors, self.hidden)
        out = F.dropout(lstm_out[-1], p=self.dropout, training=self.training)

        # fc_out.shape: (batch, num_classes)
        fc_out_category = self.fc_category(out)
        fc_out_attr = self.fc_attr(out)
        fc_out_sa = self.fc_sa(out)

        result_category = F.softmax(fc_out_category, dim=1)
        result_attr = F.softmax(fc_out_attr, dim=1)
        result_sa = F.softmax(fc_out_sa, dim=1)

        return result_category, result_attr, result_sa