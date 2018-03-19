import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ast

np.random.seed(0)
torch.manual_seed(0)

class SluConvNet(nn.Module):

    def __init__(self, params, embedding_matrix, vocab_size, num_classes):
        super(SluConvNet, self).__init__()

        filter_sizes = ast.literal_eval(params['filter_sizes'])
        num_filters = int(params['num_filters'])
        embedding_dim = int(params['embedding_dim'])
        max_sent_len = int(params['max_sent_len'])
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
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.word_embeddings(x)
        # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
        x = x.transpose(1, 2)
        x_list = [conv_block(x) for conv_block in self.conv_blocks]

        # x_list.shape: [(batch, num_filters, filter_size_3), (batch, num_filters, filter_size_4), ...]
        out = torch.cat(x_list, 2)
        # out.shape: (batch, num_filters * len(filter_sizes))
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # fc_out.shape: (batch, num_classes)
        fc_out = self.fc(out)
        result = F.softmax(fc_out, dim=1)
        # result = F.sigmoid(fc_out)

        return result



class SluCtxConvNet(nn.Module):
    """
    Convolutional Neural Network for SLU, using context utterances

    """
    def __init__(self, params, embedding_matrix, vocab_size, num_classes):
        super(SluCtxConvNet, self).__init__()

        filter_sizes = ast.literal_eval(params['filter_sizes'])
        num_filters = int(params['num_filters'])
        embedding_dim = int(params['embedding_dim'])
        max_sent_len = int(params['max_sent_len'])
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
        self.fc = nn.Linear(2 * num_filters * len(filter_sizes), num_classes)  # normal input + context input (times 2)

    def forward(self, x, ctx_x):
        # construct a cnn for modeling context utterances
        x = self.word_embeddings(x)
        ctx_x = self.word_embeddings(ctx_x)

        # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
        x = x.transpose(1, 2)
        ctx_x = ctx_x.transpose(1, 2)

        x_list = [conv_block(x) for conv_block in self.conv_blocks]
        ctx_x_list = [conv_block(ctx_x) for conv_block in self.conv_blocks]
        # x_list.shape: [(batch, num_filters, filter_size_3), (batch, num_filters, filter_size_4), ...]
        out = torch.cat(x_list, 2)
        ctx_out = torch.cat(ctx_x_list, 2)

        out = out.view(out.size(0), -1)
        ctx_out = ctx_out.view(ctx_out.size(0), -1)
        # out.shape: (batch, num_filters * len(filter_sizes))

        # concatenate out and ctx_out
        out = torch.cat([out, ctx_out], 1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # fc_out.shape: (batch, num_classes)
        fc_out = self.fc(out)
        result = F.softmax(fc_out, dim=1)
        # result = F.sigmoid(fc_out)

        return result
    pass

class SluCtxLabelConvNet(nn.Module):
    """
    Convolutional Neural Network for SLU, using context utterances

    """
    def __init__(self, params, embedding_matrix, vocab_size, num_classes):
        super(SluCtxLabelConvNet, self).__init__()

        filter_sizes = ast.literal_eval(params['filter_sizes'])
        num_filters = int(params['num_filters'])
        embedding_dim = int(params['embedding_dim'])
        max_sent_len = int(params['max_sent_len'])
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
        self.fc = nn.Linear(num_filters * len(filter_sizes) + num_classes, num_classes)  # normal input + context input (times 2)

    def forward(self, x, ctx_label):
        # construct a cnn for modeling context utterances
        x = self.word_embeddings(x)

        # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
        x = x.transpose(1, 2)

        x_list = [conv_block(x) for conv_block in self.conv_blocks]
        # x_list.shape: [(batch, num_filters, filter_size_3), (batch, num_filters, filter_size_4), ...]
        out = torch.cat(x_list, 2)

        out = out.view(out.size(0), -1)
        # out.shape: (batch, num_filters * len(filter_sizes))

        # concatenate out and ctx_out
        out = torch.cat([out, ctx_label], 1)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # fc_out.shape: (batch, num_classes)
        fc_out = self.fc(out)
        result = F.softmax(fc_out, dim=1)
        # result = F.sigmoid(fc_out)

        return result
    pass

class SluMultitaskConvNet(nn.Module):

    def __init__(self, params, embedding_matrix, vocab_size, y_shapes):
        super(SluMultitaskConvNet, self).__init__()

        filter_sizes = ast.literal_eval(params['filter_sizes'])
        num_filters = int(params['num_filters'])
        embedding_dim = int(params['embedding_dim'])
        max_sent_len = int(params['max_sent_len'])
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
        self.fc_category = nn.Linear(num_filters * len(filter_sizes), y_shapes[0])
        self.fc_attr = nn.Linear(num_filters * len(filter_sizes), y_shapes[1])
        self.fc_sa = nn.Linear(num_filters * len(filter_sizes), y_shapes[2])

    def forward(self, x):
        x = self.word_embeddings(x)
        # x.shape: (batch, sent_len, embed_dim) --> (batch, embed_dim, sent_len)
        x = x.transpose(1, 2)
        x_list = [conv_block(x) for conv_block in self.conv_blocks]

        # x_list.shape: [(batch, num_filters, filter_size_3), (batch, num_filters, filter_size_4), ...]
        out = torch.cat(x_list, 2)
        # out.shape: (batch, num_filters * len(filter_sizes))
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=self.dropout, training=self.training)

        # fc_out.shape: (batch, num_classes)
        fc_out_category = self.fc_category(out)
        fc_out_attr = self.fc_attr(out)
        fc_out_sa = self.fc_sa(out)

        result_category = F.softmax(fc_out_category, dim=1)
        result_attr = F.softmax(fc_out_attr, dim=1)
        result_sa = F.softmax(fc_out_sa, dim=1)

        return result_category, result_attr, result_sa