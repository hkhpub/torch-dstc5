import torch.utils.data as data_utils

class TensorMultiInputDataset(data_utils.Dataset):

    def __init__(self, data_tensor_tuple, target_tensor):
        assert len(data_tensor_tuple) > 0
        assert data_tensor_tuple[0].size(0) == target_tensor.size(0)
        self.data_tensor_tuple = data_tensor_tuple
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        data_tuple = tuple([tup[index] for tup in self.data_tensor_tuple])
        return data_tuple, self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)

class TensorMultiTargetDataset(data_utils.Dataset):

    def __init__(self, data_tensor, target_tensor_tuple):
        assert len(target_tensor_tuple) > 0
        assert data_tensor.size(0) == target_tensor_tuple[0].size(0)
        self.data_tensor = data_tensor
        self.target_tensor_tuple = target_tensor_tuple

    def __getitem__(self, index):
        taget_tuple = tuple([tup[index] for tup in self.target_tensor_tuple])
        return self.data_tensor[index], taget_tuple

    def __len__(self):
        return self.data_tensor.size(0)