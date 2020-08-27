# coding:utf-8
from torch.utils.data import Dataset


class HandleData(Dataset):
    def __init__(self, data_path):
        super(HandleData, self).__init__()
        self.data_path = data_path
