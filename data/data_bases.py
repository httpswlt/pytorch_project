# coding:utf-8
from torch.utils.data import Dataset


class DataBases(Dataset):
    def __init__(self, data_path):
        super(DataBases, self).__init__()
        self.data_path = data_path
