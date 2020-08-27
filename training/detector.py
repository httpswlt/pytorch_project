# coding:utf-8
from training import Base


class Detector(Base):
    def __init__(self):
        super(Detector, self).__init__()
        self.header = None
        self.anchors = None

    def train(self, *inputs):
        raise NotImplementedError

    def set_header(self, header):
        """

        :param header:
        :return:
        """
        self.header = header

    def set_hyper_parameters(self, *hyper_parameters):
        raise NotImplementedError

    def load_weights(self, *any):
        pass
