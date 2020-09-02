# coding:utf-8
from training import BaseTraining


class Detector(BaseTraining):
    def __init__(self):
        super(Detector, self).__init__()
        self.header = None
        self.anchors = None
        self.backbone = None
    
    def set_backbone(self, backbone):
        self.backbone = backbone
    
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
