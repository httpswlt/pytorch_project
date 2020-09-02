# coding:utf-8
import logging
import torch

logging.getLogger().setLevel(logging.INFO)


class BaseTraining:
    """
        the base class of all training class
    """
    
    def __init__(self):
        self.optimizer = None
        self.criteria = None
        self.epochs = None
        self.lr = None
        self.lr_scheduler = None
        self.dataloader = None
        self.cuda = True
    
    def train(self, *inputs):
        raise NotImplementedError
    
    def set_cuda(self, cuda):
        self.cuda = cuda
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_criteria(self, criteria):
        self.criteria = criteria
    
    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler
    
    def __call__(self, *inputs):
        self.train(*inputs)
    
    def set_hyper_parameters(self, *hyper_parameters):
        pass
    
    def get_lr(self):
        return self.optimizer.param_groups[0].get('lr')
    
    def __str__(self):
        messages = ''
        dic = sorted(self.__dict__.items(), key=lambda x: x[0])
        for key, value in dic:
            if value is None:
                continue
            messages += "{}: {}\n".format(key, value) + "=" * 66 + '\n'
        return messages
    
    def load_weights(self, *inputs):
        pass
    
    @staticmethod
    def get_device_info(cuda=True, device_ids=None):
        """

        :param cuda:    whether enable cuda
        :param device_ids:  which device do you want to use. example:(1, 2, ...)
        :return:
        """
        flag = True if cuda and torch.cuda.is_available() else False
        device = torch.device('cuda:0' if flag else 'cpu')
        if not flag:
            device = torch.device('cpu')
            logging.info('Using CPU')
        
        if flag:
            c = 1024 ** 2  # bytes to MB
            if device_ids:
                ng = len(device_ids)
            else:
                ng = torch.cuda.device_count()
                device_ids = range(ng)
            device = torch.device('cuda:{}'.format(device_ids[0]))
            if ng != 0:
                x = [torch.cuda.get_device_properties(i) for i in device_ids]
                for i in range(ng):
                    logging.info("device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                                 (device_ids[i], x[i].name, x[i].total_memory / c))
        return device
    
    @staticmethod
    def get_all_children_modules(modules, save_type=None):
        """

        :param modules: all the torch's modules.
        :param save_type:   the type should be list, e.g. ['conv', 'batchnorm', 'activation',...]
        :return:
        """
        if save_type is None:
            save_type = ['conv', 'batchnorm', 'activation']
        results = []
        for module in modules:
            flag = 0
            for tp in save_type:
                if tp in str(type(module)):
                    flag += 1
                    break
            
            if flag:
                results.append(module)
        return results
