import os
import torch
import importlib
import torch.nn as nn
from thop import profile, clever_format
from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, iteration, save, viz_split,
                xs, ys, **kwargs):

        preds, weights = self.model(*xs, **kwargs)

        loss  = self.loss(iteration,
                          save,
                          viz_split,
                          preds,
                          ys,
                          **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):
    def __init__(self, flag=False):
        super(NetworkFactory, self).__init__()

        nnet_module = importlib.import_module('models.LSTR')

        self.model   = DummyModule(nnet_module.model(flag=flag))
        self.loss    = nnet_module.loss()
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=[16])
        self.flag    = flag

    def cuda(self):
        self.model.cuda()

    def eval_mode(self):
        self.network.eval()

    def test(self, xs, **kwargs):
        with torch.no_grad():
            # xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)
