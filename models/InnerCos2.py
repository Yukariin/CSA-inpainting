import torch.nn as nn
import torch
from torch.autograd import Variable

import util.util as util


class InnerCos2(nn.Module):
    def __init__(self, crit='MSE', strength=1, skip=0):
        super(InnerCos2, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == 'MSE' else torch.nn.L1Loss()

        self.strength = strength
        self.target = None
        # To define whether this layer is skipped.
        self.skip = skip

    def set_mask(self, mask_global, opt):
        mask = util.cal_feat_mask(mask_global, 3, opt.threshold)
        self.mask = mask.squeeze()
        if torch.cuda.is_available:
            self.mask = self.mask.float().cuda()
        self.mask = Variable(self.mask, requires_grad=False)

    def set_target(self, targetIn):
        self.target = targetIn


    def get_target(self):
        return self.target

    def forward(self, in_data):
        self.output = in_data
        if not self.skip:
            self.former = in_data.narrow(1,0,512)
            self.former_in_mask = torch.mul(self.former, self.mask)
            self.loss = self.criterion(self.former_in_mask * self.strength, self.target)
        else:
            self.loss = 0
        return self.output


    def backward(self, retain_graph=True):
        if not self.skip:
            self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def __repr__(self):
        skip_str = 'True' if self.skip else 'False'
        return self.__class__.__name__+ '(' \
              + 'skip=' + skip_str \
              + ', strength=' + str(self.strength) + ')'
