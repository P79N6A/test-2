import torch.nn as nn
import torch
import torch.nn.functional as F
import util.util as util
import types
class InnerCos_tmp(nn.Module):
    def __init__(self, crit='MSE', strength=1, skip=0):
        super(InnerCos_tmp, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == 'MSE' else torch.nn.L1Loss()
        self.strength = strength
        # To define whether this layer is skipped.
        self.skip = skip


    def set_mask(self, mask_global, opt):
        print('1 in InnerCos')
        mask = util.cal_feat_mask(mask_global, 3, opt.threshold)
        self.mask = mask.squeeze()
        if torch.cuda.is_available:
            self.mask = self.mask.float().cuda()

    def forward(self, in_data):
        print('2 in InnerCos')
        print(in_data)
        self.bs, self.c, _, _ = in_data.size()
        self.output = in_data
        print('3 in InnerCos')
        return self.output

    def __repr__(self):
        skip_str = 'True' if not self.skip else 'False'
        return self.__class__.__name__+ '(' \
              + 'skip: ' + skip_str \
              + ' ,strength: ' + str(self.strength) + ')'
