import torch
import torch.nn as nn

import random
import torch.backends.cudnn as cudnn
import numpy as np
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

class StyleModel(nn.Module):
    def __init__(self, pretrained):
        super(StyleModel, self).__init__()
        self.pretrained = pretrained
        self.FC = nn.Linear(1000, 8)

    def forward(self, x):
        
        x = self.pretrained(x)

        # 마지막 출력에 nn.Linear를 추가
        # multilabel을 예측해야 하기 때문에
        # softmax가 아닌 sigmoid를 적용
        x = torch.sigmoid(self.FC(x))
        return x

