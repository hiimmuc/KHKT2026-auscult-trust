import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Parameter
import math



class AngularMarginLoss(nn.Module):
    def __init__(self, hidden_dim, speaker_num, s=15, m=0.3, gamma=0.5, m_target=0.2, class_nums=None, easy_margin=False, **kwargs):
        super(AngularMarginLoss, self).__init__()
        import math
        
        self.m = m
        self.s = s
        self.speaker_num = speaker_num
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(speaker_num, hidden_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.gamma = gamma

        # -log margin
        eps=1e-12
        class_nums = torch.tensor(class_nums, dtype=torch.float64)
        total = class_nums.sum().clamp_min(eps)

        # π_y = n_y / N
        pi = (class_nums / total).clamp_min(eps)

        # x_y = -log π_y = log(N / n_y)
        self.m_target = m_target
        c = float(self.m_target / torch.log(torch.tensor(4, dtype=torch.float64)))
        # m_y = c * (-log pi_y)
        self.class_margin = -torch.log(pi)*c
        self.class_margin = self.class_margin.cuda()


    def forward(self, x_BxH, labels_B, quality_score_B):

        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num


        # cos(theta)
        cosine = F.linear(F.normalize(x_BxH), F.normalize(self.weight))

        # One hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels_B.view(-1, 1), 1)
        one_hot = one_hot.double().cuda()

        class_margin = one_hot @ self.class_margin


        quality_score_B = self.m*quality_score_B.cuda()
        margin = self.gamma*quality_score_B + (1-self.gamma)*class_margin

        # self.cos_m = math.cos(self.m)
        cos_quality_m = torch.cos(margin).unsqueeze(1).cuda()
 
        # self.sin_m = math.sin(self.m)
        sin_quality_m = torch.sin(margin).unsqueeze(1).cuda()


        # self.th = math.cos(math.pi - self.m)
        th = torch.cos(torch.pi - margin).unsqueeze(1).cuda()

        # self.mm = math.sin(math.pi - self.m) * self.m
        mm = torch.sin(torch.pi - margin)*margin
        mm = mm.unsqueeze(1).cuda()
        

        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        # cos(theta+m) = phi = cosine * self.cos_m - sine * self.sin_m
        phi = cosine * cos_quality_m - sine * sin_quality_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            phi = torch.where((cosine - th) > 0, phi, cosine - mm)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output
        '''
        loss    = self.ce(output, labels_B)
        return loss
        '''


class AngularClassifier(nn.Module):
    def __init__(self, hidden_dim, speaker_num, s=15):
        super(AngularClassifier, self).__init__()
        import math

        self.speaker_num = speaker_num
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(speaker_num, hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.s=s


    def forward(self, x_BxH):
        # cos(theta)
        cosine = F.linear(F.normalize(x_BxH), F.normalize(self.weight))
        output = cosine * self.s
        
        return output
        
    
    
