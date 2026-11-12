import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=768, apply_bn=True):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        if apply_bn:
            self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.projector = nn.Sequential(self.linear1, self.relu, self.linear2)

    def forward(self, x):
        return self.projector(x)



class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=None, output_dim=None, attention=False,norm_type='bn'):
        super().__init__()
        self.attention = attention
        self.linear1 = nn.Linear(input_dim, input_dim,)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        if hidden_dim is None:
            hidden_dim = input_dim
            
        if attention:
            self.temporal_attn = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # MLP projection with optional hidden layer
        if hidden_dim is not None:
            if norm_type == 'bn':
               self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
            elif norm_type == 'ln':
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
        else:
            if norm_type == 'bn':
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim)
                )
            elif norm_type == 'ln':
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.ReLU(),
                    nn.Linear(output_dim, output_dim)
                )
        # self._init_weights()         
    def _init_weights(self):
        # linear1 → followed by BN → followed by ReLU
        # Use Kaiming because ReLU comes after
        nn.init.kaiming_uniform_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.linear1.bias, 0.0)
        
        # linear2 → final projection, nothing after
        # Use Xavier because no nonlinearity follows
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)
        
        # BatchNorm → weight=1, bias=0 (neutral start)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)      
    def forward(self, x):
        # Handle gradient options first
        # if self.proj_type == 'feat_fixed':
        #     x = x.detach()
            
        if self.attention:
            # Temporal attention weights
            attn_weights = self.temporal_attn(x)
            x = torch.sum(x * attn_weights, dim=1)
        
        # Project features
        x = self.projector(x)
        
        # Handle proj_fixed option
        # if self.proj_type == 'proj_fixed':
        #     x = x.detach()
            
        return x
        
