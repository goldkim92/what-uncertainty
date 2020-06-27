import torch
import torch.nn as nn

import models


class ResNet50_modified(nn.Module):
    def __init__(self, n_classes, mc):
        super(ResNet50_modified, self).__init__()
        self.n_classes = n_classes
        self.mc = mc
        
        # get the resnet network
        self.model = models.ResNet50()
        
        # dissect the network
        self.features_conv = nn.Sequential(*list(self.model.children())[:-1])
        
        # add the classfiier (with aleotoric design)
        self.classifier = nn.Linear(in_features=2048, out_features=n_classes*2, bias=True)
        
        # delete model variable
        del self.model
        
    def get_logprob(self, mu, rho):
        '''
        rho = log(sigma**2)
        '''
        xs = []
        for _ in range(self.mc):
            epsilon = torch.randn_like(rho)
            xs.append(mu + rho.div(2).exp() * epsilon)
        xs = torch.stack(xs, dim=2) # [B, n_classes, mc]
        xs = torch.softmax(xs, dim=1) # [B, n_classes, mc]
        logprob = xs.mean(dim=2).log() # [B, n_classes]
        return logprob
        
    def forward(self, x):
        x = self.features_conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        mu, rho = x[:,:self.n_classes], x[:,self.n_classes:]
        logprob = self.get_logprob(mu, rho)
        return logprob, mu, rho
