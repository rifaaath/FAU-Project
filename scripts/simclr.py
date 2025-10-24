import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class SimCLRModel(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super(SimCLRModel, self).__init__()

        # Load backbone
        if base_model == "resnet18":
            self.encoder = models.resnet18(pretrained=True)
            feat_dim = self.encoder.fc.in_features
        elif base_model == "resnet50":
            self.encoder = models.resnet50(pretrained=True)
            feat_dim = self.encoder.fc.in_features
        else:
            raise ValueError("Unsupported base model")

        # Remove classifier
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # remove final FC
        self.feat_dim = feat_dim

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)  # fix explosion
        return z

