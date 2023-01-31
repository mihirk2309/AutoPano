import torch
import sys
import numpy as np
import torch
import pickle 
import torch.nn.functional as F
from kornia.geometry.epipolar.fundamental import normalize_points
from kornia.geometry.transform import warp_perspective
import cv2
import warnings


# Don't generate pyc codes
sys.dont_write_bytecode = True

class HomographyModel(torch.nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()

        self.model = torch.nn.Sequential(*[
            torch.nn.Conv2d(2,64,3,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64,128,3,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,3,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(128,128,3,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,3,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

        ])

        self.regress = torch.nn.Sequential(*[ 
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 1024),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024,8) ])
    
    def forward(self, x):
        out = self.model(x)
        return self.regress(out)


class TensorDLT(torch.nn.Module):
    def __init__(self):
        super(TensorDLT, self).__init__()
    
    def forward(self, xA, xB):

        XA, u = normalize_points(xA)
        XB, v = normalize_points(xB)

        x1, y1 = torch.split(XA, 2, dim=-1)         
        x2, y2 = torch.split(XB, 2, dim=-1)
        ones = torch.ones(x1)
        zeros = torch.zeros(x1)

        ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2], dim=-1)
        ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2], dim=-1)
        A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

        _, S, V = torch.linalg.svd(A + 1e-4*torch.randn(A.shape))
        H_matrix = V[..., -1].reshape(-1, 3, 3)

        H_matrix = v.inverse() @ (H_matrix @ u)
        H_matrix = H_matrix / (H_matrix[..., -1:, -1:] + 1e-8)

        return H_matrix


class HomographyModelUnsupervised(HomographyModel):
    def __init__(self):
        super(HomographyModelUnsupervised, self).__init__()
        self.dlt = TensorDLT()
    
    def forward(self, x, ptsA, patchA): 
        out = self.model(x)
        error = self.regress(out)
        error = error*32
        error = error.view(error.shape[0],-1,2)
        H = self.dlt(ptsA, ptsA+error)
        H_inv = torch.inverse(H)
        patchB = warp_perspective(patchA, H_inv, (128,128))
        return patchB, error / 32.

def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img *255
