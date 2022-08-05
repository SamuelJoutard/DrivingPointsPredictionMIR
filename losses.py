import numpy as np
import torch
import torch.nn as nn

from utils import *


def feat_ssd(feat_fixed, feat_moving, disp, mask):
    """
    Computes sum of squared differences between a fixed feature map and moving one according to a displacement field and a mask
    """
    mse_loss = nn.MSELoss(reduction='none')
    loss = (mse_loss(feat_fixed, warp_tensor(feat_moving, disp.permute(0, 2, 3, 4, 1))) * mask).sum() / mask.float().sum()
    return loss

def SSD(x, y):
    return (((x - y)**2).sum(1)).mean()

class LNCC(nn.Module):
    def __init__(self, w=2):
        super(LNCC, self).__init__()
        self.w = torch.ones(1, 1, (2*w+1), (2*w+1), (2*w+1)).cuda() / (2*w + 1)**3
        self.conv = nn.Conv3d(1, 1, (2*w+1), 1, w, bias=False)
        self.conv.weight.data = self.w

    def forward(self, M, R):
        M_m = self.conv(M)
        R_m = self.conv(R)  
        MM_m = self.conv(M*M)
        RR_m = self.conv(R*R)
        MR_m = self.conv(M*R)
        M_var = (MM_m - M_m**2+0.00001)**0.5
        R_var = (RR_m - R_m**2+0.00001)**0.5
        corr = (MR_m - M_m * R_m) / (M_var * R_var + 0.00001) 
        
        return -corr.mean()


def Dice(lbl1, lbl2):
    """
    Computes mean dice score between two label maps
    lbl1: B, L, X, Y, Z
    lbl2: B, L, X, Y, Z
    """
    dice_score = 2 * (lbl1 * lbl2).sum((2, 3, 4)) / (lbl1.sum((2, 3, 4)) + lbl2.sum((2, 3, 4)) + 0.000001)
    return 1 - dice_score.mean()


def Dice_per_organ(lbl1, lbl2):
    """
    Computes dice score per organ between two label maps
    lbl1: B, L, X, Y, Z
    lbl2: B, L, X, Y, Z
    """
    dice_score = 2 * (lbl1 * lbl2).sum((2, 3, 4)) / (lbl1.sum((2, 3, 4)) + lbl2.sum((2, 3, 4)) + 0.000001)
    return dice_score.mean(0)


def jacobian(disp):
    """
    Compute the jacobian of a displacement field B, 3, X, Y, Z
    """
    d_dx = disp[:, :, 1:, :-1, :-1] - disp[:, :, :-1, :-1, :-1]
    d_dy = disp[:, :, :-1, 1:, :-1] - disp[:, :, :-1, :-1, :-1]
    d_dz = disp[:, :, :-1, :-1, 1:] - disp[:, :, :-1, :-1, :-1]
    jac = torch.stack([d_dx, d_dy, d_dz], dim=1) # B, [ddisp_./dx, disp_./dy, ddisp_./dz], [ddisp_x/d., ddisp_y/d., ddisp_z/d.], X, Y, Z
    return F.pad(jac, (0, 1, 0, 1, 0, 1)) # B, 3, 3, X, Y, Z


def Hessian_penalty(ddf):
    """
    Computes bending energy of the displacement field
    """
    jac = jacobian(ddf) # B, 3, 3, X, Y, Z
    B, _, __, X, Y, Z = jac.size()
    hess = jacobian(torch.reshape(jac, (B, -1, X, Y, Z)))
    return (hess**2).sum((1,2)).mean()


def Jacobian_det(disp):
    """
    Computes mean jacobian determinant of the deformation field, given displacement field
    """
    _, __, D, H, W = disp.size()
    device = disp.device

    # identity = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1, D, H, W), align_corners=True).permute(0, 4, 1, 2, 3)

    jac = jacobian((disp)[:, [2, 1, 0]])
    jac[:, 0, 0] += 1.0
    jac[:, 1, 1] += 1.0
    jac[:, 2, 2] += 1.0
    det = (
        jac[:, 0, 0] * jac[:, 1, 1] * jac[:, 2, 2] +
        jac[:, 0, 1] * jac[:, 1, 2] * jac[:, 2, 0] +
        jac[:, 0, 2] * jac[:, 1, 0] * jac[:, 2, 1] -
        jac[:, 0, 0] * jac[:, 1, 2] * jac[:, 2, 1] - 
        jac[:, 0, 1] * jac[:, 1, 0] * jac[:, 2, 2] -
        jac[:, 0, 2] * jac[:, 1, 1] * jac[:, 2, 0]
    )
    return ((det-1)**2).mean()