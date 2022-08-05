import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.functional import jacobian as J
from scipy.spatial import Delaunay

from utils import *

import time


class ProbDispRegistration(nn.Module):
    def __init__(self, conf):
        super(ProbDispRegistration, self).__init__()
        self.conf = conf
        self.cpe = DrivingPointsExtractor(conf["CPE_conf"])
        self.fe = FeatureExtraction(conf["FE_conf"])
        self.match = Matching(conf["Matching_conf"])
        self.reg = Regularizator(conf["Reg_conf"])
        self.interp = Interpolation(conf["Interp_conf"])

        self.grid = None
        self.grid_size = None
        self.k = conf["Model_conf"]["k_knn"]

    def forward(self, fix, mov, fix_mask, mov_mask):
        B, _, X, Y, Z = fix.size()
        device = fix.device

        if self.grid_size is None or self.grid_size!=(X, Y, Z):
            self.grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, X), 
                torch.linspace(-1, 1, Y),
                torch.linspace(-1, 1, Z)), dim=0).unsqueeze(0).float().to(device)
            self.grid_size = (X, Y, Z)

        features_fix = self.fe(fix, self.grid)
        features_mov = self.fe(mov, self.grid)

        kpts_fixed = self.cpe(fix, mov, features_fix, features_mov, fix_mask, mov_mask, self.grid)
        N_kp = kpts_fixed.size(1)

        cost, disp = self.match(kpts_fixed, features_fix, features_mov)
        cost_reg = self.reg(cost, kpts_fixed)
        cost_reg = F.softmax(cost_reg.view(1, N_kp, -1), 2)
        expected_disp = (disp.unsqueeze(1) * cost_reg.unsqueeze(3)).sum(2)

        ddf = self.interp(kpts_fixed, expected_disp, features_fix)
        return ddf, kpts_fixed, cost, cost_reg, disp, features_fix, features_mov

    def inference(self, fix, mov, fix_mask, mov_mask):
        B, _, X, Y, Z = fix.size()
        device = fix.device

        if self.grid_size is None or self.grid_size!=(X, Y, Z):
            self.grid = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, X), 
                torch.linspace(-1, 1, Y),
                torch.linspace(-1, 1, Z)), dim=0).unsqueeze(0).float().to(device)
            self.grid_size = (X, Y, Z)

        features_fix = self.fe(fix, self.grid)
        features_mov = self.fe(mov, self.grid)

        kpts_fixed = self.cpe.inference(fix, mov, features_fix, features_mov, fix_mask, mov_mask, self.grid)
        N_kp = kpts_fixed.size(1)

        cost, disp = self.match(kpts_fixed, features_fix, features_mov)

        cost_reg = self.reg(cost, kpts_fixed)
        expected_disp = (disp.unsqueeze(1) * F.softmax(cost_reg.view(1, N_kp, -1), 2).unsqueeze(3)).sum(2)

        ddf = self.interp(kpts_fixed, expected_disp, features_fix)
        return ddf, kpts_fixed, cost



###### Pipeline Steps ######


class FeatureExtraction(nn.Module):
    def __init__(self, conf):
        super(FeatureExtraction, self).__init__()
        self.conf = conf
        if conf["type"].lower()=="unet":
            in_c = conf["in_c"]
            out_c = conf["out_c"]
            n_down = conf["n_down"]
            n_fix = conf["n_fix"]
            C = conf["C"]
            self.model = Unet(in_c, out_c, n_down, n_fix, C)
        elif conf["type"].lower()=="mind":
            self.model = mindsscModule()    

    def forward(self, im, grid):
        if self.conf["type"].lower()=="unet":
            return self.model(torch.cat([im, grid], dim=1))
        elif self.conf["type"].lower()=="mind":
            return self.model(im)
        elif self.conf["type"].lower()=="intensity":
            return im


class DrivingPointsExtractor(nn.Module):
    def __init__(self, conf):
        super(DrivingPointsExtractor, self).__init__()
        self.conf = conf
        if self.conf["type"].lower()=="structured":
            n_down = conf.get("n_down", 3)
            n_fix = conf.get("n_fix", 4)
            in_C = conf.get("in_C", 4)
            C = conf.get("C", 16)
            channel_inflation = conf.get("channel_inflation", 2)
            disp_range = conf.get("disp_range", 0.2)
            self.model = DrivingPointPredictorGridOffset(n_down, n_fix, in_C, C, channel_inflation, disp_range=disp_range)
        elif self.conf["type"].lower()=="grid":
            D, H, W = conf["size_grid"] 
            grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, D+4), torch.linspace(-1, 1, H+4), torch.linspace(-1, 1, W+4)), dim=3)
            grid = grid[2:-2, 2:-2, 2:-2].unsqueeze(0)
            grid = torch.reshape(grid, (1, -1, 3))[..., [2, 1, 0]]
            self.register_buffer("grid", grid)
        elif self.conf["type"].lower()=="froestner":
            self.d = conf.get("d", 5)
            self.N_kp = conf["N_kp"]

    def forward(self, fix, mov, features_fix, features_mov, mask_fix, mask_mov, grid):
        if self.conf["type"].lower()=="structured":
            return self.model(torch.cat([fix, mov, features_fix, features_mov, grid], dim=1))
        elif self.conf["type"].lower()=="grid":
            return self.grid
        elif self.conf["type"].lower()=="froestner":
            return foerstner_kpts(fix, mask_fix, d=self.d, num_points=None)
    
    def inference(self, fix, mov, features_fix, features_mov, mask_fix, mask_mov, grid):
        return self.forward(fix, mov, features_fix, features_mov, mask_fix, mask_mov, grid)


class Matching(nn.Module):
    def __init__(self, conf):
        super(Matching, self).__init__()
        self.conf = conf

        self.D, self.H, self.W = conf["size"]
        self.disp_radius = conf["disp_radius"]
        self.q = conf["disp_step"]
        self.patch_radius = conf["patch_radius"]
        self.patch_step = conf["patch_step"]
        self.l_width = self.disp_radius // self.q * 2 + 1

    def forward(self, fix_kps, feat_fix, feat_mov):
        cost, disp = displacement_distribution(fix_kps, feat_fix, feat_mov, (self.D, self.H, self.W), disp_radius=self.disp_radius, disp_step=self.q, patch_radius=self.patch_radius, patch_step=self.patch_step)
        return cost.view(-1, 1, self.l_width, self.l_width, self.l_width), disp


class Regularizator(nn.Module):
    def __init__(self, conf):
        super(Regularizator, self).__init__()
        self.conf = conf

        if self.conf["type"].lower()=="graphregnet":
            base = conf.get("base", 4)
            sigma2 = conf.get("sigma2", 1)
            self.model = GraphRegNet(base, sigma2)
            self.model.apply(init_weights)
            self.k = conf["k"]
        elif self.conf["type"].lower()=="pdd_graph":
            sigma = conf["sigma"]
            self.model = deeds_graph(sigma)
            self.k = conf["k"]
        elif self.conf["type"].lower()=="pdd":
            l_w = conf["l_w"]
            size = conf["size_grid"]
            self.model = deeds(l_w, size)

    def forward(self, cost, kpts):
        if self.conf["type"].lower() in ["graphregnet", "pdd_graph"]:
            knn, _, _, knn_dist, dist = knn_graph(kpts, self.k, include_self=True)
        if self.conf["type"].lower()=="graphregnet":
            return self.model(cost, kpts, knn)
        elif self.conf["type"].lower()=="pdd_graph":
            return self.model(cost, knn[0], knn_dist[0])
        elif self.conf["type"].lower()=="pdd":
            return self.model(cost)


class Interpolation(nn.Module):
    def __init__(self, conf):
        super(Interpolation, self).__init__()
        self.conf = conf
        if self.conf["type"].lower()=="bilinear":
            size = conf["final_shape"]
            self.linear_interp = BilinearInterpolationModule(size)
        elif self.conf["type"].lower()=="bilinear_grid":
            self.D, self.H, self.W = conf["final_shape"]  # Final shape
            self.D_, self.H_, self.W_ = conf["init_shape"] # Initial shape
        
    def forward(self, kpts, disp, features_fixed):
        """
        kpts: B, N, 3
        disp: B, N, 3
        features_fixed: B, C, X, Y, Z
        """
        if self.conf["type"].lower()=="bilinear":
            disp_pred = self.linear_interp(kpts, disp)
        elif self.conf["type"].lower()=="bilinear_grid":
            disp_pred = torch.reshape(disp, (kpts.size(0), self.D_, self.H_, self.W_, 3)).permute(0, 4, 1, 2, 3)
            disp_pred = F.interpolate(disp_pred, size=(self.D, self.H, self.W), mode='trilinear')
        return disp_pred





###### Driving points prediction


class DrivingPointPredictor(nn.Module):
    def __init__(self, n_down, n_fix, in_C, C, channel_inflation, N_kp, input_size):
        super(DrivingPointPredictor, self).__init__()
        self.n_down = n_down
        self.n_fix = n_fix
        self.in_C = in_C
        self.C = C
        self.N_kp = N_kp
        self.ci = channel_inflation
        self.input_size = input_size
        X, Y, Z = input_size

        self.conv_init = nn.Sequential(nn.Conv3d(in_C, C, 3, 1, 1), 
                                       nn.ReLU(), 
                                       nn.InstanceNorm3d(C))
        for l in range(n_fix):
            setattr(self, "layer_enc_0_"+str(l), nn.Sequential(nn.Conv3d(C, C, 3, 1, 1), 
                                                           nn.ReLU(), 
                                                           nn.InstanceNorm3d(C)))
        for lvl in range(n_down):
            setattr(self, "down_"+str(lvl), nn.Sequential(nn.Conv3d(self.ci**(lvl)*C, self.ci**(lvl+1)*C, 3, 2, 1), 
                                                           nn.ReLU(), 
                                                           nn.InstanceNorm3d(self.ci**(lvl+1)*C)))
            for l in range(n_fix):
                setattr(self, "layer_enc_"+str(lvl+1)+"_"+str(l), nn.Sequential(nn.Conv3d(self.ci**(lvl+1)*C, self.ci**(lvl+1)*C, 3, 1, 1), 
                                                                          nn.ReLU(), 
                                                                          nn.InstanceNorm3d(self.ci**(lvl+1)*C)))
        
        X_f = X
        Y_f = Y
        Z_f = Z
        for i in range(n_down):
            X_f = X_f // 2 if X_f%2==0 else X_f//2+1
            Y_f = Y_f // 2 if Y_f%2==0 else Y_f//2+1
            Z_f = Z_f // 2 if Z_f%2==0 else Z_f//2+1

        self.final = nn.Sequential(
            nn.Linear(X_f * Y_f * Z_f * self.ci**(n_down)*C, N_kp * C),
            nn.ReLU(),
            nn.Linear(N_kp * C, N_kp * 4)
        )

    def forward(self, x):
        x = self.conv_init(x)
        for l in range(self.n_fix):
            x = getattr(self, "layer_enc_0_"+str(l))(x)
        for lvl in range(self.n_down):
            x = getattr(self, "down_"+str(lvl))(x)
            for l in range(self.n_fix):
                x = getattr(self, "layer_enc_"+str(lvl+1)+"_"+str(l))(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = self.final(x)
        x = torch.reshape(x, (x.size(0), self.N_kp, 4))

        kps = F.tanh(x[..., :3])
        var = torch.exp(x[..., 3:])

        # Centralized keypoints
        kps = kps - kps.mean(1, keepdim=True)

        return kps, var


class DrivingPointPredictorGridOffset(nn.Module):
    def __init__(self, n_down, n_fix, in_C, C, channel_inflation, disp_range=1.):
        super(DrivingPointPredictorGridOffset, self).__init__()
        self.n_down = n_down
        self.n_fix = n_fix
        self.in_C = in_C
        self.C = C
        self.ci = channel_inflation
        self.disp_range = disp_range

        self.conv_init = nn.Sequential(nn.Conv3d(in_C, C, 3, 1, 1), 
                                       nn.ReLU(), 
                                       nn.InstanceNorm3d(C))
        for l in range(n_fix):
            setattr(self, "layer_enc_0_"+str(l), nn.Sequential(nn.Conv3d(C, C, 3, 1, 1), 
                                                           nn.ReLU(), 
                                                           nn.InstanceNorm3d(C)))
        for lvl in range(n_down):
            setattr(self, "down_"+str(lvl), nn.Sequential(nn.Conv3d(self.ci**(lvl)*C, self.ci**(lvl+1)*C, 3, 2, 1), 
                                                           nn.ReLU(), 
                                                           nn.InstanceNorm3d(self.ci**(lvl+1)*C)))
            for l in range(n_fix):
                setattr(self, "layer_enc_"+str(lvl+1)+"_"+str(l), nn.Sequential(nn.Conv3d(self.ci**(lvl+1)*C, self.ci**(lvl+1)*C, 3, 1, 1), 
                                                                          nn.ReLU(), 
                                                                          nn.InstanceNorm3d(self.ci**(lvl+1)*C)))

        self.final = nn.Sequential(
            nn.Conv3d(self.ci**(n_down)*C, self.ci**(n_down)*C, 1, 1, 0),
            nn.ReLU(),
            nn.Conv3d(self.ci**(n_down)*C, 3, 1, 1, 0)
        )

        self.grid = None
        self.grid_shape = None

    def forward(self, x):
        device = x.device
        x = self.conv_init(x)
        for l in range(self.n_fix):
            x = getattr(self, "layer_enc_0_"+str(l))(x)
        for lvl in range(self.n_down):
            x = getattr(self, "down_"+str(lvl))(x)
            for l in range(self.n_fix):
                x = getattr(self, "layer_enc_"+str(lvl+1)+"_"+str(l))(x)
        x = self.final(x) # 1, 4, X, Y, Z
        x = F.tanh(x[:, :3]) * self.disp_range
        _, __, X, Y, Z = x.size()
        if self.grid is None or self.grid_shape!=(X, Y, Z):
            grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, X+4), torch.linspace(-1, 1, Y+4), torch.linspace(-1, 1, Z+4)), dim=3).to(device)
            self.grid = grid
            self.grid_shape = (X, Y, Z)
        else:
            grid = self.grid
        kps = grid[2:-2, 2:-2, 2:-2].unsqueeze(0) + x.permute(0, 2, 3, 4, 1)
        kps = torch.reshape(kps, (kps.size(0), -1, 3))[..., [2, 1, 0]]
        return kps




class Unet(nn.Module):
    def __init__(self, in_c, out_c, n_down, n_fix, C, Instance_norm=True):
        super(Unet, self).__init__()
        """
        Standard Unet
        n_down: number of downsampling steps
        n_fix: number of convolutional layers at each resolution
        C: number of channels at max resolutions
        Instance_norm: Weither to use instance norm or batchnorm
        """
        self.n_down = n_down
        self.n_fix = n_fix
        self.C = C
        self.IN = Instance_norm
        self.in_c = in_c
        self.out_c = out_c

        self.conv_init = nn.Sequential(nn.Conv3d(in_c, C, 3, 1, 1), 
                                       nn.ReLU(), 
                                       nn.InstanceNorm3d(C) if self.IN else nn.BatchNorm3d(C))
        for l in range(n_fix):
            setattr(self, "layer_enc_0_"+str(l), nn.Sequential(nn.Conv3d(C, C, 3, 1, 1), 
                                                           nn.ReLU(), 
                                                           nn.InstanceNorm3d(C) if self.IN else nn.BatchNorm3d(C)))
        for lvl in range(n_down):
            setattr(self, "down_"+str(lvl), nn.Sequential(nn.Conv3d(2**(lvl)*C, 2**(lvl+1)*C, 3, 2, 1), 
                                                           nn.ReLU(), 
                                                           nn.InstanceNorm3d(2**(lvl+1)*C) if self.IN else nn.BatchNorm3d(2**(lvl+1)*C)))
            for l in range(n_fix):
                setattr(self, "layer_enc_"+str(lvl+1)+"_"+str(l), nn.Sequential(nn.Conv3d(2**(lvl+1)*C, 2**(lvl+1)*C, 3, 1, 1), 
                                                                          nn.ReLU(), 
                                                                          nn.InstanceNorm3d(2**(lvl+1)*C) if self.IN else nn.BatchNorm3d(2**(lvl+1)*C)))
        for lvl in range(n_down):
            setattr(self, "up_"+str(lvl), nn.Sequential(nn.ConvTranspose3d(2**(lvl+1)*C, 2**(lvl)*C, 4, 2, 1), 
                                                        nn.ReLU(), 
                                                        nn.InstanceNorm3d(2**(lvl)*C) if self.IN else nn.BatchNorm3d(2**(lvl)*C)))
            for l in range(n_fix):
                if l==0:
                    setattr(self, "layer_dec_"+str(lvl)+"_0", nn.Sequential(nn.Conv3d(2**(lvl+1)*C, 2**(lvl)*C, 3, 1, 1), 
                                                                        nn.ReLU(), 
                                                                        nn.InstanceNorm3d(2**(lvl)*C) if self.IN else nn.BatchNorm3d(2**(lvl)*C)))
                else:
                    setattr(self, "layer_dec_"+str(lvl)+"_"+str(l), nn.Sequential(nn.Conv3d(2**(lvl)*C, 2**(lvl)*C, 3, 1, 1), 
                                                                              nn.ReLU(), 
                                                                              nn.InstanceNorm3d(2**(lvl)*C) if self.IN else nn.BatchNorm3d(2**(lvl)*C)))
        for lvl in range(self.n_down+1):
            setattr(self, "final_layer_"+str(lvl), nn.Conv3d(2**lvl*C, out_c, 3, 1, 1))
        
    def forward(self, x):
        x = self.conv_init(x)
        L = []
        for l in range(self.n_fix):
            x = getattr(self, "layer_enc_0_"+str(l))(x)
        L.append(x)
        for lvl in range(self.n_down):
            x = getattr(self, "down_"+str(lvl))(x)
            for l in range(self.n_fix):
                x = getattr(self, "layer_enc_"+str(lvl+1)+"_"+str(l))(x)
            L.append(x)
        for lvl in range(self.n_down-1, -1, -1):
            x = getattr(self, "up_"+str(lvl))(x)
            x = torch.cat([x, L[lvl]], dim=1)
            for l in range(self.n_fix):
                x = getattr(self, "layer_dec_"+str(lvl)+"_"+str(l))(x)
        x = getattr(self, "final_layer_0")(x)
        return x
    
    def forward_to_lvl(self, x, level):
        """
        Return embeddings at a certain resolution
        x: input
        level: resolution of the returned embedding (0 is max resolution, then resolution is divided by 2**level)
        """
        x = self.conv_init(x)
        L = []
        for l in range(self.n_fix):
            x = getattr(self, "layer_enc_0_"+str(l))(x)
        L.append(x)
        for lvl in range(self.n_down):
            x = getattr(self, "down_"+str(lvl))(x)
            for l in range(self.n_fix):
                x = getattr(self, "layer_enc_"+str(lvl+1)+"_"+str(l))(x)
            L.append(x)
        if level==self.n_down:
            return getattr(self, "final_layer_"+str(level))(x)
        for lvl in range(self.n_down-1, level-1, -1):
            x = getattr(self, "up_"+str(lvl))(x)
            x = torch.cat([x, L[lvl]], dim=1)
            for l in range(self.n_fix):
                x = getattr(self, "layer_dec_"+str(lvl)+"_"+str(l))(x)
        x = getattr(self, "final_layer_"+str(level))(x)
        return x





# graphregnet

class GaussianSmoothing(nn.Module):
    def __init__(self, sigma):
        super(GaussianSmoothing, self).__init__()
        
        sigma = torch.tensor([sigma])
        N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
        weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N), 2) / (2 * torch.pow(sigma, 2)))
        weight /= weight.sum()
        
        self.register_buffer("weight", weight)
        
    def forward(self, x):
        x = filter1D(x, self.weight, 0)
        x = filter1D(x, self.weight, 1)
        x = filter1D(x, self.weight, 2)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base=4):
        super(Encoder, self).__init__()
    
        self.conv_in = nn.Sequential(nn.Conv3d(in_channels, base, 3, stride=1, padding=1, bias=False),
                                     nn.InstanceNorm3d(base),
                                     nn.LeakyReLU())
        
        self.conv1 = nn.Sequential(nn.Conv3d(base, 2*base, 3, stride=2, padding=1, bias=False),
                                   nn.InstanceNorm3d(2*base),
                                   nn.LeakyReLU())
        
        self.conv2 = nn.Sequential(nn.Conv3d(2*base, 4*base, 3, stride=2, padding=1, bias=False),
                                   nn.InstanceNorm3d(4*base),
                                   nn.LeakyReLU())
        
    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        return x1, x2, x3
            

class Decoder(nn.Module):
    def __init__(self, out_channels=1, base=4):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv3d(4*base, 2*base, 3, stride=1, padding=1, bias=False),
                                   nn.InstanceNorm3d(2*base),
                                   nn.LeakyReLU())
    
        self.conv1a = nn.Sequential(nn.Conv3d(4*base, 2*base, 3, stride=1, padding=1, bias=False),
                                    nn.InstanceNorm3d(2*base),
                                    nn.LeakyReLU())
        
        self.conv2 = nn.Sequential(nn.Conv3d(2*base, base, 3, stride=1, padding=1, bias=False),
                                   nn.InstanceNorm3d(base),
                                   nn.LeakyReLU())
        
        self.conv2a = nn.Sequential(nn.Conv3d(2*base, base, 3, stride=1, padding=1, bias=False),
                                    nn.InstanceNorm3d(base),
                                    nn.LeakyReLU())
        
        self.conv_out = nn.Sequential(nn.Conv3d(base, 1, 3, padding=1))
        
    def forward(self, x1, x2, x3):
        x = F.interpolate(x3, size=x2.shape[-3:], mode='trilinear')
        x = self.conv1(x)
        x = self.conv1a(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, size=x1.shape[-3:], mode='trilinear')
        x = self.conv2(x)
        x = self.conv2a(torch.cat([x, x1], dim=1))
        x = self.conv_out(x)
        return x
    

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
    
        self.conv = nn.Sequential(
            nn.Conv3d(self.in_channels * 2, self.out_channels, 1, bias=False),
            nn.InstanceNorm3d(self.out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x, ind):
        B, N, C, D, _, _ = x.shape
        k = ind.shape[2]
        y = x.view(B*N, C, D*D*D)[ind.view(B*N, k)].view(B, N, k, C, D*D*D)
        x = x.view(B, N, C, D*D*D).unsqueeze(2).expand(-1, -1, k, -1, -1)
        x = torch.cat([y - x, x], dim=3).permute(0, 3, 1, 2, 4)
        x = self.conv(x)
        x = x.mean(dim=3).permute(0, 2, 1, 3).view(B, N, -1, D, D, D)
        return x
    

class GCN(nn.Module):
    def __init__(self, base=4):
        super(GCN, self).__init__()
        
        self.base = base
        
        self.conv1 = EdgeConv(4*self.base + 3, 4*self.base)
        self.conv2 = EdgeConv(2*4*self.base + 3, 4*self.base)
        self.conv3 = EdgeConv(3*4*self.base + 3, 4*self.base)
        
    def forward(self, x1, x2, x3, kpts, ind):
        expand = x3.shape[-1]
        xa = self.conv1(torch.cat([x3, kpts.view(-1, 3, 1, 1, 1).expand(-1, -1, expand, expand, expand)], dim=1).unsqueeze(0), ind).squeeze(0)
        xb = self.conv2(torch.cat([torch.cat([x3, kpts.view(-1, 3, 1, 1, 1).expand(-1, -1, expand, expand, expand)], dim=1), xa], dim=1).unsqueeze(0), ind).squeeze(0)
        xc = self.conv3(torch.cat([torch.cat([x3, kpts.view(-1, 3, 1, 1, 1).expand(-1, -1, expand, expand, expand)], dim=1), xa, xb], dim=1).unsqueeze(0), ind).squeeze(0)
        return x1, x2, xc
    

class GraphRegNet(nn.Module):
    def __init__(self, base, smooth_sigma):
        super(GraphRegNet, self).__init__()
        
        self.base = base
        self.smooth_sigma = smooth_sigma
        
        self.pre_filter1 = GaussianSmoothing(self.smooth_sigma)
        self.pre_filter2 = GaussianSmoothing(self.smooth_sigma)
            
        self.encoder1 = Encoder(2, self.base)
        self.gcn1 = GCN(self.base)
        self.decoder1 = Decoder(1, self.base)
        
        self.encoder2 = Encoder(4, self.base)
        self.gcn2 = GCN(self.base)
        self.decoder2 = Decoder(1, self.base)
        
    def forward(self, x, kpts, kpts_knn):
        x1 = self.encoder1(torch.cat([x, self.pre_filter1(x)], dim=1))
        x1 = self.gcn1(*x1, kpts, kpts_knn)
        x1 = self.decoder1(*x1)
        x1 = F.interpolate(x1, size=x.shape[-3:], mode='trilinear')
        x2 = self.encoder2(torch.cat([x, self.pre_filter1(x), x1, self.pre_filter2(x1)], dim=1))
        x2 = self.gcn2(*x2, kpts, kpts_knn)
        x2 = self.decoder2(*x2)
        return x2


def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)


class deeds_graph(nn.Module):
    def __init__(self, sigma):
        super(deeds_graph, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1,.1,1,1,.1,1]))#.cuda()

        self.pad1 = nn.ReplicationPad3d(3)#.cuda()
        self.avg1 = nn.AvgPool3d(3,stride=1)#.cuda()
        self.max1 = nn.MaxPool3d(3,stride=1)#.cuda()
        self.pad2 = nn.ReplicationPad3d(2)#.cuda()##

        self.sigma = sigma

    def forward(self, cost, knn, dist_nn):
        """
        cost: N, 1, disp_width, disp_width, disp_width
        knn: N, K
        dist_nn : N, K
        """
        N, _, l_w, l_w, l_w = cost.size()
        K = knn.size(1)

        cost = self.alpha[1] + self.alpha[0] * cost

        dist_nn = torch.exp(- dist_nn / (2 * self.sigma**2))

        # remove mean (not really necessary)
        #deeds_cost = deeds_cost.view(-1,displacement_width**3) - deeds_cost.view(-1,displacement_width**3).mean(1,keepdim=True)[0]
        cost = cost.view(1,-1,l_w,l_w,l_w)
    
        # approximate min convolution / displacement compatibility
        cost_approx = self.avg1(self.avg1(self.max1(self.pad1(cost)))) # 1, -1, X*Y*Z, l_w, l_w, l_w

        # grid-based mean field inference (one iteration)
        norm = dist_nn.sum(1)
        cost_approx_nn = torch.stack([cost_approx[:, knn[:, k], :, :, :] for k in range(K)], dim=2) # 1, N, K, l_w, l_w, l_w
        cost_approx = (cost_approx_nn * dist_nn[None, :, :, None, None, None]).sum(2) # 1, N, l_w, l_w, l_w
        cost_approx = cost_approx / norm[None, :, None, None, None]

        # second path
        cost = self.alpha[4]+self.alpha[2]*cost+self.alpha[3]*cost_approx
        cost = self.avg1(self.avg1(self.max1(self.pad1(cost))))

        # grid-based mean field inference (one iteration)
        cost_nn = torch.stack([cost[:, knn[:, k], :, :, :] for k in range(K)], dim=2) # 1, N, K, l_w, l_w, l_w
        cost = (cost_nn * dist_nn[None, :, :, None, None, None]).sum(2) # 1, N, l_w, l_w, l_w
        cost = cost / norm[None, :, None, None, None]

        return self.alpha[5] * cost.view(1, N, -1)


class deeds(nn.Module):
    def __init__(self, l_w, size):

        super(deeds, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1,.1,1,1,.1,1]))#.cuda()

        self.pad1 = nn.ReplicationPad3d(3)#.cuda()
        self.avg1 = nn.AvgPool3d(3,stride=1)#.cuda()
        self.max1 = nn.MaxPool3d(3,stride=1)#.cuda()
        self.pad2 = nn.ReplicationPad3d(2)#.cuda()##

        self.l_w = l_w
        self.X, self.Y, self.Z = size

    def forward(self, cost):

        cost = self.alpha[1] + self.alpha[0] * cost
        
        # remove mean (not really necessary)
        #deeds_cost = deeds_cost.view(-1,displacement_width**3) - deeds_cost.view(-1,displacement_width**3).mean(1,keepdim=True)[0]
        deeds_cost = deeds_cost.view(1,-1,self.l_w,self.l_w,self.l_w)
    
        # approximate min convolution / displacement compatibility
        cost = self.avg1(self.avg1(self.max1(self.pad1(deeds_cost))))
   
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2,3,4,0,1).view(1,self.l_w**3,self.X,self.Y,self.Z)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0,2,3,4,1).view(1,-1,self.l_w,self.l_w,self.l_w)
        
        # second path
        cost = self.alpha[4]+self.alpha[2]*deeds_cost+self.alpha[3]*cost_avg
        cost = self.avg1(self.avg1(self.max1(self.pad1(cost))))
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2,3,4,0,1).view(1,self.l_w**3,self.X,self.Y,self.Z)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0,2,3,4,1).view(1, self.X*self.Y*self.Z,self.l_w**3)

        return self.alpha[5]*cost_avg



class MarkovRandomField(nn.Module):
    def __init__(self, sigma_transition, sigma_pairwise, alpha_unary, l_max, q, size):
        super(MarkovRandomField, self).__init__()
        self.st = sigma_transition
        self.sp = sigma_pairwise
        self.au = alpha_unary
        
        D, H, W = size
        disp = (torch.stack(torch.meshgrid(torch.arange(0, 2 * l_max + 1, q),
                                      torch.arange(0, 2 * l_max + 1, q),
                                      torch.arange(0, 2 * l_max + 1, q)), dim=3) - l_max).contiguous().view(-1, 3).float()
        disp = (disp.flip(-1) * 2 / torch.tensor([W-1, H-1, D-1]))
        A = torch.exp(-((disp[:, None, :] - disp[None, :, :])**2).sum(-1) / (2 * self.st**2))
        self.register_buffer("A", A)

    def forward(self, unary, dist):
        device = unary.device
        unary = self.au * unary
        pairwise = torch.exp(- dist / (2*self.sp**2))
        _, tree_next, tree_pred, dist_from_root = MinSpanningTree(dist)
        res = Viterbi(dist_from_root, tree_next, tree_pred, unary, pairwise, self.A)
        disp_out = torch.zeros(unary.size()).to(device) - 1
        for i in res.keys():
            disp_out[i, res[i]] = 1
        return disp_out * 100000


class transformerReg(nn.Module):
    def __init__(self, N_kpts, C, D, n_layers):
        super(transformerReg, self).__init__()
        self.N_kpts = N_kpts
        self.C = C
        self.D = D
        self.n_layers = n_layers

        for l in range(n_layers):
            setattr(self, "layer_f_{}".format(l), nn.Sequential(nn.Linear(C+3, C+3), nn.ReLU(), nn.LayerNorm((N_kpts, C+3))))
            setattr(self, "layer_d_{}".format(l), nn.Sequential(nn.Linear(D, D), nn.ReLU()))


    def forward(self, kpts, disps_distr, feats):
        """
        kpts: B, N, 3
        disps_distr: N, 1, D**1/3, D**1/3, D**1/3
        feats: B, N, C
        """
        B, N, _ = kpts.size()
        feats_kpts = torch.cat([kpts, feats], dim=-1)
        disps_distr = torch.reshape(disps_distr, (1, N, -1))
        for l in range(self.n_layers):
            gpu_usage()
            feats_kpts = getattr(self, "layer_f_{}".format(l))(feats_kpts)
            gpu_usage()
            disps_distr = getattr(self, "layer_d_{}".format(l))(disps_distr)
            gpu_usage()
            K = torch.exp(-((feats_kpts[:, :, None, :] - feats_kpts[:, None, :, :])**2).sum(-1)) # B, N, N
            K = K / K.sum(-1, keepdim=True)
            gpu_usage()
            disps_distr = (K[:, :, :, None] * disps_distr[:, None, :, :]).sum(2)
            gpu_usage()

        return disps_distr


class transformerReg_PAM(nn.Module):
    def __init__(self, N_kpts, C, D, n_layers, n_blocks, share_w=False):
        super(transformerReg_PAM, self).__init__()
        self.N_kpts = N_kpts
        self.C = C
        self.D = D
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.share_w = share_w

        if share_w:
            for l in range(n_layers-1):
                setattr(self, "layer_0_f_{}".format(l), nn.Sequential(nn.Linear(C+3, C+3), nn.ReLU(), nn.LayerNorm((N_kpts, C+3))))
                setattr(self, "layer_0_d_{}".format(l), nn.Sequential(nn.Linear(D, D), nn.ReLU()))
            setattr(self, "layer_0_f_{}".format(n_layers-1), nn.Sequential(nn.Linear(C+3, 8), nn.ReLU(), nn.LayerNorm((N_kpts, 8))))
            setattr(self, "layer_0_d_{}".format(n_layers-1), nn.Sequential(nn.Linear(D, D), nn.ReLU()))
            self.block_to_layer = {i:0 for i in range(n_blocks)}
        else:
            for b in range(n_blocks):
                for l in range(n_layers-1):
                    setattr(self, "layer_{}_f_{}".format(b, l), nn.Sequential(nn.Linear(C+3, C+3), nn.ReLU(), nn.LayerNorm((N_kpts, C+3))))
                    setattr(self, "layer_{}_d_{}".format(b, l), nn.Sequential(nn.Linear(D, D), nn.ReLU()))
                setattr(self, "layer_{}_f_{}".format(b, n_layers-1), nn.Sequential(nn.Linear(C+3, 8), nn.ReLU(), nn.LayerNorm((N_kpts, 8))))
                setattr(self, "layer_{}_d_{}".format(b, n_layers-1), nn.Sequential(nn.Linear(D, D), nn.ReLU()))
            self.block_to_layer = {i:i for i in range(n_blocks)}


    def forward(self, kpts, disps_distr, feats):
        """
        kpts: B, N, 3
        disps_distr: B, N, D
        feats: B, N, C
        """
        B, N, _ = kpts.size()
        feats_kpts = torch.cat([kpts, feats], dim=-1)
        disps_distr = torch.reshape(disps_distr, (1, N, -1))
        for b in range(self.n_blocks):
            # t0 = time.time()
            for l in range(self.n_layers):
                feats_kpts_latt = getattr(self, "layer_{}_f_{}".format(self.block_to_layer[b], l))(feats_kpts)
                disps_distr = getattr(self, "layer_{}_d_{}".format(self.block_to_layer[b], l))(disps_distr)
            # t1 = time.time()
            # print("Layers for regularization block {} took: {} s.".format(b, t1 - t0))
            # t0 = time.time()
            disps_distr = PermutohedralLattice.apply(feats_kpts_latt.permute(0, 2, 1), disps_distr.permute(0, 2, 1)).permute(0, 2, 1)
            # t1 = time.time()
            # print("PL for regularization block {} took: {} s.".format(b, t1 - t0))

        return disps_distr


        







#################################################################################################################################################################################################################
##################################### Interpolation #############################################################################################################################################################
#################################################################################################################################################################################################################


def get_barycentric_coordinates(points_tri, target):
    s = points_tri.find_simplex(target)
    dim = target.shape[1]
    
    b0 = (points_tri.transform[s, :dim].transpose([1, 0, 2]) *
          (target - points_tri.transform[s, dim])).sum(axis=2).T
    coord = np.c_[b0, 1 - b0.sum(axis=1)]

    return coord, s


def linear_interpolation_material(points, target):
    """
    Linearly interpolate signal at target locations
    points: numpy array (N, D)
    target: numpy array (N, D)
    """
    points_triangulated = Delaunay(points)
    c, s = get_barycentric_coordinates(points_triangulated, target)
    
    return points_triangulated.simplices, points_triangulated.transform, c, s


class LinearInterpolation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, values, target):
        """
        points: points where the signal is known; torch tensor (B, N, D)
        values: signal; torch tensor (B, N, C)
        target: where the signal needs to be interpolated; torch tensor (B, M, D)
        """
        device = points.device
        B = points.size(0)
        dtype = points.dtype

        if B>1:
            raise NotImplementedError("Linear interpolation not implemented for batches larger than 1.")

        points_np = points.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        simplices, T, coords, s = linear_interpolation_material(points_np[0], target_np[0])
        simplices = torch.tensor(simplices).long().to(device)  # n_simplices, D+1
        T = torch.tensor(T).float().to(device) # n_simplices, D+1
        coords = torch.tensor(coords).float().to(device) # M, D+1
        # T = torch.tensor(T).float().to(device) # n_simplices, D+1
        # coords = torch.tensor(coords).float().to(device) # M, D+1
        s = torch.tensor(s).long().to(device) # M
        
        # res = values[0, simplices[s]] # M, D+1, C

        res = (values[0, simplices[s]] * coords[:, :, None]).sum(1) # M,C
        
        ctx.save_for_backward(points, values, target, simplices, T, coords, s, res)
        return res[None, :]

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: 1, M, C
        """
        points, values, target, simplices, T, coords, s, res = ctx.saved_tensors
        device = points.device
        dim = points.size(2)
        _, N, C = values.size()
        M = target.size(1)
        
        simplices_C = torch.stack([simplices]*C, dim=-1)
        
        grad_values = torch.zeros(values.size()).float().to(device)
        for d in range(dim+1):
            grad_values.scatter_add_(1, simplices_C[s][None, :, d], (coords[:, d, None] * grad_output[0])[None, :]) # B, M, C

        grad_coords = (values[0, simplices[s]] * grad_output[0, :, None, :]).sum(-1) # M, D+1

        grad_points = torch.zeros(points.size()).float().to(device)

        simples_points = torch.stack([simplices]*dim, dim=-1) # n_simplices, D+1, D

        for d in range(dim):
            grad_col = []
            for pd in range(dim):
                
                A = torch.zeros((M, dim, dim)).float().to(device)
                A[:, pd, d] = 1
                
                d_lmbda123_dp = (- torch.bmm(torch.bmm(T[s][:, :dim], A), T[s][:, :dim]) * (target[0] - T[s][:, dim])[:, None, :]).sum(-1) # M, D

                d_lmbda1234_dp = torch.cat([d_lmbda123_dp, -d_lmbda123_dp.sum(-1, keepdim=True)], dim=1) # M, D+1

                d_lmbda1234_dp = d_lmbda1234_dp * grad_coords # dlmbdas / dp_d^pd  ;  M, D+1

                grad_col.append(d_lmbda1234_dp.sum(-1))   # M,
            grad_col = torch.stack(grad_col, dim=1) # M, D
            grad_points.scatter_add_(1, simples_points[s][None, :, d], grad_col[None, :])
        
        grad_col = []
        for pd in range(dim):
            A = torch.zeros((M, dim, dim)).float().to(device)
            A[:, pd, :] = -1
            # A = torch.stack([A]*M, dim=0)
            d_lmbda123_dp = (- torch.bmm(torch.bmm(T[s][:, :dim], A), T[s][:, :dim]) * (target[0] - T[s][:, dim])[:, None, :]).sum(-1) - T[s][:, :dim, pd]# M, D
            d_lmbda1234_dp = torch.cat([d_lmbda123_dp, -d_lmbda123_dp.sum(-1, keepdim=True)], dim=1) # M, D+1
            d_lmbda1234_dp = d_lmbda1234_dp * grad_coords # dlmbdas / dp_d^pd  ;  M, D+1
            grad_col.append(d_lmbda1234_dp.sum(-1))   # M,
        grad_col = torch.stack(grad_col, dim=1) # M, D
        grad_points.scatter_add_(1, simples_points[s][None, :, dim], grad_col[None, :])

        return grad_points, grad_values, None



class BilinearInterpolationModule(nn.Module):
    def __init__(self, size):
        super(BilinearInterpolationModule, self).__init__()

        self.size = size

        grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1)+size, align_corners=True).view(1, -1, 3)
        self.register_buffer("grid", grid)

        pads = torch.ones((1, 8, 3))
        pads[0, 1, 0] = -1
        pads[0, 2, 1] = -1
        pads[0, 3, 2] = -1
        pads[0, 4, 0] = -1
        pads[0, 4, 1] = -1
        pads[0, 5, 0] = -1
        pads[0, 5, 2] = -1
        pads[0, 6, 1] = -1
        pads[0, 6, 2] = -1
        pads[0, 7, 0] = -1
        pads[0, 7, 1] = -1
        pads[0, 7, 2] = -1
        self.register_buffer("pads", pads)

        pads_values = torch.zeros((1, 8, 3))
        self.register_buffer("pads_values", pads_values)

    def forward(self, kpts, disp):
        """
        kpts: B, N, 3
        disp: B, N, 3
        """
        kpts_pad = torch.cat([kpts, self.pads], dim=1)
        disp_pad = torch.cat([disp, self.pads_values], dim=1)
        interp = LinearInterpolation.apply(kpts_pad, disp_pad, self.grid)
        return torch.reshape(interp, (kpts.size(0),)+self.size+(3,)).permute(0, 4, 1, 2, 3)


class KernelInterpolationFeat(nn.Module):
    def __init__(self, size, C, n_layers=1):
        super(KernelInterpolationFeat, self).__init__()

        self.size = size

        grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1)+size, align_corners=True)
        self.register_buffer("grid", grid)

        self.C = C
        self.n_layers = n_layers

        for l in range(n_layers):
            setattr(self, "layer_{}".format(l), nn.Sequential(nn.Conv3d(C+3, C+3, 3, 1, 1), nn.ReLU(), nn.InstanceNorm3d(C+3)))

    def forward(self, kpts, disp, feats):
        """
        kpts: B, N, 3
        disp: B, N, 3
        feats: B, N_grid, C
        """
        feats_grid = torch.cat([feats, self.grid.permute(0, 4, 1, 2, 3)], dim=1)
        for l in range(self.n_layers):
            feats_grid = getattr(self, "layer_{}".format(l))(feats_grid)
        
        feats_kpts = F.grid_sample(feats_grid, kpts[:, None, None, :, :], mode="bilinear", align_corners=True)[:, :, 0, 0] # B, C+3, N

        feats_grid = torch.reshape(feats_grid, (1, self.C+3, -1))

        K = torch.exp(-((feats_kpts[:, :, None, :] - feats_grid[:, :, :, None])**2).sum(1)) # B, N_grid, N

        K = K / K.sum(2, keepdim=True)

        disp_grid = (K[:, :, :, None] * disp[:, None, :, :]).sum(2) # B, N_grid, 3
        disp_grid = torch.reshape(disp_grid, (1,)+self.size+(3,)).permute(0, 4, 1, 2, 3)

        return disp_grid


class KernelInterpolationFeatPAM(nn.Module):
    def __init__(self, size, C, n_layers=1):
        super(KernelInterpolationFeatPAM, self).__init__()

        self.size = size

        grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1)+size, align_corners=True)
        self.register_buffer("grid", grid)

        self.C = C
        self.n_layers = n_layers

        for l in range(n_layers-1):
            setattr(self, "layer_{}".format(l), nn.Sequential(nn.Conv3d(C+3, C+3, 3, 1, 1), nn.ReLU(), nn.InstanceNorm3d(C+3)))
        setattr(self, "layer_{}".format(n_layers-1), nn.Sequential(nn.Conv3d(C+3, 8, 3, 1, 1), nn.ReLU(), nn.InstanceNorm3d(8)))

    def forward(self, kpts, disp, feats):
        """
        kpts: B, N, 3
        disp: B, N, 3
        feats: B, C, X, Y, Z
        """
        feats_grid = torch.cat([feats, self.grid.permute(0, 4, 1, 2, 3)], dim=1)
        for l in range(self.n_layers):
            feats_grid = getattr(self, "layer_{}".format(l))(feats_grid)
        feats_kpts = F.grid_sample(feats_grid, kpts[:, None, None, :, :], mode="bilinear", align_corners=True)[:, :, 0, 0] # B, C+3, N
        # feats_grid = torch.reshape(feats_grid, (1, self.C+3, -1))
        feats_grid = torch.reshape(feats_grid, (1, 8, -1))
        disp_grid = PermutohedralLatticeAsymmetric.apply(feats_kpts, feats_grid, disp.permute(0, 2, 1))
        return torch.reshape(disp_grid, (1, 3)+self.size)





if __name__=="__main__":

    from time import time


    ################# Gradient correctness check ###############


    N = 25
    D = 3
    M = 10


    points = torch.rand(1, N, D)
    values = torch.rand(1, N, 2)*10
    target = torch.rand(1, M, D)

    points.requires_grad = True
    values.requires_grad = True
    target.requires_grad = False

    # value_emp = (lmbdas[None, :] * values).sum()

    print(torch.autograd.gradcheck(lambda *x: LinearInterpolation.apply(x[0], x[1], target.double()), (points.double(), values.double())))



    #################### Speed test ######################

    # N = 2500
    # D = 3
    # M = 100000


    # points = torch.rand(1, N, D).cuda()
    # values = torch.rand(1, N, 2).cuda()*10
    # target = torch.rand(1, M, D).cuda()

    # points.requires_grad = True
    # values.requires_grad = True
    # target.requires_grad = False

    # obj = torch.rand(1, M, 2).cuda()

    # t0 = time()
    # val = LinearInterpolation.apply(points, values, target)
    # t1 = time()
    # print("Forward pass took: {} s.".format(t1-t0))

    # loss = ((val - obj)**2).sum()

    # t0 = time()
    # loss.backward()
    # t1 = time()
    # print("Backward pass took: {} s.".format(t1-t0))


