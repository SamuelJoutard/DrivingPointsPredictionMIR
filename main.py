import json
import nibabel as nib
import numpy as np
import os
import random
import time
import warnings
warnings.filterwarnings('ignore')
from skimage.measure import label
import errno

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.autograd.functional import jacobian as J

from utils import *
import argparse
from models import *
from train import *
from datatools import *

parser = argparse.ArgumentParser()




parser.add_argument('--prediction_type', "-pt", type=str, help="""One of "structured", "Froestner", "Grid".""", default="structured")
parser.add_argument('--descriptor_type', "-dt", type=str, help="""One of "MIND", "UNet", "Intensity".""", default="UNet")
parser.add_argument('--regularization_type', "-rt", type=str, help="""One of "graphregnet", "MeanField", "pdd_graph".""", default="MeanField")
parser.add_argument('--interpolation_method', "-im", type=str, help="""One of "bilinear".""", default="bilinear")

parser.add_argument('--name', "-n", type=str)
parser.add_argument('--task', '-t', type=str, default="CT")
parser.add_argument('--path_save', "-path", type=str, default="./models_L2R2020_task3")

# Training config
parser.add_argument('--epochs', "-e", default=15, type=int)
parser.add_argument('--save_iter', "-si", default=1, type=int)
parser.add_argument('--learning_rate', "-lr", default=0.0001, type=float)
parser.add_argument('--reg_loss_w', "-rlw", default=0.01, type=float)
parser.add_argument('--jac_det_w', "-jdw", default=0.0, type=float)
parser.add_argument('--ssd_w', "-ssdw", default=0.0, type=float)

# Model config
parser.add_argument('--num_key_pts', "-N_kp", default=1920, type=int)

# Matching confing
parser.add_argument('--l_max', "-lmax", type=int, help="""Max displacement (in voxels).""", default=25)
parser.add_argument('--Q', "-q", type=int, help="""Displacements quantization.""", default=5)
parser.add_argument('--patch_radius', "-pr", type=int, help="""Patch radius (in voxels).""", default=3)
parser.add_argument('--patch_step', "-ps", type=int, help="""Patch sampling space (in voxels).""", default=2)
parser.add_argument('--K', "-k", type=int, help="""Number of neighbors in the KNN graph.""", default=15)

parser.add_argument('--load_model', "-L", type=str, help="""Path to load model weights.""", default="")

args = parser.parse_args()



prediction_type = args.prediction_type
descriptor_type = args.descriptor_type
regularization_type = args.regularization_type.lower()
if regularization_type=="meanfield":
    if prediction_type=="Grid":
        regularization_type = "pdd"
    else:
        regularization_type = "pdd_graph"
interpolation_meth = args.interpolation_method
if prediction_type.lower()=="grid" and "bilinear" in interpolation_meth:
    interpolation_meth = "bilinear_grid"

model_dir = args.name
task = args.task
path_save = args.path_save

N_kp = args.num_key_pts

# training
num_epochs = args.epochs
init_lr = args.learning_rate
save_iter = args.save_iter
reg_loss_w = args.reg_loss_w
jac_det_w = args.jac_det_w
ssd_w = args.ssd_w

# Matching confing
k = args.K
l_max = args.l_max # 8 (for refinement stage)
q = args.Q # 1 (for refinement stage)
l_width = l_max // q * 2 + 1
patch_radius = args.patch_radius
patch_step = args.patch_step

# model
base = 4
sigma2 = 1

# Resume training
load_w = args.load_model

in_C = 0
if descriptor_type=="UNet":
    in_C = 5 + 32
elif descriptor_type.lower()=="intensity":
    in_C = 5 + 2
else: # MIND
    in_C = 5 + 24

conf = {
    "name" : args.name,
    "CPE_conf" : {
        "type" : prediction_type,
        "disp_range" : 0.2,
        "size_grid" : (96 // 8, 80 // 8, 128 // 8),
        "N_kp" : N_kp,
        "in_C" : in_C
    },
    "FE_conf" : {
        "type" : descriptor_type,
        "in_c" : 4,
        "out_c" : 16,
        "n_down" : 3,
        "n_fix" : 4,
        "C" : 16
    },
    "Matching_conf" : {
        "size" : (96, 80, 128),
        "disp_radius" : l_max,
        "disp_step" : q,
        "patch_radius" : patch_radius,
        "patch_step" : patch_step
    },
    "Reg_conf" : {
        "type" : regularization_type,
        "sigma" : 0.1,
        "l_w" : 2 * (l_max // q) + 1,
        "size_grid" : (96 // 8, 80 // 8, 128 // 8),
        "k" : k
    },
    "Interp_conf" : {
        "type" : interpolation_meth,
        "intermediate_shape" : (96//3, 80//3, 128//3),
        "final_shape" : (96, 80, 128),
        "init_shape" : (96 // 8, 80 // 8, 128 // 8),
        "C" : 16,
        "n_layers": 3
    },
    "Training_conf" : {
        "lr" : init_lr,
        "reg_w" : reg_loss_w,
        "jac_det_w" : jac_det_w,
        "ssd_w" : ssd_w,
    },
    "Model_conf" : {
        "k_knn" : k
    }
}


if not os.path.exists(os.path.join(path_save, model_dir)):
    os.makedirs(os.path.join(path_save, model_dir))

with open(os.path.join(path_save, model_dir, "config.json"), "w") as outfile:
    json.dump(conf, outfile) 

if task=="CT":
    data_path = "../Chest_CT_reg/data/img_resampled"
    mask_path = "../Chest_CT_reg/data/label_resampled"
    train_list = ["02", "03", "05", "06", "08", "09", "21", "22", "24", "25", "27", "28", "30", "31", "33", "34", "36", "37", "39", "40"]
    test_list = ["01", "04", "07", "10", "23", "26", "29", "32", "35", "38"]
    data_dict = load_data_CT(train_list, test_list, data_path, mask_path)
else: # MR
    data_path = "../Chest_CT_reg/data/L2R_Task1_MR_resampled"
    train_list = ["02", "03", "05", "06", "08", "09", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "24", "25", "27", "28", "30", "31", "33", "34", "36", "37", "39", "40"]
    test_list = ["01", "04", "07", "10", "23", "26", "29", "32", "35", "38"]
    data_dict = load_data_MRI(train_list, test_list, data_path)

# misc
device = 'cuda'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



train_cases = []
for case1 in train_list:
    for case2 in train_list:
        if case1!=case2:
            case = case1 + "_" + case2
            train_cases.append(case)
test_cases = []
for case1 in test_list:
    for case2 in test_list:
        if case1!=case2:
            case = case1 + "_" + case2
            test_cases.append(case)

_, _, D, H, W = data_dict["01"]["img"].shape

model = ProbDispRegistration(conf).cuda()

if os.path.isfile(os.path.join(path_save, model_dir, "model_w.pth")):
    model.load_state_dict(torch.load(os.path.join(path_save, model_dir, "model_w.pth")))

if load_w!="":
    if os.path.isfile(os.path.join(load_w, "model_w.pth")):
        model.load_state_dict(torch.load(os.path.join(load_w, "model_w.pth")))
    else:
        raise FileNotFoundError("File {} missing.".format(os.path.join(load_w, "model_w.pth")))

# statistics
losses = []
losses_tracker = {}
if os.path.isfile(os.path.join(path_save, model_dir, "losses.json")):
    with open(os.path.join(path_save, model_dir, "losses.json")) as json_file:
        losses_tracker = json.load(json_file)

e_start = 0
for keys in losses_tracker.keys():
    if "val" in keys and keys!="val_dice_organs":
        e_start = len(losses_tracker[keys]) // 90

torch.cuda.synchronize()
t0 = time.time()


# optimizer
optimizer = optim.Adam(model.parameters(), init_lr)

if os.path.isfile(os.path.join(path_save, model_dir, "optimizer.pth")):
    optimizer.load_state_dict(torch.load(os.path.join(path_save, model_dir, "optimizer.pth")))

for epoch in range(e_start, num_epochs):
    losses_tracker = Evaluate(
            conf["Training_conf"], 
            model, 
            losses_tracker, 
            test_cases, 
            data_dict, 
            os.path.join(path_save, model_dir), 
            epoch, 
            device=device
        )
    model, optimizer, losses_tracker = train_one_epoch(
        conf["Training_conf"],
        model, 
        optimizer, 
        train_cases, 
        data_dict, 
        losses_tracker, 
        os.path.join(path_save, model_dir), 
        epoch, 
        device=device,
        save=(epoch%save_iter==0))
    if epoch%save_iter==0:
        losses_tracker = Evaluate(
            conf["Training_conf"], 
            model, 
            losses_tracker, 
            test_cases, 
            data_dict, 
            os.path.join(path_save, model_dir), 
            epoch, 
            device=device
        )
losses_tracker = Evaluate(
    conf["Training_conf"],
    model, 
    losses_tracker, 
    test_cases, 
    data_dict, 
    os.path.join(path_save, model_dir), 
    "final", 
    device=device
)



