import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import json

from losses import *
from utils import *



def train_one_epoch(conf_train, model, optimizer, train_cases, data_dict, losses_tracker, model_dir, epoch, device="cuda", save=False):
    # statistics
    running_loss = 0.0

    reg_loss_w = conf_train["reg_w"]
    jac_det_w = conf_train.get("jac_det_w", 0)
    ssd_w = conf_train["ssd_w"]
    # lncc w is 1
    
    lncc = LNCC(w=3).to(device)

    # shuffle training cases
    train_cases_perm = random.sample(train_cases, len(train_cases))
    
    # for all training cases
    for it, case in tqdm(enumerate(train_cases_perm)):
        
        # zero out gradients
        optimizer.zero_grad()

        patient_fix, patient_mov = case.split("_")
    
        # load data
        img_fixed = data_dict[patient_fix]["img"].to(device)
        img_moving = data_dict[patient_mov]["img"].to(device)
        mask_fixed = data_dict[patient_fix].get("mask", torch.ones(img_fixed.size())).to(device)
        mask_moving = data_dict[patient_mov].get("mask", torch.ones(img_moving.size())).to(device)
        lbl_fixed = data_dict[patient_fix]["lbl"].to(device)
        lbl_moving = data_dict[patient_mov]["lbl"].to(device)

        disp_pred, kpts_fixed, cost, cost_reg, disp, features_fix, features_mov = model(img_fixed, img_moving, mask_fixed, mask_moving)

        # loss
        resampled = warp_tensor(img_moving, disp_pred.permute(0, 2, 3, 4, 1))
        l_lncc = lncc(resampled, img_fixed)
        l_ssd = SSD(img_fixed, resampled)
        l_reg = Hessian_penalty(disp_pred)
        l_jac_det = Jacobian_det(disp_pred)

        loss = l_lncc + ssd_w * l_ssd + reg_loss_w * l_reg + jac_det_w * l_jac_det

        # Monitor dice
        resampled_lbl = warp_tensor(lbl_moving, disp_pred.permute(0, 2, 3, 4, 1))
        l_dice = Dice_per_organ(resampled_lbl, lbl_fixed)
        
        # backward + optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # statistics
        losses_tracker["lncc"] = losses_tracker.get("lncc", []) + [l_lncc.item()]
        losses_tracker["ssd"] = losses_tracker.get("ssd", []) + [l_ssd.item()]
        losses_tracker["reg"] = losses_tracker.get("reg", []) + [l_reg.item()]
        losses_tracker["jac_det"] = losses_tracker.get("jac_det", []) + [l_jac_det.item()]
        dic_dice = losses_tracker.get("dice_organs", {})
        for i in range(l_dice.size(0)):
            dic_dice[i] = dic_dice.get(i, []) + [l_dice[i].item()]
        losses_tracker["dice_organs"] = dic_dice
        losses_tracker["dice"] = losses_tracker.get("dice", []) + [l_dice.mean().item()]

    if save:  # Saves a an example of the training set

        aff = data_dict[patient_fix]["aff"]
        path_epoch = os.path.join(model_dir, str(epoch))
        if not os.path.isdir(path_epoch):
            os.makedirs(path_epoch)
        path_epoch_train = os.path.join(path_epoch, "training")
        if not os.path.isdir(path_epoch_train):
            os.makedirs(path_epoch_train)
        _, __, X, Y, Z = img_fixed.size()
        kpts_fixed = torch.clip(kpts_fixed, -1, 1)
        kpts_fixed = (kpts_fixed + 1) / 2
        kpts_fixed = kpts_fixed[:, :, [2, 1, 0]] * torch.tensor([[[X-1, Y-1, Z-1]]]).float().to(device)
        kp_mask = np.zeros((X, Y, Z))
        for kp in kpts_fixed[0]:
            x = int(kp[0].item())
            y = int(kp[1].item())
            z = int(kp[2].item())

            kp_mask[x,y,z] = 1

        mask_nib = nib.Nifti1Image(kp_mask, aff)
        nib.save(mask_nib, os.path.join(path_epoch_train, case+"_keypoints.nii.gz"))
        resampled = resampled.detach().cpu().numpy()[0, 0]
        resampled_nib = nib.Nifti1Image(resampled, aff)
        nib.save(resampled_nib, os.path.join(path_epoch_train, case+"_resampled.nii.gz"))

        np.save(os.path.join(path_epoch_train, case+"ddf.npy"), disp_pred.permute(0, 2, 3, 4, 1)[0].detach().cpu().numpy())

        torch.save(model.state_dict(), os.path.join(path_epoch, 'model_w.pth'))
        torch.save(optimizer.state_dict(), os.path.join(path_epoch, 'optimizer.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_w.pth'))
        torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer.pth'))
        
    return model, optimizer, losses_tracker


def Evaluate(conf_train, model, losses_tracker, test_cases, data_dict, model_dir, epoch, device="cuda"):

    path_epoch = os.path.join(model_dir, str(epoch))
    if not os.path.isdir(path_epoch):
        os.makedirs(path_epoch)
    path_epoch_val = os.path.join(path_epoch, "validation")
    if not os.path.isdir(path_epoch_val):
        os.makedirs(path_epoch_val)

    lncc = LNCC(w=3).to(device)

    with torch.no_grad():
        # for all cases
        for case in tqdm(test_cases):

            patient_fix, patient_mov = case.split("_")
            
            # load data
            img_fixed = data_dict[patient_fix]["img"].to(device)
            img_moving = data_dict[patient_mov]["img"].to(device)
            mask_fixed = data_dict[patient_fix].get("mask", torch.ones(img_fixed.size())).to(device)
            mask_moving = data_dict[patient_mov].get("mask", torch.ones(img_moving.size())).to(device)
            lbl_fixed = data_dict[patient_fix]["lbl"].to(device)
            lbl_moving = data_dict[patient_mov]["lbl"].to(device)
            aff = data_dict[patient_fix]["aff"]

            disp_pred, kpts_fixed, cost = model.inference(img_fixed, img_moving, mask_fixed, mask_moving)

            # loss
            resampled = warp_tensor(img_moving, disp_pred.permute(0, 2, 3, 4, 1))
            l_lncc = lncc(resampled, img_fixed)
            l_ssd = SSD(img_fixed, resampled)
            l_reg = Hessian_penalty(disp_pred)
            l_jac_det = Jacobian_det(disp_pred)
            resampled_lbl = warp_tensor(lbl_moving, disp_pred.permute(0, 2, 3, 4, 1))
            l_dice = Dice_per_organ(resampled_lbl, lbl_fixed)

            losses_tracker["val_lncc"] = losses_tracker.get("val_lncc", []) + [l_lncc.item()]
            losses_tracker["val_ssd"] = losses_tracker.get("val_ssd", []) + [l_ssd.item()]
            losses_tracker["val_reg"] = losses_tracker.get("val_reg", []) + [l_reg.item()]

            dic_dice = losses_tracker.get("val_dice_organs", {})
            for i in range(l_dice.size(0)):
                dic_dice[i] = dic_dice.get(i, []) + [l_dice[i].item()]
            losses_tracker["val_dice_organs"] = dic_dice

            losses_tracker["val_dice"] = losses_tracker.get("val_dice", []) + [l_dice.mean().item()]
            losses_tracker["val_jac_det"] = losses_tracker.get("val_jac_det", []) + [l_jac_det.item()]

            print("Case {}: lncc {}, jac_det {}".format(case, l_lncc.item(), l_jac_det.item()))

            _, __, X, Y, Z = img_fixed.size()
            kpts_fixed = torch.clip(kpts_fixed, -1, 1)
            kpts_fixed = (kpts_fixed + 1) / 2
            kpts_fixed = kpts_fixed[:, :, [2, 1, 0]] * torch.tensor([[[X-1, Y-1, Z-1]]]).float().to(device)
            kp_mask = np.zeros((X, Y, Z))
            for kp in kpts_fixed[0]:
                x = int(kp[0].item())
                y = int(kp[1].item())
                z = int(kp[2].item())

                w = 1

                for i_ in range(-w, w+1):
                    for j_ in range(-w, w+1):
                        for k_ in range(-w, w+1):
                            x_ = np.clip(x+i_, 0, X-1)
                            y_ = np.clip(y+j_, 0, Y-1)
                            z_ = np.clip(z+k_, 0, Z-1)
                            kp_mask[x_,y_,z_] = 1

                # kp_mask[x,y,z] = 1

            mask_nib = nib.Nifti1Image(kp_mask, aff)
            nib.save(mask_nib, os.path.join(path_epoch_val, case+"_keypoints.nii.gz"))
            resampled = resampled.detach().cpu().numpy()[0, 0]
            resampled_nib = nib.Nifti1Image(resampled, aff)
            nib.save(resampled_nib, os.path.join(path_epoch_val, case+"_resampled.nii.gz"))
            resampled = torch.argmax(resampled_lbl, dim=1).detach().cpu().numpy().astype(float)[0]
            resampled_nib = nib.Nifti1Image(resampled, aff)
            nib.save(resampled_nib, os.path.join(path_epoch_val, case+"_resampled_lbl.nii.gz"))
            np.save(os.path.join(path_epoch_val, case+"ddf.npy"), disp_pred.permute(0, 2, 3, 4, 1)[0].detach().cpu().numpy())

    with open(os.path.join(model_dir, "losses.json"), "w") as outfile:
            json.dump(losses_tracker, outfile) 

    return losses_tracker

