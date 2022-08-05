import torch
import nibabel as nib
import os
import numpy as np
from skimage.measure import label


def load_data_CT(train_list, test_list, data_path, mask_path):
    data_dict = {}
    OneHotConverter = torch.eye(14)

    for patient in train_list:
        path_img = os.path.join(data_path, "img00"+patient+".nii.gz")
        img = nib.load(path_img)

        img_aff = img.affine
        img_np = img.get_data()

        mask = label((img_np<-900).astype(int))
        mask = (mask==(np.argmax(np.bincount(mask.flat)[1:])+1))
        mask = 1 - mask

        img = (torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().clamp_(-1000, 1500) + 1000) / 2500
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).bool()

        lbl_path = os.path.join(mask_path, "label00"+patient+".nii.gz")
        lbl = nib.load(lbl_path).get_data()
        lbl = torch.from_numpy(lbl).long().unsqueeze(0)
        lbl = OneHotConverter[lbl].permute(0, 4, 1, 2, 3)

        data_dict[patient] = {}
        data_dict[patient]["img"] = img
        data_dict[patient]["mask"] = mask
        data_dict[patient]["aff"] = img_aff
        data_dict[patient]["lbl"] = lbl

    for patient in test_list:
        path_img = os.path.join(data_path, "img00"+patient+".nii.gz")
        img = nib.load(path_img)

        img_aff = img.affine
        img_np = img.get_data()

        mask = label((img_np<-900).astype(int))
        mask = (mask==(np.argmax(np.bincount(mask.flat)[1:])+1))
        mask = 1 - mask

        img = (torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().clamp_(-1000, 1500) + 1000) / 2500
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).bool()

        lbl_path = os.path.join(mask_path, "label00"+patient+".nii.gz")
        lbl = nib.load(lbl_path).get_data()
        lbl = torch.from_numpy(lbl).long().unsqueeze(0)
        lbl = OneHotConverter[lbl].permute(0, 4, 1, 2, 3)

        data_dict[patient] = {}
        data_dict[patient]["img"] = img
        data_dict[patient]["mask"] = mask
        data_dict[patient]["aff"] = img_aff
        data_dict[patient]["lbl"] = lbl
    return data_dict


def load_data_MRI(train_list, test_list, data_path):
    data_dict = {}
    OneHotConverter = torch.eye(5)

    for patient in train_list:
        path_img = os.path.join(data_path, "img00"+patient+"_chaos_MR.nii.gz")
        img = nib.load(path_img)

        img_aff = img.affine
        img_np = img.get_data()

        img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float()

        lbl_path = os.path.join(data_path, "seg00"+patient+"_chaos_MR.nii.gz")
        lbl = nib.load(lbl_path).get_data()
        lbl = torch.from_numpy(lbl).long().unsqueeze(0)
        lbl = OneHotConverter[lbl].permute(0, 4, 1, 2, 3)

        data_dict[patient] = {}
        data_dict[patient]["img"] = img
        data_dict[patient]["aff"] = img_aff
        data_dict[patient]["lbl"] = lbl

    for patient in test_list:
        path_img = os.path.join(data_path, "img00"+patient+"_chaos_MR.nii.gz")
        img = nib.load(path_img)

        img_aff = img.affine
        img_np = img.get_data()

        img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float()

        lbl_path = os.path.join(data_path, "seg00"+patient+"_chaos_MR.nii.gz")
        lbl = nib.load(lbl_path).get_data()
        lbl = torch.from_numpy(lbl).long().unsqueeze(0)
        lbl = OneHotConverter[lbl].permute(0, 4, 1, 2, 3)

        data_dict[patient] = {}
        data_dict[patient]["img"] = img
        data_dict[patient]["aff"] = img_aff
        data_dict[patient]["lbl"] = lbl
    return data_dict

