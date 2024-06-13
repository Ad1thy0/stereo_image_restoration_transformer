from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random
import torch
import numpy as np
from glob import glob
from patchify import patchify

    
def augmentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        if random.random()<0.5:     # flip horizonly
            lr_image_left_ = lr_image_right[:, ::-1, :]
            lr_image_right_ = lr_image_left[:, ::-1, :]
            hr_image_left_ = hr_image_right[:, ::-1, :]
            hr_image_right_ = hr_image_left[:, ::-1, :]
            lr_image_left, lr_image_right = lr_image_left_, lr_image_right_
            hr_image_left, hr_image_right = hr_image_left_, hr_image_right_

        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

def toTensor(img):
    if img.ndim == 3:
        img = torch.from_numpy(img.transpose((2, 0, 1)))
    elif img.ndim == 4:
        img = torch.from_numpy(img.transpose((0,3,1,2)))
    else:
        print("Neither image nor patches of image")
        exit()
    return img.float().div(255)


class TrainSetLoader(Dataset):
    def __init__(self, data_dir = '/home/adithyal/Stereo_LLE/data/',
                 exposure = 'mixed',
                 mode = 'train', patch_size = (128,256)):
        super(TrainSetLoader, self).__init__()

        self.samples = []
        self.patch_size = patch_size
        if exposure == 'mixed':
            ll_l_files = sorted(glob(data_dir+'1_*/'+mode+'/cam1/*.png'))
            ll_r_files = sorted(glob(data_dir+'1_*/'+mode+'/cam2/*.png'))
            wl_l_files = sorted(glob(data_dir+'well_lit/'+mode+'/cam1/*.png'))*3
            wl_r_files = sorted(glob(data_dir+'well_lit/'+mode+'/cam2/*.png'))*3
        else:
            ll_l_files = sorted(glob(data_dir+exposure+'/'+mode+'/cam1/*.png'))
            ll_r_files = sorted(glob(data_dir+exposure+'/'+mode+'/cam2/*.png'))
            wl_l_files = sorted(glob(data_dir+'well_lit/'+mode+'/cam1/*.png'))
            wl_r_files = sorted(glob(data_dir+'well_lit/'+mode+'/cam2/*.png'))
            
        assert len(ll_l_files) == len(ll_r_files) == len(wl_l_files) == len(wl_r_files)

        for i in range(len(ll_l_files)):
            sample = dict()
            sample['ll_l'] = ll_l_files[i]
            sample['ll_r'] = ll_r_files[i]
            sample['wl_l'] = wl_l_files[i]
            sample['wl_r'] = wl_r_files[i]

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)*100

    def __getitem__(self, index):
        sample = dict()

        sample_path = self.samples[index%len(self.samples)]
        ll_l_img = np.array(Image.open(sample_path['ll_l']), dtype=np.float32)
        ll_r_img = np.array(Image.open(sample_path['ll_r']), dtype=np.float32)
        wl_l_img = np.array(Image.open(sample_path['wl_l']), dtype=np.float32)
        wl_r_img = np.array(Image.open(sample_path['wl_r']), dtype=np.float32)

        ori_height, ori_width = ll_l_img.shape[:2]
        y, x = np.random.randint(ori_height-self.patch_size[0]+1), np.random.randint(ori_width-self.patch_size[1]+1)

        ll_l_img = ll_l_img[y:y+self.patch_size[0],x:x+self.patch_size[1]]
        ll_r_img = ll_r_img[y:y+self.patch_size[0],x:x+self.patch_size[1]]
        wl_l_img = wl_l_img[y:y+self.patch_size[0],x:x+self.patch_size[1]]
        wl_r_img = wl_r_img[y:y+self.patch_size[0],x:x+self.patch_size[1]]

        wl_l_img, wl_r_img, ll_l_img, ll_r_img = augmentation(wl_l_img, wl_r_img, ll_l_img, ll_r_img)

        sample['ll_l']=toTensor(ll_l_img)
        sample['ll_r']=toTensor(ll_r_img)
        sample['wl_l']=toTensor(wl_l_img)
        sample['wl_r']=toTensor(wl_r_img)

        return sample


class ValSetLoader(Dataset):
    def __init__(self, data_dir = '/home/adithyal/Stereo_LLE/data/',
                 exposure = 'mixed',
                 mode = 'val', patch_size = (128,256)):  
        super(ValSetLoader, self).__init__()

        self.samples = []
        self.patch_size = patch_size
        if exposure == 'mixed':
            ll_l_files = sorted(glob(data_dir+'1_*/'+mode+'/cam1/*.png'))
            ll_r_files = sorted(glob(data_dir+'1_*/'+mode+'/cam2/*.png'))
            wl_l_files = sorted(glob(data_dir+'well_lit/'+mode+'/cam1/*.png'))*3
            wl_r_files = sorted(glob(data_dir+'well_lit/'+mode+'/cam2/*.png'))*3
        else:
            ll_l_files = sorted(glob(data_dir+exposure+'/'+mode+'/cam1/*.png'))
            ll_r_files = sorted(glob(data_dir+exposure+'/'+mode+'/cam2/*.png'))
            wl_l_files = sorted(glob(data_dir+'well_lit/'+mode+'/cam1/*.png'))
            wl_r_files = sorted(glob(data_dir+'well_lit/'+mode+'/cam2/*.png'))
            
        assert len(ll_l_files) == len(ll_r_files) == len(wl_l_files) == len(wl_r_files)

        for i in range(len(ll_l_files)):
            sample = dict()
            sample['ll_l'] = ll_l_files[i]
            sample['ll_r'] = ll_r_files[i]
            sample['wl_l'] = wl_l_files[i]
            sample['wl_r'] = wl_r_files[i]

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = dict()

        sample_path = self.samples[index]
        ll_l_img = np.array(Image.open(sample_path['ll_l']), dtype=np.float32)
        ll_r_img = np.array(Image.open(sample_path['ll_r']), dtype=np.float32)
        wl_l_img = np.array(Image.open(sample_path['wl_l']), dtype=np.float32)
        wl_r_img = np.array(Image.open(sample_path['wl_r']), dtype=np.float32)

        ll_l_img = patchify(ll_l_img, (self.patch_size[0],self.patch_size[1],3), (self.patch_size[0], self.patch_size[1], 3)).reshape(-1,self.patch_size[0], self.patch_size[1],3)
        ll_r_img = patchify(ll_r_img, (self.patch_size[0],self.patch_size[1],3), (self.patch_size[0], self.patch_size[1], 3)).reshape(-1,self.patch_size[0], self.patch_size[1],3)
        wl_l_img = patchify(wl_l_img, (self.patch_size[0],self.patch_size[1],3), (self.patch_size[0], self.patch_size[1], 3)).reshape(-1,self.patch_size[0], self.patch_size[1],3)
        wl_r_img = patchify(wl_r_img, (self.patch_size[0],self.patch_size[1],3), (self.patch_size[0], self.patch_size[1], 3)).reshape(-1,self.patch_size[0], self.patch_size[1],3)

        sample['ll_l']=toTensor(ll_l_img)
        sample['ll_r']=toTensor(ll_r_img)
        sample['wl_l']=toTensor(wl_l_img)
        sample['wl_r']=toTensor(wl_r_img)

        return sample