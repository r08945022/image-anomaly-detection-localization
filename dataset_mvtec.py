import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from perlin import generate_perlin_mask
from cutpaste import patch_ex

import imgaug.augmenters as iaa


CLASSNAME = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 
                'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

class MVTECdataset(Dataset):
    def __init__(self, datasetdir, classname, phase='train', resize_wh=256):
        assert classname in CLASSNAME, 'classname: {} is not in mvtecdataset: {}'.format(classname, CLASSNAME)
        assert phase == 'train' or phase == 'test', 'phase must be train or test'
        self.datasetdir = datasetdir
        self.classname = classname
        self.phase = phase
        self.resize_wh = resize_wh
        
        self.labellist, self.imgpathlist, self.maskpathlist = self.load_data_from_folder()

        self.augcolor = [
                    
                    ]

        self.augpillike = [

                    ]

        self.auggeo = [
                        iaa.Rot90([0, 1, 2, 3]),
                    ]


    def __getitem__(self, idx):
        
        label = self.labellist[idx]
        imgpath = self.imgpathlist[idx]
        maskpath = self.maskpathlist[idx]


        img = cv2.imread(imgpath, 1) 
        img = cv2.resize(img, (self.resize_wh, self.resize_wh))
        img = img[..., [2,1,0]] 
        

        if self.phase == 'train':

            rand_idx = np.random.randint(len(self.imgpathlist))
            img2path = self.imgpathlist[rand_idx]
            img2 = cv2.imread(img2path, 1)
            img2 = cv2.resize(img2, (self.resize_wh, self.resize_wh))
            img2 = img2[..., [2,1,0]]
            cutpaste_aug = patch_ex(img, img2)

            # img_aug mask
            rand_idx4 = np.random.randint(len(self.imgpathlist))
            img4path = self.imgpathlist[rand_idx4]
            img4 = cv2.imread(img4path, 1)
            img4 = cv2.resize(img4, (self.resize_wh, self.resize_wh))
            img4 = img4[..., [2,1,0]]
            aug4 = self.randAugmenter()
            img4 = aug4(image=img4)
            mask = generate_perlin_mask(self.resize_wh, threshold=0.9)
            perlin_aug = img * (1 - mask) + img4 * mask

            
            img = np.transpose((img.astype(np.float32) / 255.0), (2, 0, 1))
            cutpaste_aug = np.transpose((cutpaste_aug.astype(np.float32) / 255.0), (2, 0, 1))
            perlin_aug = np.transpose((perlin_aug.astype(np.float32) / 255.0), (2, 0, 1))

            return img, cutpaste_aug, perlin_aug


        elif self.phase == 'test':

            if maskpath is not None:
                mask = cv2.imread(maskpath, 0)   
                mask = cv2.resize(mask, (self.resize_wh, self.resize_wh))
                mask = mask.clip(max=1)  
                mask = np.expand_dims(mask, axis=2)

            else:
                mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

            mask = np.transpose((mask.astype(np.float32)), (2, 0, 1))
            img = np.transpose( (img.astype(np.float32) / 255.0), (2, 0, 1))
            label = np.asarray(label, dtype=np.float32)

            return label, img, mask
            

    def __len__(self):
        return len(self.imgpathlist)


    def load_data_from_folder(self):
        '''
        image for training: only normal images
        image for testing: contain abnormal images
        mvtec folder structure: 
        1.datasetdir/classname/train/good/normal.png
        2.datasetdir/classname/test/anomalous_situation/anomaly.png
          datasetdir/classname/test/good/normal.png
        3.datasetdir/classname/ground_truth/anomalous_situation/mask.png   <--corresponding seg mask for testing imgs 
        '''
        imgdir = os.path.join(self.datasetdir, self.classname, self.phase)    #1,2
        maskdir = os.path.join(self.datasetdir, self.classname, 'ground_truth')  #3

        img_types = sorted(os.listdir(imgdir))

        imgpathlist = []
        labellist = []  # normal=0 anomal=1
        maskpathlist = []

        for imgtype in img_types:
            filedir = os.path.join(imgdir, imgtype)
            filelist = sorted(os.listdir(filedir))
            
            if imgtype == 'good':
                for i in range(len(filelist)):
                    imgpathlist.append(os.path.join(filedir, filelist[i]))
                    labellist.append(0)
                    maskpathlist.append(None)

            else:
                gtfiledir = os.path.join(maskdir, imgtype)
                gtfilelist = sorted(os.listdir(gtfiledir))
                for i in range(len(filelist)):
                    imgpathlist.append(os.path.join(filedir, filelist[i]))
                    labellist.append(1)
                    maskpathlist.append(os.path.join(gtfiledir, gtfilelist[i]))
                

        assert len(imgpathlist)==len(labellist)==len(maskpathlist), 'length should be the same'
        for i in range(len(imgpathlist)):

            if maskpathlist[i] is not None:
                b1=imgpathlist[i].split("/")[-2]
                b2=maskpathlist[i].split("/")[-2]

                a1=imgpathlist[i].split("/")[-1][:3]
                a2=maskpathlist[i].split("/")[-1][:3]

                assert b1==b2 and a1==a2, 'anomal imgtype: imagefile mismatch gtfile '
            else:
                b1=imgpathlist[i].split("/")[-2]

                assert b1=='good' and labellist[i]==0 and maskpathlist[i]==None, 'normal imgtype, label=0, maskpath=None '
                

        return labellist, imgpathlist, maskpathlist



    def randAugmenter(self):
        # aug_ind1 = np.random.choice(np.arange(len(self.augcolor)), 1, replace=False)[0]
        # aug_ind2 = np.random.choice(np.arange(len(self.augpillike)), 1, replace=False)[0]
        aug_ind3 = np.random.choice(np.arange(len(self.auggeo)), 1, replace=False)[0]

        aug = iaa.Sequential([ 
                            #   self.augpillike[aug_ind2],
                            #   self.augcolor[aug_ind1], 
                              self.auggeo[aug_ind3], 
                            ])
        return aug


