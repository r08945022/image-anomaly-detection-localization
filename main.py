import os
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter   

from dataset_mvtec import MVTECdataset
from wavecnet import wresnet50



def train_1epoch(loader, netD, optD, epoch):
    
    ce = nn.CrossEntropyLoss(reduction='mean')

    total_d_loss = 0
    per_acc = np.zeros((3,))

    netD.cuda()
      
    for img, cutpaste_aug, perlin_aug in loader:

        img = img.cuda()
        cutpaste_aug = cutpaste_aug.cuda()
        perlin_aug = perlin_aug.cuda()

        if epoch == 1:
            writer.add_images('_img', img, epoch, dataformats='NCHW') 
            writer.add_images('_cutpaste_aug', cutpaste_aug, epoch, dataformats='NCHW')
            writer.add_images('_perlin_aug', perlin_aug, epoch, dataformats='NCHW')


        netD.train()
        optD.zero_grad()

        d_input = torch.cat((img.detach(), cutpaste_aug.detach(), perlin_aug.detach()), dim=0)
        

        img = img.detach()


        label = torch.arange(3).cuda()
        label = label.repeat_interleave(img.size(0))
        
        

        out_logit, _ = netD(d_input)


        loss_d = ce(out_logit, label) 
        loss_d.backward(inputs=list(netD.parameters())) 
        optD.step()

        total_d_loss += loss_d.detach().cpu().item()*img.size(0)


        predicted = torch.argmax(out_logit.detach(),axis=1)

        acc = confusion_matrix(label.cpu().numpy(), predicted.cpu().numpy(), normalize='true').diagonal()

        per_acc += acc


    total_d_loss /= len(loader)
    per_acc /= len(loader)


    return total_d_loss, per_acc




def train(train_loader, netD, optD, epoch):

    total_d_loss, per_acc = train_1epoch(train_loader, netD, optD, epoch)
    total_acc = per_acc.sum()
    

    return total_acc



def main(epochs, batch, learninig_rate, weight_decay, momentum, save_des, dataset):

    netD = wresnet50()
    modelname = 'wave50'

    optD = torch.optim.SGD(netD.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)

    train_loader_all = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=24)



    for epoch in trange(1, epochs+1):
        
        total_acc = train(train_loader_all, netD, optD, epoch)
        
        if epoch==epochs:

            tmp = save_des+"-{}".format(modelname)
            if not os.path.exists(tmp):
                os.makedirs(tmp)
            ckpt_name = os.path.join(tmp, "e{}_model.pt".format(epoch))
            torch.save({
            'epoch': epoch,
            'netD_state_dict': netD.state_dict(),
            },  ckpt_name)



                


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', action='store', type=str, default="all", required=False)
    parser.add_argument('--e', action='store', type=str, required=True)
    parser.add_argument('--b', action='store', type=str, required=True)
    parser.add_argument('--lr', action='store', type=str, required=True)
    parser.add_argument('--wd', action='store', type=str, required=True)
    parser.add_argument('--m', action='store', type=str, required=True)

    args = parser.parse_args()

    class_list = [  
                    'zipper',       
                    'leather',   
                    'grid',      
                    'tile',  
                    'bottle',     
                    'toothbrush',
                    'cable',   
                    'hazelnut',  
                    'metal_nut',
                    'pill',
                    'transistor',
                    'wood',
                    'carpet',
                    'capsule',     
                    'screw',    
                ]
    

    if args.c == 'all':
        picked_classes = class_list
    else:
        if args.c in class_list:
            picked_classes = [args.c]
        else:
            print('class is not in mvtec dataset.')
            exit()


    epochs = args.e
    batch = args.b
    learninig_rate = args.lr
    weight_decay = args.wd
    momentum = args.m
    

    dataset_dir = "./mvtec_anomaly_detection/"
    
    save_dir = "./ckpt/"

    
    img_size = 256



    print(picked_classes)
    
    for object in picked_classes:

        t = str(datetime.datetime.now().replace(microsecond=0))
        save_des = os.path.join(save_dir, '{t}-{obj}'.format(t=t, obj=object) )
        
        writer = SummaryWriter('runs/{t}-{obj}'.format(t=t, obj=object))

        dataset = MVTECdataset(dataset_dir, object, resize_wh=img_size)
        

        print(object)
        main(epochs, batch, learninig_rate, weight_decay, momentum, save_des, dataset)



    