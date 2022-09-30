import datetime, time
import glob
import cv2
import math
import gc
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics import roc_auc_score

from dataset_mvtec import MVTECdataset
from wavecnet import wresnet50



def test_real_anomaly(netD, test_loader, train_feature_layers, select_layers):


    with torch.no_grad():

        netD.cuda()
        netD.eval()

        gt_list_px_lvl = []
        pred_list_px_lvl = []
        gt_list_img_lvl = []
        pred_list_img_lvl = []

        metric_dict = {}

        for label, test_img, gt_mask in test_loader:

            test_img = test_img.cuda()
            _, layerlist = netD(test_img)

            m = torch.nn.AvgPool2d(3, 1, 1)

            feature_map_list = []
            n_neighbors=1

            for i in range(len(select_layers)):

                l = select_layers[i]

                f = m(layerlist[l].detach())
                
                f_size = f.size(1)
                feature = f.permute(0, 2, 3, 1).reshape(-1, f_size)

                
                distances = torch.cdist(feature.cuda(), train_feature_layers[i].cuda(), p=2.0)  # euclidean norm
                score_patches, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
                scores = np.mean(score_patches.cpu().numpy(), axis=-1)
                size = int(math.sqrt(score_patches.shape[0]))
                feature_map = scores.reshape((size,size))
                feature_map_resized = cv2.resize(feature_map, (test_img.shape[-1], test_img.shape[-1]))
                feature_map_list.append(feature_map_resized)
            

            multi_map = feature_map_list[0]
            for i in range(1, len(feature_map_list)):
                multi_map = multi_map * feature_map_list[i]

            score = multi_map.max()

            gt = gt_mask.numpy()[0,0].astype(np.float32)

            gt_list_px_lvl.extend(gt.flatten())
            pred_list_px_lvl.extend(multi_map.flatten())
            gt_list_img_lvl.append(label.numpy())
            pred_list_img_lvl.append(score)
            

        patchcore_auroc_pixel = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
        metric_dict["auc_pixel_patchcore"] = patchcore_auroc_pixel

        patchcore_auroc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)
        metric_dict["auc_image_patchcore"] = patchcore_auroc
        

        return metric_dict



def get_feature(train_loader, netD, writer_dict, select_layers):


    with torch.no_grad():

        netD.cuda()
        netD.eval()


        train_feature_layers_before_stack = []
        for i in range(len(select_layers)):
            train_feature_layers_before_stack.append([])

        train_feature_layers = []

        for img, _, _ in train_loader:

            _, layerlist_n = netD(img.cuda())

            m = torch.nn.AvgPool2d(3, 1, 1)
    
            for i in range(len(select_layers)):
                l = select_layers[i]

                f = m(layerlist_n[l].detach())

                f_size = f.size(1)
                feature = f.permute(0, 2, 3, 1).reshape(-1, f_size)
                train_feature_layers_before_stack[i].extend(feature)
            
        
        for i in range(len(select_layers)):
            train_feature_layers.append( torch.vstack(train_feature_layers_before_stack[i]).cuda() )

        

        return train_feature_layers, writer_dict



if __name__ == "__main__":
    
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', action='store', type=str, required=True)
    parser.add_argument('--p', action='store', type=str, required=True)
    args = parser.parse_args()

    if args.c in class_list:
        picked_classes = [args.c]
    else:
        print('class is not in mvtec dataset.')
        exit()

    path_list = [args.p]

    dataset_dir = "./mvtec_anomaly_detection/"
    
    img_size = 256
    batch = 1

    netD = wresnet50()
    modelname = 'wave50'
    
    

    for object in picked_classes:
        print(object)
        
        for prefix in path_list:
            if object in prefix:
                print(prefix)
                path1 =  glob.glob('./ckpt/*{obj}-{model}/e1600*'.format(obj=object, model=modelname))

                for i in range(len(path1)):
                    if prefix in str(path1[i]):
                        print(path1[i])
                        ckpt_path = path1[i]


        select_layers_list = [ [1,2] ]  # 0,1,2,3; 1,2 == layer2,layer3
        
        for item in select_layers_list:

            netD = wresnet50()
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            epoch = checkpoint['epoch']
            netD.load_state_dict(checkpoint['netD_state_dict'])

            test_data = MVTECdataset(dataset_dir, object, resize_wh=img_size, phase='test')
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=24)

            writer_dict = {}
            writer_dict["ckpt"] = ckpt_path
            writer_dict['epoch'] = int(epoch)
            t = str(datetime.datetime.now().replace(microsecond=0))
            writer = SummaryWriter('./inference/multiplication-{t}-{obj}'.format(t=t, obj=object))

            dataset = MVTECdataset(dataset_dir, object, resize_wh=img_size)
            train_loader_all = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=24)
            embedding_dir = "./features/"
            embedding_des = os.path.join(embedding_dir, '{t}-{obj}'.format(t=t, obj=object) )
            embedding_des = embedding_des +"-"+modelname
        
        
            select_layers = item

            
            train_feature_layers, writer_dict = get_feature(train_loader_all, netD, writer_dict, select_layers=select_layers)
            metric_dict = test_real_anomaly(netD, test_loader, train_feature_layers, select_layers=select_layers)


            writer_dict['layers'] = str(select_layers) 

            print(writer_dict)
            print(metric_dict)
            writer.add_hparams(writer_dict, metric_dict)
            writer.close()

            
            gc.collect()
            torch.cuda.empty_cache() 



