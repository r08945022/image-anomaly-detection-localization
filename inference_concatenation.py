import datetime, time
import glob
import cv2
import math
import gc
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics import roc_auc_score

from dataset_mvtec import MVTECdataset
from wavecnet import wresnet50
from concate_features import stpm_multilayer




def test_real_anomaly(netD, test_loader, total_embeddings, select_layers):


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


            features_list = []
            for i in select_layers:
                f = m(layerlist[i].detach())
                features_list.append(f)
            
            embedding_test = stpm_multilayer(features_list)


            n_neighbors=1
            
            trained_feature = total_embeddings

            distances = torch.cdist(embedding_test.cuda(), trained_feature.cuda(), p=2.0) 
            score_patches, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
            scores = np.mean(score_patches.cpu().numpy(), axis=-1)

            score = max(scores)
         

            size = int(math.sqrt(score_patches.shape[0]))
            anomaly_map = scores.reshape((size,size)) 
            
            
            gt = gt_mask.numpy()[0,0].astype(np.float32)
            anomaly_map_resized = cv2.resize(anomaly_map, (test_img.shape[-1], test_img.shape[-1]))


            gt_list_px_lvl.extend(gt.flatten())
            pred_list_px_lvl.extend(anomaly_map_resized.flatten())
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

        embedding_list = []

        for img, _, _ in train_loader:

            _, layerlist_n = netD(img.cuda())

            m = torch.nn.AvgPool2d(3, 1, 1)
            
            features_list = []
            for i in select_layers:

                f = m(layerlist_n[i].detach())
                features_list.append(f)
            
            embeddings = stpm_multilayer(features_list)

            embedding_list.extend(embeddings)

            

        total_embeddings = torch.vstack(embedding_list).cuda() 

        print('embedding size : ', total_embeddings.shape) 

        return total_embeddings, writer_dict



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
            prefix = prefix.replace(":", "")
            if object in prefix:
                print(prefix)
                path1 =  glob.glob('./ckpt/{obj}-{model}/e1600*'.format(obj=object, model=modelname))
                print(path1)
                for i in range(len(path1)):
                    if prefix in str(path1[i]):
                        print(path1[i])
                        ckpt_path = path1[i]
                

        select_layers_list = [ [1,2] ]  ## 0,1,2,3; 1,2 == layer2,layer3
        
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
            writer = SummaryWriter('./inference/concatencation-{t}-{obj}'.format(t=t, obj=object))


            dataset = MVTECdataset(dataset_dir, object, resize_wh=img_size)
            train_loader_all = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=24)

            select_layers = item

            total_embeddings, writer_dict = get_feature(train_loader_all, netD, writer_dict, select_layers=select_layers)
            metric_dict = test_real_anomaly(netD, test_loader, total_embeddings, select_layers=select_layers)

            writer_dict['layers'] = str(select_layers) 

            print(writer_dict)
            print(metric_dict)
            writer.add_hparams(writer_dict, metric_dict)
            writer.close()

            gc.collect()
            torch.cuda.empty_cache() 



