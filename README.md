# Self-Supervised Image Anomaly Detection and Localization with Synthetic Anomalies

# Abstract
In visual anomaly detection, anomalies are often rare and unpredictable. For this reason, we aim to build a detection framework that can detect unseen anomalies with only anomaly-free examples.
In this paper, we introduce a two-stage framework for detecting and localizing anomalies in images using self-supervised learning. We simulate anomalies through the designed augmentation strategies, and the model learns to distinguish normal data from synthetic anomalies. In addition, we compare two methods for combining representations from different semantic levels of our network, and both of the methods obtain competitive results for defect detection. 
Without extra training samples and pre-trained models, the proposed approach achieves 96.4% detection AUROC and 96.1% localization AUROC on the MVTec AD benchmark, which is competitive against existing unsupervised methods. The results demonstrate the potential of our method for industrial applications.

# Usage

## Benchmark
download mvtec ad dataset 

[MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

and save the dataset in folder

`
mvtec_anomaly_detection
`

## training
train all subdataset
```
python main.py
```
train one subdataset, for example: zipper
```
python main.py --c zipper
```
checkpoints will be saved in ckpt folder
```
├── ckpt
│   ├── subpath
│   │   ├── model.pt
```
## trained models
the trained models for our experiment results were uploaded in ```ckpt``` folder,  
due to file size limits on github,  
better download [here](https://drive.google.com/drive/folders/1PXFR3b30GrkI1YsP1OfVtdiRmFzlho6i?usp=sharing)  

## inference
two ways for aggregating multi-level features:
### concatenation method
```
python inference_concatenation.py --c <class_name> --p <ckpt_subpath>
```
for example:
```
python inference_concatenation.py --c zipper --p zipper-wave50
```
### multiplication method
```
python inference_multiplication.py --c <class_name> --p <ckpt_subpath>
```
for example:
```
python inference_multiplication.py --c leather --p leather-wave50
```

# Code Reference

[WaveCNet](https://github.com/LiQiufu/WaveCNet)  
[anomalib](https://github.com/openvinotoolkit/anomalib)  
[hcw-00: PatchCore](https://github.com/hcw-00/PatchCore_anomaly_detection)  
[Runinho: CutPaste](https://github.com/Runinho/pytorch-cutpaste)  
