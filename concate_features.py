import torch
from torch.nn import functional as F
import numpy as np

#https://github.com/hcw-00/PatchCore_anomaly_detection/blob/eeb4d676a06fc2c56ca98aa1fe70b9d9b44edc11/train.py

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])

    return embedding_list


def generate_embedding(features1, feature2):
    """Generate embedding from hierarchical feature map.
    Args:
        features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
        features: Dict[str:Tensor]:
    Returns:
        Embedding vector
    """
    feature2 = F.interpolate(feature2, size=features1.shape[-2:], mode="nearest")
    embeddings = torch.cat((features1, feature2), 1)

    return embeddings

def stpm(feature1, feature2):
    
    embedding = generate_embedding(feature1, feature2)
    embedding_size = embedding.size(1)
    embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    return embedding

def stpm_cpu(feature1, feature2):
    # numpy array
    embedding = embedding_concat(feature1, feature2)
    embedding = np.array(embedding)
    embedding = reshape_embedding(embedding) 
    return embedding



def stpm_multilayer(features_list):
    embedding = generate_multi_embedding(features_list)
    embedding_size = embedding.size(1)
    embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    return embedding


def generate_multi_embedding(features_list):
    """Generate embedding from hierarchical feature map.
    Args:
        features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
        features: Dict[str:Tensor]:
    Returns:
        Embedding vector
    """
    embeddings = features_list[0]
    for i in range(1, len(features_list)):
        features_list[i] = F.interpolate(features_list[i], size=features_list[0].shape[-2:], mode="nearest")
        embeddings = torch.cat((embeddings, features_list[i]), 1)

    return embeddings
