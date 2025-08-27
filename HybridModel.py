import logging
import time
import gc
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.stats import ttest_rel
from tqdm import tqdm
from vit_pytorch import *
import torch
from vit_pytorch import ViT
from vit_pytorch.simple_vit import SimpleViT
from vit_pytorch.mae import MAE
from vit_pytorch.dino import Dino
from pretraining.dcg import DCG as LocalGlobalFeatureExtractor
from pretraining.dcg import SimpleAggregator
from pretraining.resnet import ResNet18
from utils import *
from vit import ViTFeatureForClassfication
plt.style.use('ggplot')

# Ensure that the constraints are still satisfied




class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.features = LocalGlobalFeatureExtractor(config)       
        self.Classifier_Regressor = SimpleAggregator()

    # Compute guiding prediction as a feature extractor helper
    def compute_LocalGlobalFeature(self, x):
        y_fusion, y_global, y_local = self.features(x)
        return y_fusion, y_global, y_local

    def forward(self, x):
        _,y_global, y_local = self.compute_LocalGlobalFeature(x)  # (batch_size, n_classes)

        # Concatenate the tensors along the second dimension
        clf_output, reg_output = self.Classifier_Regressor(y_global, y_local)

        return clf_output, reg_output

