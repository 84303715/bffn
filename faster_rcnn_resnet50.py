import torch
import torch.nn as nn
from torchvision.ops import roi_pool
from torchvision import models
import torch.nn.functional as F

from au_crop_net_50 import FasterRCNNResnet50 as au_crop_50
from triplet_attention import TripletAttention

import numpy as np

class FasterRCNN(nn.Module):
    
    def __init__(self, extractor, extractor_roi, tail, res5_layer):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.extractor_roi = extractor_roi
        self._tail = tail
        self.res_tail = res5_layer

        self.model = au_crop_50()

        self.checkpoint = torch.load('./models/pretrain/au/resnet-50-epoch_198_precision_0.8163929566368591.pth')
        self.model.load_state_dict(self.checkpoint['model_state_dict'],strict=False)

    def extract(self, x):

        x = self.extractor(x)

        return x

    def forward(self, image, bbox):

        x = self.extract(image)

        x_rois = self.model(image, bbox)
        x_rois = self.extractor_roi(x, x_rois)

        return self._tail(x, x_rois)

class FasterRCNNResnet50(FasterRCNN):


    feat_stride = 16

    def __init__(self, pretrained_model='resnet50'):

        weight = nn.Parameter(torch.ones(7), requires_grad=False)
        extractor = ResnetFeatureExtractor(BottleNeck)
        extractor_roi = RoIBlock(roi_size=14, spatial_scale=1. / self.feat_stride, w=weight)
        tail = TailBlock()
        res5_layer = Res_5(BottleNeck)

        super(FasterRCNNResnet50, self).__init__(
            extractor,
            extractor_roi,
            tail,
            res5_layer
        )

        if pretrained_model == 'resnet50':
            self._copy_imagenet_pretrained_resnet50()
            print("Loading {} pretrained weights...".format(pretrained_model))

    def _copy_imagenet_pretrained_resnet50(self):
        model = models.resnet50()
        model.load_state_dict(torch.load('./models/pretrain/resnet50-19c8e357.pth'))
        self.extractor.conv1.load_state_dict(model.conv1.state_dict())
        self.extractor.bn1.load_state_dict(model.bn1.state_dict())
        self.extractor.res2.load_state_dict(model.layer1.state_dict())
        self.extractor.res3.load_state_dict(model.layer2.state_dict())
        self.extractor.res4.load_state_dict(model.layer3.state_dict())

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class ResnetFeatureExtractor(nn.Module):

    def __init__(self, block, input_channels=3):
        super(ResnetFeatureExtractor, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.res2 = self._make_layer(block, 64, 3)
        self.res3 = self._make_layer(block, 128, 4, 2)
        self.res4 = self._make_layer(block, 256, 6, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def _make_layer(self, block, out_channels, num_block, stride=1):

        downsample = None
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        return x

class Res_5(nn.Module):

    def __init__(self, block, input_channels=3):
        super(Res_5, self).__init__()
        self.in_channels = 1024
        
        self.res5 = self._make_layer(block, 512, 3, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def _make_layer(self, block, out_channels, num_block, stride=1):

        downsample = None
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):

        x = self.res5(input)

        return x

class RoIBlock(nn.Module):

    def __init__(self, roi_size, spatial_scale, w):
        super(RoIBlock, self).__init__()

        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.w = w
        # self.reduce_dim = nn.Sequential(
        #     nn.Conv2d(in_channels=7168, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(inplace=True),
        # )

    # def forward(self,x, rois, roi_indices):
    def forward(self, x, rois):
        batch_size = x.shape[0]
        # indices_and_rois = np.concatenate(
        #     (roi_indices[:, None], rois), axis=1)
        # rois = _roi_pooling_2d_yx(
        #     x, indices_and_rois, self.roi_size, self.spatial_scale)

        rois = rois.chunk(batch_size, dim=0)

        # W normalization
        # W_normalization = []
        # for i in range(7):
        #     W_normalization.append(torch.exp(self.w[i]) / torch.sum(torch.exp(self.w)))
        # print(W_normalization)
        # roi re-weight
        # for i, roi in enumerate(rois):
        #     for j in range(roi.shape[0]):
        #         roi[j] = roi[j] * W_normalization[j]
        # weighted_rois = []
        # for i, roi in enumerate(rois):
        #     for j in range(roi.shape[0]):
        #         t_list = []
        #         t = roi[j] * W_normalization[j]
        #         t_list.append(t)
        #     weighted_rois.append(t_list)

        c_temp = []
        for i, roi in enumerate(rois):
            temp = roi
            a = temp[0]
            for j in range(1, len(temp)):
                a = torch.cat([a + temp[j]], dim=0)
            c_temp.append(a)
        h = torch.stack(c_temp, 0)
        # h = self.reduce_dim(h)
        return h


class TailBlock(nn.Module):
    # N is batch size
    def __init__(self):
        super(TailBlock, self,).__init__()

        self.dropout_em = nn.Dropout(p=0.4)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.GAP = nn.AdaptiveAvgPool2d((1,1)) 
        # self.res5 = Res_5(BottleNeck)
        self.tripletattention = TripletAttention()
        self.pred_em = nn.Linear(in_features=1024,out_features=7,bias=True)

    def forward(self, x, x_rois):
        x = self.dropout_em(x)
        # x_rois = self.dropout_au(x_rois)
        featureMap0 = torch.cat([x, x_rois], dim=1)   
        featureMap1 = self.tripletattention(featureMap0)
        featureMap2 = self.reduce_dim(featureMap1)
        featureMap3 = self.maxpool(featureMap2)
        featureMap4 = self.GAP(featureMap3)
        feature = featureMap4.view(featureMap4.size(0),featureMap4.size(1))   
        h = self.pred_em(feature)  

        return h

def _roi_pooling_2d_yx(x, indices_and_rois, outh, spatial_scale):

    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    xy_indices_and_rois = torch.Tensor(xy_indices_and_rois).cuda()

    pool = roi_pool(
        x, xy_indices_and_rois, outh, spatial_scale)
    return pool

if __name__ == '__main__':
    model = Res_5(BottleNeck)
    # model = ResRoIHead(7, roi_size=14, spatial_scale=1. / 16)
    input = torch.randn(16, 1024, 14, 14)
    out = model(input)
    print(out.shape)

