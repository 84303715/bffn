import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_pool

import numpy as np

class FasterRCNN(nn.Module):
    
    def __init__(self, extractor, extractor_roi, tail, res_tail):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.extractor_roi = extractor_roi
        self.tail = tail
        self.res_tail = res_tail

    def extract_roi(self, image, bbox):

        x = image
        b, _, o_H, o_W = x.shape

        bbox = resize_bbox(bbox, (o_H, o_W), (o_H, o_W))

        roi_feature = self.forward(x, bbox)

        print("roi_feature.shape ", roi_feature.shape)
        return roi_feature

    def forward(self, x, bboxes):

        _, _, H, W = x.shape
        h = self.extractor(x) # h is tensor

        rois = list()
        roi_indices = list()
        for n in range(x.shape[0]): # n is batch index
            bbox = bboxes[n].data
            bbox = bbox.cpu().numpy()

            bbox[:, 0::2] = np.clip(
                bbox[:, 0::2], 0, H)  
            bbox[:,  1::2] = np.clip(
                bbox[:,  1::2], 0, W)

            rois.extend(bbox.tolist())
            roi_indices.extend((n * np.ones(bbox.shape[0])).tolist())
        rois = np.asarray(rois, dtype=np.float32)
        roi_indices = np.asarray(roi_indices, dtype=np.int32)

        f = self.extractor_roi(h, rois, roi_indices)
        f = self.tail(f)
        return f


class FasterRCNNResnet18(FasterRCNN):


    feat_stride = 16

    def __init__(self, pretrained_model='resnet34'):

        extractor = ResNet(BasicBlock)
        extractor_roi = RoIBlock(roi_size=14, spatial_scale=1. / self.feat_stride)
        tail = TailBlock()
        res_tail = Resnet_tail(BasicBlock)

        super(FasterRCNNResnet18, self).__init__(
            extractor,
            extractor_roi,
            tail,
            res_tail
        )

        if pretrained_model == 'resnet34':
            self._copy_imagenet_pretrained_resnet34()
            print("Loading {} pretrained weights...".format(pretrained_model))

    # def _copy_imagenet_pretrained_resnet18(self):

    #     model = models.resnet18()
    #     pretrained_resnet18 = torch.load('./resnet18_msceleb.pth')
    #     model.load_state_dict(pretrained_resnet18['state_dict'],strict=False)
    #     self.extractor.conv1.load_state_dict(model.conv1.state_dict())
    #     self.extractor.layer1.load_state_dict(model.layer1.state_dict())
    #     self.extractor.layer2.load_state_dict(model.layer2.state_dict())
    #     self.extractor.layer3.load_state_dict(model.layer3.state_dict())

    #     self.res_tail.layer4.load_state_dict(model.layer4.state_dict())

    def _copy_imagenet_pretrained_resnet34(self):
        model = models.resnet34()
        model.load_state_dict(torch.load('./models/pretrain/resnet34-b627a593.pth'))
        self.extractor.conv1.load_state_dict(model.conv1.state_dict())
        self.extractor.layer1.load_state_dict(model.layer1.state_dict())
        self.extractor.layer2.load_state_dict(model.layer2.state_dict())
        self.extractor.layer3.load_state_dict(model.layer3.state_dict())

class BasicBlock(nn.Module):#resnet 18\34 层
    expansion = 1

    # downsample 对应有没有虚线的结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel) # 输出特征矩阵的深度
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample # 传递下采样方法# 这个是shortcut的操作

    def forward(self, x): 
        identity = x 
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) 

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):# 网络结构
    def __init__(self, block):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, 3)  # conv2_x
        self.layer2 = self._make_layer(block, 128, 4, stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, 6, stride=2)  # conv4_x
        # self.layer4 = self._make_layer(block, 512, 2, stride=2)  # conv5_x


        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)



        return x 


class Resnet_tail(nn.Module):
    def __init__(self, block):
        super(Resnet_tail, self).__init__()
        self.in_channel = 256
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channel, block_num, stride=1):
            downsample = None
            if stride != 1 or self.in_channel != channel * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channel * block.expansion))
            
            layers = []
            layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
            self.in_channel = channel * block.expansion
            
            for _ in range(1, block_num):
                layers.append(block(self.in_channel, channel))

            return nn.Sequential(*layers)

    def forward(self, x):

        out = self.layer4(x)
        return out

class RoIBlock(nn.Module):

    def __init__(self, roi_size, spatial_scale):
        super(RoIBlock, self).__init__()

        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def forward(self, x, rois, roi_indices):
        batch_size = x.shape[0]
        indices_and_rois = np.concatenate(
            (roi_indices[:, None], rois), axis=1)
        rois = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.spatial_scale)
        rois = rois.chunk(batch_size, dim=0)
        c_temp = []
        for i, roi in enumerate(rois):
            temp = roi
            a = temp[0]
            for j in range(1, len(temp)):
                a = torch.cat([a + temp[j]], dim=0)
            c_temp.append(a)
        h = torch.stack(c_temp, 0) 
        return h

class TailBlock(nn.Module):
    # N is batch size
    def __init__(self):
        super(TailBlock, self,).__init__()

        self.resnet_layer_4 = Resnet_tail(BasicBlock)
        self.pool = nn.AvgPool2d(7, stride=2)
        self.fc = nn.Linear(512, 12, bias=True)
        

    def forward(self, x_rois):

        x_rois = self.resnet_layer_4(x_rois)


        h = x_rois

        
        h = self.pool(h)
        h = h.view(h.shape[0], h.shape[1])
        
        h = self.fc(h)

        return h

def _roi_pooling_2d_yx(x, indices_and_rois, outh, spatial_scale):

    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    xy_indices_and_rois = torch.Tensor(xy_indices_and_rois).cuda()

    pool = roi_pool(
        x, xy_indices_and_rois, outh, spatial_scale)
    return pool

def resize_bbox(bbox, in_size, out_size):

    bbox = bbox.clone()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox
