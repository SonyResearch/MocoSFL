'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19']

NUM_CHANNEL_GROUP = 4
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

class VGGView(nn.Module): # This module is taken from https://github.com/joe-siyuan-qiao/WeightStandardization
    def __init__(self):
        super().__init__()

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, -1)
        out = input.view(shape)
        return out

class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, feature, num_client = 1, num_class = 10):
        super(VGG, self).__init__()
        self.expansion = 1
        self.current_client = 0
        self.cloud_classifier_merge = False
        self.local_list = []
        self.num_client = num_client
        for i in range(num_client):
            if i == 0:
                self.local_list.append(feature[0])
                self.local_list[0].apply(init_weights)
            else:
                new_copy = copy.deepcopy(self.local_list[0])

                self.local_list.append(new_copy)
        
        self.cloud = feature[1]
        
        classifier_list = [nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True)]
        classifier_list += [nn.Linear(512, num_class)]

        self.classifier = nn.Sequential(*classifier_list)

        self.original_num_cloud = self.get_num_of_cloud_layer()

        # Initialize weights
        self.cloud.apply(init_weights)
        self.classifier.apply(init_weights)
    
    def forward(self, x, client_id = 0):
        if self.cloud_classifier_merge:
            x = self.local_list[client_id](x)
            x = self.cloud(x)
        else:
            x = self.local_list[client_id](x)
            x = self.cloud(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x
    
    def __call__(self, x, client_id = 0):
        return self.forward(x, client_id)

    def merge_classifier_cloud(self):
        self.cloud_classifier_merge = True
        cloud_list = list(self.cloud.children())
        cloud_list.append(VGGView())
        classifier_list = list(self.classifier.children())
        cloud_list.extend(classifier_list)
        self.cloud = nn.Sequential(*cloud_list)

    def unmerge_classifier_cloud(self):
        self.cloud_classifier_merge = False
        cloud_list = list(self.cloud.children())
        orig_cloud_list = []
        for i, module in enumerate(cloud_list):
            if "VGGView" in str(module):
                break
            else:
                orig_cloud_list.append(module)
        self.cloud = nn.Sequential(*orig_cloud_list)

    def get_num_of_cloud_layer(self):
        num_of_cloud_layer = 0
        if not self.cloud_classifier_merge:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "Conv2d" in str(module) or "Linear" in str(module):
                    num_of_cloud_layer += 1
            num_of_cloud_layer += 3
        else:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "Conv2d" in str(module) or "Linear" in str(module):
                    num_of_cloud_layer += 1
        return num_of_cloud_layer

    def recover(self):
        if self.cloud_classifier_merge:
            self.resplit(self.original_num_cloud)
            self.unmerge_classifier_cloud()
            
    def resplit(self, num_of_cloud_layer):
        if not self.cloud_classifier_merge:
            self.merge_classifier_cloud()
            
        for i in range(self.num_client):
            list_of_layers = list(self.local_list[i].children())
            list_of_layers.extend(list(self.cloud.children()))
            entire_model_list = copy.deepcopy(list_of_layers)
            total_layer = 0
            for _, module in enumerate(entire_model_list):
                if "Conv2d" in str(module) or "Linear" in str(module):
                    total_layer += 1
            
            num_of_local_layer = (total_layer - num_of_cloud_layer)
            local_list = []
            local_count = 0
            cloud_list = []
            for _, module in enumerate(entire_model_list):
                if "Conv2d" in str(module) or "Linear" in str(module):
                    local_count += 1
                if local_count <= num_of_local_layer:
                    local_list.append(module)
                else:
                    cloud_list.append(module)
            
            self.local_list[i] = nn.Sequential(*local_list)
        self.cloud = nn.Sequential(*cloud_list)

    def get_smashed_data_size(self, batch_size = 1, input_size = 32):
        with torch.no_grad():
            noise_input = torch.randn([batch_size, 3, input_size, input_size])
            try:
                device = next(self.local_list[0].parameters()).device
                noise_input = noise_input.to(device)
            except:
                pass
            smashed_data = self.local_list[0](noise_input)
        return smashed_data.size()

    
def make_layers(cutting_layer,cfg, batch_norm=True, adds_bottleneck = False, bottleneck_option = "C8S1", group_norm = False, WS = True):
    local = []
    cloud = []
    in_channels = 3
    
    #Modified Local part - Experimental feature
    channel_mul = 1
    for v_idx,v in enumerate(cfg):
        if v_idx < cutting_layer - 1:
            if v == 'M':
                local += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, int(v * channel_mul), kernel_size=3, padding=1)
                
                if group_norm:
                    if WS:
                        local += [WSConv2d(in_channels, int(v * channel_mul), kernel_size=3, padding=1), nn.GroupNorm(num_groups = max(int(v * channel_mul)//NUM_CHANNEL_GROUP, 1), num_channels = int(v * channel_mul)), nn.ReLU(inplace=True)]
                    else:
                        local += [nn.Conv2d(in_channels, int(v * channel_mul), kernel_size=3, padding=1), nn.GroupNorm(num_groups = max(int(v * channel_mul)//NUM_CHANNEL_GROUP, 1), num_channels = int(v * channel_mul)), nn.ReLU(inplace=True)]
                
                elif batch_norm:
                    local += [conv2d, nn.BatchNorm2d(int(v * channel_mul)), nn.ReLU(inplace=True)]
                else:
                    local += [conv2d, nn.ReLU(inplace=True)]

                in_channels = int(v * channel_mul)
        elif v_idx == cutting_layer - 1:
            if v == 'M':
                local += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                
                if group_norm:
                    if WS:
                        local += [WSConv2d(in_channels, v, kernel_size=3, padding=1), nn.GroupNorm(num_groups = max(v//NUM_CHANNEL_GROUP, 1), num_channels = v), nn.ReLU(inplace=True)]
                    else:
                        local += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.GroupNorm(num_groups = max(v//NUM_CHANNEL_GROUP, 1), num_channels = v), nn.ReLU(inplace=True)]
                
                
                elif batch_norm:
                    local += [conv2d, nn.BatchNorm2d(int(v * channel_mul)), nn.ReLU(inplace=True)]
                else:
                    local += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            
            # Adding a pair of bottleneck layers for communication-efficiency
            if adds_bottleneck:
                print("original channel size of smashed-data is {}".format(in_channels))
                try:
                    if "noRELU" in bottleneck_option or "norelu" in bottleneck_option or "noReLU" in bottleneck_option:
                        relu_option = False
                    else:
                        relu_option = True
                    if "K" in bottleneck_option:
                        bn_kernel_size = int(bottleneck_option.split("C")[0].split("K")[1])
                    else:
                        bn_kernel_size = 3
                    bottleneck_channel_size = int(bottleneck_option.split("S")[0].split("C")[1])
                    if "S" in bottleneck_option:
                        bottleneck_stride = int(bottleneck_option.split("S")[1])
                    else:
                        bottleneck_stride = 1
                except:
                    print("auto extract bottleneck option fail (format: CxSy, x = [1, max_channel], y = {1, 2}), set channel size to 8 and stride to 1")
                    bn_kernel_size = 3
                    bottleneck_channel_size = 8
                    bottleneck_stride = 1
                    relu_option = True
                # cleint-side bottleneck
                if bottleneck_stride == 1:
                    local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
                elif bottleneck_stride >= 2:
                    local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
                    for _ in range(int(np.log2(bottleneck_stride//2))):
                        if relu_option:
                            local += [nn.ReLU()]
                        local += [nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
                if relu_option:
                    local += [nn.ReLU()]
                # server-side bottleneck
                if bottleneck_stride == 1:
                    cloud += [nn.Conv2d(bottleneck_channel_size, in_channels, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
                elif bottleneck_stride >= 2:
                    for _ in range(int(np.log2(bottleneck_stride//2))):
                        cloud += [nn.ConvTranspose2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                        if relu_option:
                            cloud += [nn.ReLU()]
                    cloud += [nn.ConvTranspose2d(bottleneck_channel_size, in_channels, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                if relu_option:
                    cloud += [nn.ReLU()]
                print("added bottleneck, new channel size of smashed-data is {}".format(bottleneck_channel_size))
        else:
            if v == 'M':
                cloud += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

                if group_norm:
                    if WS:
                        cloud += [WSConv2d(in_channels, v, kernel_size=3, padding=1), nn.GroupNorm(num_groups = max(v//NUM_CHANNEL_GROUP, 1), num_channels = v), nn.ReLU(inplace=True)]
                    else:
                        cloud += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.GroupNorm(num_groups = max(v//NUM_CHANNEL_GROUP, 1), num_channels = v), nn.ReLU(inplace=True)]
                elif batch_norm:
                    cloud += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    cloud += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

    return nn.Sequential(*local), nn.Sequential(*cloud)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

def vgg11(cutting_layer, num_client = 1, num_class = 10, group_norm=False, batch_norm=True, input_size= 32, adds_bottleneck = False, bottleneck_option = "C8S1", c_residual = False, WS = True):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cutting_layer,cfg['A'], batch_norm=batch_norm, group_norm=group_norm, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, WS = WS), num_client = num_client, num_class = num_class)

def vgg13(cutting_layer, num_client = 1, num_class = 10, group_norm=False, batch_norm=True, input_size= 32, adds_bottleneck = False, bottleneck_option = "C8S1", c_residual = False, WS = True):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cutting_layer,cfg['B'], batch_norm=batch_norm, group_norm=group_norm, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, WS = WS), num_client = num_client, num_class = num_class)

def vgg16(cutting_layer, num_client = 1, num_class = 10, group_norm=False, batch_norm=True, input_size= 32, adds_bottleneck = False, bottleneck_option = "C8S1", c_residual = False, WS = True):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cutting_layer,cfg['D'], batch_norm=batch_norm, group_norm=group_norm, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, WS = WS), num_client = num_client, num_class = num_class)

def vgg19(cutting_layer, num_client = 1, num_class = 10, group_norm=False, batch_norm=True, input_size= 32, adds_bottleneck = False, bottleneck_option = "C8S1", c_residual = False, WS = True):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cutting_layer,cfg['E'], batch_norm=batch_norm, group_norm=group_norm, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, WS = WS), num_client = num_client, num_class = num_class)
