'''MobileNetV2 in PyTorch.
Fetched from https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.


'''
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

NUM_CHANNEL_GROUP = 4

class MobView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = F.avg_pool2d(input, 4)
        batch_size = input.size(0)
        shape = (batch_size, -1)
        out = out.view(shape)
        return out

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, residual = True, WS = True):
        super(Block, self).__init__()
        self.stride = stride
        self.residual = residual
        self.expansion = expansion
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if residual and stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out = out + self.shortcut(x) if self.stride==1 else out
        return out

class Block_gn(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, residual = True, WS = True):
        super(Block_gn, self).__init__()
        self.stride = stride
        self.residual = residual
        self.expansion = expansion
        planes = expansion * in_planes
        if WS:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        if WS:
            self.conv2 = WSConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        if WS:
            self.conv3 = WSConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.GroupNorm(num_groups = max(out_planes//NUM_CHANNEL_GROUP, 1), num_channels = out_planes)

        self.shortcut = nn.Sequential()
        if residual and stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out = out + self.shortcut(x) if self.stride==1 else out
        return out

class WSConv2d(nn.Conv2d): # This module is taken from https://github.com/joe-siyuan-qiao/WeightStandardization

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


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

class MobileNet(nn.Module):
    

    def __init__(self, feature, expansion = 1, num_client = 1, num_class = 10):
        super(MobileNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.current_client = 0
        self.num_client = num_client
        self.expansion = expansion
        self.local_list = []
        for i in range(num_client):
            if i == 0:
                self.local_list.append(feature[0])
                self.local_list[0].apply(init_weights)
            else:
                new_copy = copy.deepcopy(self.local_list[0])

                self.local_list.append(new_copy)


        self.cloud = feature[1]
        self.classifier = nn.Linear(1280*self.expansion, num_class)
        self.cloud_classifier_merge = False
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
            # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

    def __call__(self, x, client_id = 0):
        return self.forward(x, client_id)

    def merge_classifier_cloud(self):
        self.cloud_classifier_merge = True
        cloud_list = list(self.cloud.children())
        cloud_list.append(MobView())
        cloud_list.append(self.classifier)
        self.cloud = nn.Sequential(*cloud_list)

    def unmerge_classifier_cloud(self):
        self.cloud_classifier_merge = False
        cloud_list = list(self.cloud.children())
        orig_cloud_list = []
        for i, module in enumerate(cloud_list):
            if "MobView" in str(module):
                break
            else:
                orig_cloud_list.append(module)
        self.cloud = nn.Sequential(*orig_cloud_list)

    def get_num_of_cloud_layer(self):
        num_of_cloud_layer = 0
        if not self.cloud_classifier_merge:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if ("Conv2d" in str(module) and "Block" not in str(module)) or "Linear" in str(module) or "Block" in str(module):
                    num_of_cloud_layer += 1
            num_of_cloud_layer += 1
        else:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if ("Conv2d" in str(module) and "Block" not in str(module)) or "Linear" in str(module) or "Block" in str(module):
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
            total_layer = 0
            for _, module in enumerate(list_of_layers):
                print(str(module))
                if ("Conv2d" in str(module) and "Block" not in str(module)) or "Linear" in str(module) or "Block" in str(module):
                    total_layer += 1
            
            num_of_local_layer = (total_layer - num_of_cloud_layer)
            local_list = []
            local_count = 0
            cloud_list = []
            for _, module in enumerate(list_of_layers):
                if ("Conv2d" in str(module) and "Block" not in str(module)) or "Linear" in str(module) or "Block" in str(module):
                    local_count += 1
                if local_count <= num_of_local_layer:
                    local_list.append(module)
                else:
                    cloud_list.append(module)
            self.local_list[i] = nn.Sequential(*local_list)
        self.cloud = nn.Sequential(*cloud_list)

    def get_smashed_data_size(self, batch_size = 1, input_size = 32):
        self.local_list[0].eval()
        with torch.no_grad():
            noise_input = torch.randn([batch_size, 3, input_size, input_size])
            try:
                device = next(self.local_list[0].parameters()).device
                noise_input = noise_input.to(device)
            except:
                pass
            smashed_data = self.local_list[0](noise_input)
        return smashed_data.size()

# (expansion, out_planes, num_blocks, stride)
cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

def make_layers(cutting_layer, cfg, in_planes, adds_bottleneck = False, bottleneck_option = "C8S1", group_norm = False, residual = True, WS = True):
        local_layer_list = []
        cloud_layer_list = []
        current_layer = 0
        in_channels = 3
        if cutting_layer > current_layer:
            if group_norm and WS:
                local_layer_list.append(WSConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False))
            else:
                local_layer_list.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False))
            if not group_norm:
                local_layer_list.append(nn.BatchNorm2d(32))
            else:
                local_layer_list.append(nn.GroupNorm(num_groups = max(32//NUM_CHANNEL_GROUP, 1), num_channels = 32))
            in_channels = 32
        else:
            if group_norm and WS:
                cloud_layer_list.append(WSConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False))
            else:
                cloud_layer_list.append(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False))
            if not group_norm:
                cloud_layer_list.append(nn.BatchNorm2d(32))
            else:
                cloud_layer_list.append(nn.GroupNorm(num_groups = max(32//NUM_CHANNEL_GROUP, 1), num_channels = 32))
        
        for expansion, out_planes, num_blocks, stride in cfg:
            current_layer += 1
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                if cutting_layer > current_layer:
                    if not group_norm:
                        local_layer_list.append(Block(in_planes, out_planes, expansion, stride, residual, WS))
                    else:
                        local_layer_list.append(Block_gn(in_planes, out_planes, expansion, stride, residual, WS))
                    in_channels = out_planes
                else:
                    if not group_norm:
                        cloud_layer_list.append(Block(in_planes, out_planes, expansion, stride, residual, WS))
                    else:
                        cloud_layer_list.append(Block_gn(in_planes, out_planes, expansion, stride, residual, WS))
                in_planes = out_planes
        current_layer += 1
        if cutting_layer > current_layer:
            if group_norm and WS:
                local_layer_list.append(WSConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False))
            else:
                local_layer_list.append(nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False))
            if not group_norm:
                local_layer_list.append(nn.BatchNorm2d(1280))
            else:
                local_layer_list.append(nn.GroupNorm(num_groups = max(1280//NUM_CHANNEL_GROUP, 1), num_channels = 1280))
        else:
            if group_norm and WS:
                cloud_layer_list.append(WSConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False))
            else:
                cloud_layer_list.append(nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False))
            if not group_norm:    
                cloud_layer_list.append(nn.BatchNorm2d(1280))
            else:
                cloud_layer_list.append(nn.GroupNorm(num_groups = max(1280//NUM_CHANNEL_GROUP, 1), num_channels = 1280))

        local = []
        cloud = []
        if adds_bottleneck: # to enable gooseneck, simply copy below to other architecture
            print("original channel size of smashed-data is {}".format(in_channels))
            try:
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
            # cleint-side bottleneck
            if bottleneck_stride == 1:
                local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
            elif bottleneck_stride >= 2:
                local += [nn.Conv2d(in_channels, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
                for _ in range(int(np.log2(bottleneck_stride//2))):
                    local += [nn.ReLU()]
                    local += [nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
            local += [nn.ReLU()]
            # server-side bottleneck
            if bottleneck_stride == 1:
                cloud += [nn.Conv2d(bottleneck_channel_size, in_channels, kernel_size=1, stride= 1)]
            elif bottleneck_stride >= 2:
                for _ in range(int(np.log2(bottleneck_stride//2))):
                    cloud += [nn.ConvTranspose2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                    cloud += [nn.ReLU()]
                cloud += [nn.ConvTranspose2d(bottleneck_channel_size, in_channels, kernel_size=3, output_padding=1, padding=1, stride= 2)]
            cloud += [nn.ReLU()]
            print("added bottleneck, new channel size of smashed-data is {}".format(bottleneck_channel_size))
        local_layer_list += local
        cloud_layer_list = cloud + cloud_layer_list

        return nn.Sequential(*local_layer_list), nn.Sequential(*cloud_layer_list)


def MobileNetV2(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    return MobileNet(make_layers(cutting_layer,cfg, in_planes=input_size, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, residual=c_residual, WS = WS), expansion= 1, num_client = num_client, num_class = num_class)

# def test():
#     net = MobileNetV2(9)
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     # print(y.size())

# test()
