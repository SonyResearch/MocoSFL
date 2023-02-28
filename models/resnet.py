'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
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
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        # out = F.avg_pool2d(input, 4)
        out = self.avgpool(input)
        batch_size = input.size(0)
        shape = (batch_size, -1)
        out = out.view(shape)
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
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, residual = True, WS = True):
        super(BasicBlock, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if residual and (stride != 1 or in_planes != self.expansion*planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_gn(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, residual = True, WS = True):
        super(BasicBlock_gn, self).__init__()
        self.residual = residual
        if WS:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        if WS:
            self.conv2 = WSConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)

        self.shortcut = nn.Sequential()

        if residual and (stride != 1 or in_planes != self.expansion*planes):
            if WS:
                self.shortcut = nn.Sequential(
                    WSConv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups = max(self.expansion*planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups = max(self.expansion*planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, residual = True, WS = True):
        super(Bottleneck, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if residual and (stride != 1 or in_planes != self.expansion*planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_gn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, residual = True, WS = True):
        super(Bottleneck_gn, self).__init__()
        self.residual = residual
        if WS:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        if WS:
            self.conv2 = WSConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        if WS:
            self.conv3 = WSConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)


        self.bn3 = nn.GroupNorm(num_groups = max(self.expansion * planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion * planes)

        self.shortcut = nn.Sequential()

        if residual and (stride != 1 or in_planes != self.expansion*planes):
            self.shortcut = nn.Sequential(
                WSConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups = max(self.expansion*planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class conv3x3(nn.Module):
    def __init__(self, in_planes, planes, input_size=32):
        super(conv3x3, self).__init__()
        if input_size == 224:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=7, stride = 2, padding = 3, bias=False)
        elif input_size == 64:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride = 2, padding = 1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride = 1, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out

class conv3x3_gn(nn.Module):
    def __init__(self, in_planes, planes, input_size=32):
        super(conv3x3_gn, self).__init__()
        if input_size == 224:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=7, stride = 2, padding = 3, bias=False)
        elif input_size == 64:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, stride = 2, padding = 1, bias=False)
        else:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, stride = 1, padding = 1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out

class ResNet(nn.Module):
    '''
    ResNet model 
    '''
    def __init__(self, feature, expansion = 1, num_client = 1, num_class = 10, input_size = 32):
        super(ResNet, self).__init__()
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
        self.classifier = nn.Linear(512*expansion, num_class)
        self.cloud_classifier_merge = False
        self.original_num_cloud = self.get_num_of_cloud_layer()

        # Initialize weights
        self.cloud.apply(init_weights)
        self.classifier.apply(init_weights)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x, client_id = 0):
        if self.cloud_classifier_merge:
            x = self.local_list[client_id](x)
            x = self.cloud(x)
        else:
            x = self.local_list[client_id](x)
            x = self.cloud(x)
            # x = F.avg_pool2d(x, 4)
            x = self.avg_pool(x)
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
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module) or "BottleNeck" in str(module):
                    num_of_cloud_layer += 1
            num_of_cloud_layer += 1
        else:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module) or "BottleNeck" in str(module):
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
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module) or "BottleNeck" in str(module):
                    total_layer += 1
            
            num_of_local_layer = (total_layer - num_of_cloud_layer)
            local_list = []
            local_count = 0
            cloud_list = []
            for _, module in enumerate(entire_model_list):
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module) or "BottleNeck" in str(module):
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
    
    


def make_layers(block, layer_list, cutting_layer, adds_bottleneck = False, bottleneck_option = "C8S1", group_norm = False, input_size = 32, residual = True, WS = True):

    layers = []
    current_image_dim = input_size
    count = 1
    if not group_norm:
        layers.append(conv3x3(3, 64, input_size))
    else:
        layers.append(conv3x3_gn(3, 64, input_size))
    in_planes = 64

    strides = [1] + [1]*(layer_list[0]-1)
    for stride in strides:
        if count >= cutting_layer:
            residual = True
        layers.append(block(in_planes, 64, stride, residual, WS))
        count += 1
        current_image_dim = current_image_dim // stride

        in_planes = 64 * block.expansion

    strides = [2] + [1]*(layer_list[1]-1)
    for stride in strides:
        if count >= cutting_layer:
            residual = True
        layers.append(block(in_planes, 128, stride, residual, WS))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 128 * block.expansion

    strides = [2] + [1]*(layer_list[2]-1)
    for stride in strides:
        if count >= cutting_layer:
            residual = True
        layers.append(block(in_planes, 256, stride, residual, WS))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 256 * block.expansion

    strides = [2] + [1]*(layer_list[3]-1)
    for stride in strides:
        if count >= cutting_layer:
            residual = True
        layers.append(block(in_planes, 512, stride, residual, WS))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 512 * block.expansion
    try:
        local_layer_list = layers[:cutting_layer]
        cloud_layer_list = layers[cutting_layer:]
    except:
        print("Cutting layer is greater than overall length of the ResNet arch! set cloud to empty list")
        local_layer_list = layers[:]
        cloud_layer_list = []

    # Adding a pair of bottleneck layers for communication-efficiency
    temp_local = nn.Sequential(*local_layer_list)
    with torch.no_grad():
        noise_input = torch.randn([1, 3, input_size, input_size])
        smashed_data = temp_local(noise_input)
    input_nc = smashed_data.size(1)

    local = []
    cloud = []
    if adds_bottleneck: # to enable gooseneck, simply copy below to other architecture
        print("original channel size of smashed-data is {}".format(input_nc))
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
            local += [nn.Conv2d(input_nc, bottleneck_channel_size, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
        elif bottleneck_stride >= 2:
            local += [nn.Conv2d(input_nc, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
            for _ in range(int(np.log2(bottleneck_stride//2))):
                if relu_option:
                    local += [nn.ReLU()]
                local += [nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
        if relu_option:
            local += [nn.ReLU()]
        # server-side bottleneck
        if bottleneck_stride == 1:
            cloud += [nn.Conv2d(bottleneck_channel_size, input_nc, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
        elif bottleneck_stride >= 2:
            for _ in range(int(np.log2(bottleneck_stride//2))):
                cloud += [nn.ConvTranspose2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                if relu_option:
                    cloud += [nn.ReLU()]
            cloud += [nn.ConvTranspose2d(bottleneck_channel_size, input_nc, kernel_size=3, output_padding=1, padding=1, stride= 2)]
        if relu_option:
            cloud += [nn.ReLU()]
        print("added bottleneck, new channel size of smashed-data is {}".format(bottleneck_channel_size))
        input_nc = bottleneck_channel_size
    local_layer_list += local
    cloud_layer_list = cloud + cloud_layer_list
    local = nn.Sequential(*local_layer_list)
    cloud = nn.Sequential(*cloud_layer_list)

    return local, cloud

def ResNet18(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(BasicBlock, [2, 2, 2, 2], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(BasicBlock_gn, [2, 2, 2, 2], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), num_client = num_client, num_class = num_class, input_size = input_size)

def ResNet34(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(BasicBlock, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(BasicBlock_gn, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), num_client = num_client, num_class = num_class, input_size = input_size)

def ResNet50(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(Bottleneck, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(Bottleneck_gn, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)

def ResNet101(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(Bottleneck, [3, 4, 23, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(Bottleneck_gn, [3, 4, 23, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)

def ResNet152(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(Bottleneck, [3, 8, 36, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(Bottleneck_gn, [3, 8, 36, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)

