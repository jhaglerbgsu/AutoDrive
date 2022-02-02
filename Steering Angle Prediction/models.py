import torch
import torch.nn as nn 
from collections import namedtuple
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Any
import torch.nn.functional as F
import torchvision 
from torchvision.models import googlenet

#import resnet18
#import resnet50
#import googlenet



class GoogLeNet (nn.Module):
    """
    
    """

    __constants__ = ['aux_logits', 'transform_input']

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None
    ) -> None:
        super(GoogLeNet, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            init_weights = False
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.inception3c = inception_block(320, 192, 160, 256, 48, 160, 96) # Module Step 1
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.inception4f = inception_block(528, 256, 160, 320, 32, 128, 128) # Module Step 2
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.inception5c = inception_block(832, 384, 192, 384, 48, 128, 128) # Module Step 3

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x


class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)
        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
  

class TruckResnet18(nn.Module):
    """
    
    """

    def __init__(self):
        super(TruckResnet18, self).__init__()

        self.resnet18 = resnet18(pretrained=True)
        self.freeze_params(self.resnet18)
        self.resnet18.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(512, 256),                                   # N x 2048 -> N x 512 /  N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                    # N x 512 -> N x 256 / N x 256 -> N x 64 
            nn.ELU(),
            nn.Linear(64, 32),                                     # N x 256 -> N x 64 / N x 64 -> N x 32 
            nn.ELU()
        )

        self.out = nn.Linear(32, 1)                                 # N x 64 -> N x 1 / N x 32 -> N x 1 

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet18(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x
    

class TruckResnet34(nn.Module):
    """
    
    """

    def __init__(self):
        super(TruckResnet34, self).__init__()

        self.resnet34 = resnet34(pretrained=True)
        self.freeze_params(self.resnet34)
        self.resnet34.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(512, 256),                                   # N x 2048 -> N x 512 /  N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                    # N x 512 -> N x 256 / N x 256 -> N x 64 
            nn.ELU(),
            nn.Linear(64, 32),                                     # N x 256 -> N x 64 / N x 64 -> N x 32 
            nn.ELU()
        )

        self.out = nn.Linear(32, 1)                                 # N x 64 -> N x 1 / N x 32 -> N x 1 

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet34(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x    
    
    
    
    
class TruckResnet50(nn.Module):
    """
    A modified CNN model, leverages the pretrained resnet50 for features extraction https://arxiv.org/abs/1512.00567
    Transfer Learning from pretrained Resnet-50, connected with 3 dense layers. 
    Total params: 24.7M (24704961), pretrained 14.5M (14582848), trainable 10.1M (10122113)
 
    """

    def __init__(self):
        super(TruckResnet50, self).__init__()

        self.resnet50 = resnet50(pretrained=True)
        self.freeze_params(self.resnet50)
        self.resnet50.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),                                   # N x 2048 -> N x 512
            nn.ELU(),
            nn.Linear(512, 256),                                    # N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                     # N x 256 -> N x 64
            nn.ELU()
        )

        self.out = nn.Linear(64, 1)                                 # N x 64 -> N x 1

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet50(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x
    
class TruckResnet101(nn.Module):
    """
    A modified CNN model, leverages the pretrained resnet101 for features extraction https://arxiv.org/abs/1512.00567
    Transfer Learning from pretrained Resnet-101, connected with 3 dense layers. 
    Total params: 24.7M (24704961), pretrained 14.5M (14582848), trainable 10.1M (10122113)
 
    """

    def __init__(self):
        super(TruckResnet101, self).__init__()

        self.resnet101 = resnet101(pretrained=True)
        self.freeze_params(self.resnet101)
        self.resnet101.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),                                   # N x 2048 -> N x 512
            nn.ELU(),
            nn.Linear(512, 256),                                    # N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                     # N x 256 -> N x 64
            nn.ELU()
        )

        self.out = nn.Linear(64, 1)                                 # N x 64 -> N x 1

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet101(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x    

class TruckResnet151(nn.Module):
    """
    A modified CNN model, leverages the pretrained resnet151 for features extraction https://arxiv.org/abs/1512.00567
    Transfer Learning from pretrained Resnet-151, connected with 3 dense layers. 
    Total params: 24.7M (24704961), pretrained 14.5M (14582848), trainable 10.1M (10122113)
 
    """

    def __init__(self):
        super(TruckResnet151, self).__init__()

        self.resnet151 = resnet151(pretrained=True)
        self.freeze_params(self.resnet151)
        self.resnet151.fc = nn.Identity()                            # N x 3 x 224 x 224 -> N x 2048

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),                                   # N x 2048 -> N x 512
            nn.ELU(),
            nn.Linear(512, 256),                                    # N x 512 -> N x 256
            nn.ELU(),
            nn.Linear(256, 64),                                     # N x 256 -> N x 64
            nn.ELU()
        )

        self.out = nn.Linear(64, 1)                                 # N x 64 -> N x 1

    def freeze_params(self, model):
        count = 0
        for param in model.parameters():
            count += 1
            if count <= 141:
                param.requires_grad = False

    def forward(self, x):
        
        x = x.view(x.size(0), 3, 224, 224)                          # N x 3 x H x W, H = 224, W = 224

        x = self.resnet151(x)                                        # N x 2048

        # input dimension needs to be monitored
        x = self.fc(x)                                              # N x 64

        x = self.out(x)                                             # N x 1

        return x        
