import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
from collections import OrderedDict

from ciconv2d import CIConv2d
from torchvision.models.vgg import __all__, model_urls, make_layers, cfgs

model_urls['w_vgg16'] = 'https://gitlab.tudelft.nl/attilalengyel/ciconv/-/raw/master/model_zoo/w_vgg16.pth'


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        invariant: str = None,
        num_classes: int = 1000,
        init_weights: bool = True,
        return_features: bool = False
    ) -> None:
        super(VGG, self).__init__()

        if invariant:
            self.ciconv = CIConv2d(invariant=invariant)
            features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        else:
            self.ciconv = None

        self.features = features
        self.return_features = return_features

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ciconv:
            x = self.ciconv(x)
        x = self.features(x)
        if not self.return_features:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def _vgg(arch: str, invariant: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), invariant=invariant, **kwargs)

    if pretrained:
        arch_try = invariant.lower()+'_'+arch if invariant else arch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # Check if pre-trained color invariants weights are available
            state_dict = load_state_dict_from_url(model_urls[arch_try], progress=progress, map_location=device)
        except Exception as e:
            print(e)
            # Else load pre-trained RGB weights
            print('Color invariant weights not found, loading Torchvision weights...')
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=device)
            state_dict['features.0.weight'] = torch.sum(state_dict['features.0.weight'],
                                                        dim=1, keepdim=True)
        if 'num_classes' in kwargs.keys():
            if kwargs['num_classes'] != 1000:
                print('Skipping fc layer parameters.')
                k_del = [k for k in state_dict.keys() if 'classifier' in k]
                for k in k_del:
                    state_dict.pop(k, None)
        # Load weights and print result
        r = model.load_state_dict(state_dict, strict=False)
        print(r)
    return model


def vgg11(invariant: str = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        invariant (str): Color invariant in CIConv layer, or None
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', invariant, 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(invariant: str = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        invariant (str): Color invariant in CIConv layer, or None
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', invariant, 'A', True, pretrained, progress, **kwargs)


def vgg13(invariant: str = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        invariant (str): Color invariant in CIConv layer, or None
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', invariant, 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(invariant: str = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        invariant (str): Color invariant in CIConv layer, or None
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', invariant, 'B', True, pretrained, progress, **kwargs)


def vgg16(invariant: str = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        invariant (str): Color invariant in CIConv layer, or None
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', invariant, 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(invariant: str = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        invariant (str): Color invariant in CIConv layer, or None
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', invariant, 'D', True, pretrained, progress, **kwargs)


def vgg19(invariant: str = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        invariant (str): Color invariant in CIConv layer, or None
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', invariant, 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(invariant: str = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._

    Args:
        invariant (str): Color invariant in CIConv layer, or None
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', invariant, 'E', True, pretrained, progress, **kwargs)
