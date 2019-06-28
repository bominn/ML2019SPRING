import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo


__all__ = ['vgg16_bn','vgg19_bn']
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, output_size=1274, image_size=448):
        super(VGG, self).__init__()
        self.features = features
        self.image_size = image_size

        self.yolo = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 7*7*11),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.yolo(x)
        x = torch.sigmoid(x) 
        x = x.view(-1,7,7,11)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag=True
    for v in cfg:
        s=1
        if (v==64 and first_flag):
            s=2
            first_flag=False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}




def Yolov1_vgg16bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    yolo = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        yolo_state_dict = yolo.state_dict()
        for k in vgg_state_dict.keys():
            #print(k)
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
        yolo.load_state_dict(yolo_state_dict)
    return yolo

def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    yolo = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        yolo_state_dict = yolo.state_dict()
        for k in vgg_state_dict.keys():
            #print(k)
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
        yolo.load_state_dict(yolo_state_dict)
    return yolo


def test():
    import torch
    model = vgg19_bn(pretrained=True)
    img = torch.rand(1,3,448,448)
    print(model)
    model.eval()
    output = model(img)
    print(output.size())

if __name__ == '__main__':
    test()
