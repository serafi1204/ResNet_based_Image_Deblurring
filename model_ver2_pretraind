from torchvision.models.resnet import _ovewrite_named_param, ResNet, Bottleneck
from torchvision.models import *

class DecoderLayer(nn.Module):
    def __init__(self, c0, c1, c2=0, c3=0, k0=7, k1=1, reluEable=True):
        super(DecoderLayer, self).__init__()
        self.reluEable = reluEable
        c2 = c2 if (c2!=0) else int(c1/4)
        c3 = c3 if (c3!=0) else c0
        # Layers
        self.trans = nn.ConvTranspose2d(in_channels=c0, out_channels=c1, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=k0, padding=int(k0/2), padding_mode='reflect'),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=k1, padding=int(k1/2), padding_mode='reflect'),
            nn.BatchNorm2d(c3)
        )
        self.relu = nn.ReLU()

    def forward(self, x, skipX=None):
        x0 = self.trans(x)
        x = x0 if skipX is None else x0+skipX #torch.cat((x0, skipX), dim=1)
        x = self.conv(x)
        if (self.reluEable):
            x += x0
            x = self.relu(x)

        return x


class CNN(ResNet):
    def __init__(self, weights, progress=True, **kwargs):
        # For override with pre-traind paramters
        weight = ResNet50_Weights.verify(ResNet50_Weights.DEFAULT)
        if weights is not None:  _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        super(CNN, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)
        if weights is not None: self.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

        # Custom layers:
        self.decoder0 = DecoderLayer(1024, 512, 256, 512)
        self.decoder1 = DecoderLayer(512, 256, 128, 256)
        self.decoder2 = DecoderLayer(256, 64, 32, 64)
        self.decoder3 = DecoderLayer(64, 32, 16, 3, reluEable=False)



    def forward(self, xin: Tensor) -> Tensor: # Assume input map [3, 128, 128]
        x = self.conv1(xin)
        x = self.bn1(x)
        x64 = self.relu(x) # [64, 64, 64]
        x = self.maxpool(x)

        x32 = self.layer1(x) # [256, 32, 32]
        x16 = self.layer2(x32) # [512, 16, 16]
        x = self.layer3(x16) # [1024, 8, 8]

        # Origin layers at the end
        # x = self.layer4(x) # [2048, 4, 4]
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        # Custom layers
        x = self.decoder0(x, x16)
        x = self.decoder1(x, x32)
        x = self.decoder2(x, x64)
        x = self.decoder3(x)

        return x + xin
