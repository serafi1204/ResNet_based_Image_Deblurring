class Bottleneck(nn.Module):
  def __init__(self, c0, c1,c2, st=1):
    super(Bottleneck, self).__init__()
    self.layer =  nn.Sequential(
            nn.Conv2d(in_channels=c0, out_channels=c1, kernel_size=1, padding=0, padding_mode='reflect'),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=3, padding=1, stride=st, padding_mode='reflect'),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, padding=0, padding_mode='reflect'),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU()
        )

  def forward(self, xin):
    return self.layer(xin)+xin

class Bottleneck1x1(Bottleneck):
  def __init__(self, c0, c1, c2):
    super(Bottleneck1x1, self).__init__(c0, c1, c2, 2)
    self.conv = nn.Conv2d(in_channels=c0, out_channels=c2, kernel_size=1, padding=0, stride=2,  padding_mode='reflect')

  def forward(self, xin):
    return self.layer(xin)+self.conv(xin)

class ResBlock(nn.Module):
    def __init__(self, c0, c1, k=5, N=5):
        super(ResBlock, self).__init__()

        # Layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c0, out_channels=int(c1), kernel_size=1, padding=0, padding_mode='reflect'),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=int(c1), out_channels=c1, kernel_size=k, padding=int(k/2), padding_mode='reflect'),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=c1, out_channels=c0, kernel_size=1, padding=0, padding_mode='reflect'),
            nn.BatchNorm2d(c0),
            nn.LeakyReLU()
        )

    def forward(self, xin):
        x = self.conv(xin)
        return x+xin

class ResLayer (nn.Module):
    def __init__(self, c0, c1, k=5, N=4):
        super(ResLayer, self).__init__()

        # Layers
        self.convArray = []
        for i in range(N):
            self.convArray.append(ResBlock(c0, c1, k))
        self.conv = nn.Sequential(*self.convArray)

    def forward(self, xin):
        return self.conv(xin)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.stage0 = nn.Sequential(
          Bottleneck1x1(3, 64, 128),
          Bottleneck(128, 64, 128), Bottleneck(128, 64, 128)
        ) # 128 x 64 x 64
        self.stage1 = nn.Sequential(
          Bottleneck1x1(128, 64, 256),
          Bottleneck(256, 128, 256), Bottleneck(256, 128, 256), Bottleneck(256, 128, 256)
        )
        self.stage2 = nn.Sequential(
          Bottleneck1x1(256, 128, 512),
          Bottleneck(512, 256, 512), Bottleneck(512, 256, 512), Bottleneck(512, 256, 512), Bottleneck(512, 256, 512)
        ) #256 x 16 x 16
        self.stage3 = nn.Sequential(
          Bottleneck1x1(512, 256, 1024),
          Bottleneck(1024, 512, 1024), Bottleneck(1024, 512, 1024), Bottleneck(1024, 512, 1024), Bottleneck(1024, 512, 1024), Bottleneck(1024, 512, 1024), Bottleneck(1024, 512, 1024)
        ) # 512 x 8 x 8
        self.transConv0 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder0 = ResLayer(512, 256) # 256 x 16 x 16
        self.transConv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = ResLayer(256, 128) # 128 x 32 x 32
        self.transConv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = ResLayer(128, 64) # 128 x 64 x 64
        self.transConv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = ResLayer(64, 32)
        self.convOut = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, padding=0)


    def forward(self, xin: Tensor) -> Tensor: # Assume input map [3, 128, 128]
        x64 = self.stage0(xin) # [64, 64, 64]
        x32 = self.stage1(x64) # [128, 32, 32]
        x16 = self.stage2(x32) # [256, 16, 16]
        x8 = self.stage3(x16) # [512, 8, 8]

        # Origin layers at the end
        # x = self.layer4(x) # [2048, 4, 4]
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        # Custom layers
        x = self.transConv0(x8)
        x = self.decoder0(x+x16)
        x = self.transConv1(x)
        x = self.decoder1(x+x32)
        x = self.transConv2(x)
        x = self.decoder2(x+x64)
        x = self.transConv3(x)
        x = self.decoder3(x)
        x = self.convOut(x)

        return x+xin
