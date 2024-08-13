import torch 
import torch.nn as nn 
from math import ceil

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    # depth: d = alpha ** phi 
    # width: w = beta ** phi 
    # resolution: r = gamma ** phi 
    # alpha * beta ** 2 * gamma ** 2 = 2
    "b0": (0, 224, 0.2), 
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups, #depthwise convolution: if groups = in_channels, it is depthwise convolution, if groups = 1, it is normal convolution
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1), #C x 1 x 1 -> C // r x 1 x 1
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1), #C // r x 1 x 1 -> C x 1 x 1
            nn.Sigmoid(), # To make the output between 0 and 1, like attention mechanism for each channel
        )

    def forward(self, x):
        return x * self.se(x)
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4, survival_prob=0.8):
        super().__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1 
        self.hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != self.hidden_dim
        reduced_dim = int(in_channels / reduction)
        if self.expand:
            self.expand_conv = CNNBlock(in_channels, self.hidden_dim, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            CNNBlock(self.hidden_dim, self.hidden_dim, kernel_size, stride, groups=self.hidden_dim, padding=padding),
            SqueezeExcitation(self.hidden_dim, reduced_dim),
            nn.Conv2d(self.hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def stochastic_depth(self, x):
        if not self.training:
            return x 
        #randomly remove some layers during training
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor
    
    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super().__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )


    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, _, drop_rate = phi_values[version]
        depth_factor = alpha ** phi 
        width_factor = beta ** phi 
        return width_factor, depth_factor, drop_rate
    
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels 
        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)
            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2, #if kernel_size = 3, padding = 1, if kernel_size = 5, padding = 2 
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )
        return nn.Sequential(*features)
    
    def forward(self, x):
        print("before all: ", x.shape)
        x = self.features(x)
        print("after features: ", x.shape)
        x = self.pool(x)
        print("after pool: ", x.shape)

        res =  self.classifier(x.view(x.shape[0], -1))
        print("after classifier: ", res.shape)
        return res
    

def test():
    version = "b0"
    _, res, _ = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res))
    model = EfficientNet(version, num_classes)
    print(model(x).shape)

test()


