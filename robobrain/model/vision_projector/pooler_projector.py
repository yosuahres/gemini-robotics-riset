import torch.nn as nn

class PoolerProjector(nn.Module):
    def __init__(self, config, vision_cfg):
        super().__init__()
        self._config = config
        self.hw = vision_cfg.image_size // vision_cfg.patch_size

        self.conv_pool = nn.Conv2d(config.mm_hidden_size, config.hidden_size, kernel_size=2, stride=2)

        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        height = width = self.hw
        assert height * width == x.shape[1]
        x = x.view(x.shape[0], height, width, -1).permute(0, 3, 1, 2)
        x = self.conv_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": "pooler"}
    

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
