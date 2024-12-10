import torch
from torch import nn
from torch.nn import functional as torchF
import math
from utils.misc import (get_norm)


class AttentionBlock(nn.Module):
    """
    This is a self-attention with a residual connection for QKV. Reference: https://arxiv.org/pdf/1706.03762.pdf

    Input:
        x: tensor of shape (B, in_channels, H, W)
        in_channels: number of input channels
        norm (optional): which normalization to use (instance, group, batch, or none). Default: "group_norm"
        num_groups (optional): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (B, in_channels, H, W)
    """
    def __init__(self, in_channels, norm="group_norm", num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        # creating Q, K, V
        self.get_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        # the output of this module
        self.get_output = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # Split into 3 groups [Q, K, V] after convolution on the channel dimension
        q, k, v = torch.split(self.get_qkv(self.norm(x)), self.in_channels, dim=1)

        # FIXME: Change for einsum later!
        # Rearranging things to get bmm to work properly
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        # remember (QxK) -> where to look at
        weights = torch.bmm(q, k) * (c ** (-0.5))
        
        # results in a squared matrix
        if weights.shape != (b, h * w, h * w):
            raise ValueError(f"Weights shape must be {(b, h * w, h * w)} but we've got {weights.shape}")

        attention_map = torch.softmax(weights, dim=-1)
        
        # remember [(QxK) x V] -> what to look at!
        tmp_output = torch.bmm(attention_map, v)
        
        # the result should be the same as input size with weights on each important embedding
        if tmp_output.shape != (b, h * w, c):
            raise ValueError(f"Weights shape must be {(b, h * w, h * w)} but we've got {weights.shape}")

        # rearranging things and reshaping accordingly
        tmp_output = tmp_output.view(b, h, w, c).permute(0, 3, 1, 2)

        # remember we need a residual that is why reshaping and rearranging above was needed
        return self.get_output(tmp_output) + x

class PositionalEmbedding(nn.Module):
    """
    Positional Embedding for a given timestep. For reference you can check the paper 
    section 3.5 from https://arxiv.org/pdf/1706.03762.pdf

    Input:
        batch_steps: is a tensor/vector (b,1) with the batch containing the step for each sample.
        emb_dim: embedding dimensionality
        scale_factor (optional): linear scale to be applied to timesteps. Default: 1.0
    Output:
        emb: tensor of shape (b, dim) containing the embeddings for the different timesteps specified 
                for each sample.

    Why does it work? 
        1. https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        2. https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/
    Implementation ideas:
        https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    """
    def __init__(self, emb_dim, scale_factor=1.0):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError(f"Dimensionality for the timestep embeddings should be even (given: {emb_dim})")
        self.emb_dim = emb_dim
        self.scale_factor = scale_factor
    
    def forward(self, batch_steps):
        device = batch_steps.device
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(batch_steps * self.scale_factor, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downscale(nn.Module):
    """
    Layer in charge of downscaling the feature maps in the encoder module
    Input:
        x: tensor of shape (B, in_channels, H, W)
        in_channels: number of input channels
    Output:
        tensor of shape (B, in_channels, H/2, W/2)
    """
    def __init__(self, in_channels):
        super().__init__()

        # half the resolution as we use stride = 2
        self.downscale = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        if x.shape[2] % 2 == 1:
            raise ValueError("Height of tensor should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("Width of tensor should be even")

        return self.downscale(x)


class Upscale(nn.Module):
    """
    Layer in charge of upscaling the feature maps in the decoder module
    Input:
        x: tensor of shape (B, in_channels, H, W)
        in_channels: number of input channels
    Output:
        tensor of shape (B, in_channels, H/*2, W*2)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x):
        return self.upscale(x)


class ResidualBlock(nn.Module):
    """
    Residual blocks for the encoder/decoder modules. As from the paper it is composed by 2 convolutional blocks.
    Also, if timesteps and/or class embeddings are provided we will consider them in the construction of this
    residual block. For reference check appendix B in https://arxiv.org/pdf/2006.11239.pdf

    Input:
        x: input tensor of shape (B, in_channels, H, W)
        time_embeddings (optional): time embedding tensor of shape (B, time_emb_dim) or None if the block doesn't use conditioning for timestemps
        yclass (optional): classes tensor of shape (B) or None if the block doesn't use conditioning for classes
        in_channels: number of input channels
        out_channels: number of output channels
        time_emb_dim (optional): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        num_classes (optional): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (optional): activation function. Default: RELU
        norm (optional): which normalization to use (instance, group, batch, or none). Default: "group_norm"
        num_groups (optional): number of groups used in group normalization. Default: 32
        use_attention (optional): if True applies AttentionBlock to the output. Default: False
    Output:
        tensor of shape (B, out_channels, H, W)
    """

    def __init__(self, in_channels, out_channels, dropout, time_emb_dim=None, num_classes=None, activation=torchF.relu,
                 norm="group_norm", num_groups=32, use_attention=False):
        super().__init__()
        
        # activation function to be used in the model
        self.activation = activation

        # 1st Convolutional block
        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # 2nd Convolutional block
        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(nn.Dropout(p=dropout), nn.Conv2d(out_channels, out_channels, 3, padding=1))

        # teaches the model where in the diffusion process the samples are.
        if time_emb_dim is not None:
            self.time_bias = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_bias = None

        # teaches the model where the sample came from even when we are close to T-step
        if num_classes is not None:
            self.class_bias = nn.Embedding(num_classes, out_channels)
        else:
            self.class_bias = None

        # make sure the convolutional blocks are compatible when adding residual connection
        if in_channels != out_channels:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_connection = nn.Identity()
        
        # if activated
        if not use_attention :
            self.attention = nn.Identity()
        else:
            self.attention = AttentionBlock(out_channels, norm, num_groups)
    
    def forward(self, x, time_embeddings=None, yclass=None):
        # 1st convolutional block
        residual_block = self.activation(self.norm_1(x))
        residual_block = self.conv_1(residual_block)

        # shifting/bias-towards by the step embeddings
        if self.time_bias is not None:
            if time_embeddings is None:
                raise ValueError("Conditioning on timestep was activated but time_embeddings is None")
            residual_block += self.time_bias(self.activation(time_embeddings))[:, :, None, None]

        # shifting/bias-towards by the class conditional embeddings
        if self.class_bias is not None:
            if yclass is None:
                raise ValueError("Conditioning on class was activated but yclass is None")

            residual_block += self.class_bias(yclass)[:, :, None, None]

        # 2nd convolutional block
        residual_block = self.activation(self.norm_2(residual_block))
        residual_block = self.conv_2(residual_block)
        
        # residual between X and conv2 where X channels are modified if needed
        residual_block += self.residual_connection(x)

        # self-attention
        residual_block = self.attention(residual_block)

        return residual_block


class AutoEncoderUnet(nn.Module):
    """
    Unet-based autoencoder Model to be used to estimate the noise added to the image at different timestemps. For reference please check
    https://arxiv.org/pdf/2006.11239.pdf

    Input:
        x: tensor of shape (B, in_channels, H, W)
        tstep_emb (optional): timestep embedding tensor of shape (B, time_emb_dim) or None if the block doesn't condition on timestemps
        yclass (optional): classes tensor of shape (B) or None if the block doesn't condition on classes
        img_channels: number of image channels
        initial_channels: number of initial channels (output of first convolution)
        channel_mults (optional): tuple of channel multiplers. Default: (1, 2, 4, 8)
        time_emb_dim (optional): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        time_emb_scale (optional): linear scale to be applied to timesteps. Default: 1.0
        num_classes (optional): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (optional): activation function. Default: torch.nn.functional.relu
        dropout (optional): dropout rate at the end of each residual block
        attention_res (optional): list of relative resolutions at which to apply attention. Default: ()
        norm (optional): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (optional): number of groups used in group normalization. Default: 32
        image_pad (optional): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    Output:
        tensor of shape (B, out_channels, H, W)
    """
    def __init__(self, img_channels, initial_channels, channel_mults=(1, 2, 4, 8), num_res_blocks=2, time_emb_dim=None,
                 time_emb_scale=1.0, num_classes=None, activation=torchF.relu, dropout=0.1, attention_res=(),
                 norm="group_norm", num_groups=32, image_pad=0):
        super().__init__()

        self.activation = activation
        self.image_pad = image_pad
        self.num_classes = num_classes

        # projects 3-channels images to initial_channels feature maps
        self.initial_map = nn.Conv2d(img_channels, initial_channels, 3, padding=1)

        if time_emb_dim is not None:
            # Transformers Sinusoidal Positional Embeddings, ref: section 3.5 https://arxiv.org/pdf/1706.03762.pdf
            self.time_embedding_layer = nn.Sequential(
                PositionalEmbedding(initial_channels, time_emb_scale),
                nn.Linear(initial_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_embedding_layer = None

        self.downscalers = nn.ModuleList()
        self.upscalers = nn.ModuleList()

        channels = [initial_channels]
        curr_channels = initial_channels

        # Encoder Layers
        for i, mult in enumerate(channel_mults):
            out_channels = initial_channels * mult

            for _ in range(num_res_blocks):
                # no resolution changed here, only in the channels dimension
                use_att = i in attention_res
                self.downscalers.append(ResidualBlock(in_channels=curr_channels, out_channels=out_channels, dropout=dropout,
                                                      time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation,
                                                      norm=norm, num_groups=num_groups, use_attention=use_att)
                                        )
                curr_channels = out_channels
                # keep track for reversing the process
                channels.append(curr_channels)
            
            if i != len(channel_mults) - 1:
                # downscales as resolution needs to be changed.
                self.downscalers.append(Downscale(in_channels=curr_channels))
                channels.append(curr_channels)

        # embedding in the model (Unet based auto-encoder), here we don't touch the resolution
        # Follow design from PixelCNN++ from https://github.com/openai/pixel-cnn
        self.model_emb = nn.ModuleList(
                [
                    ResidualBlock(in_channels=curr_channels, out_channels=curr_channels, dropout=dropout, time_emb_dim=time_emb_dim,
                                  num_classes=num_classes, activation=activation, norm=norm, num_groups=num_groups, use_attention=True),
                    ResidualBlock(in_channels=curr_channels, out_channels=curr_channels, dropout=dropout, time_emb_dim=time_emb_dim,
                                  num_classes=num_classes, activation=activation, norm=norm, num_groups=num_groups, use_attention=False)
                ]
            )

        # Decoder Layers, here we increase/upsample resolutions
        for i, mult in reversed(list(enumerate(channel_mults))):
            # we go in reverse
            out_channels = initial_channels * mult

            for _ in range(num_res_blocks + 1):
                # as we are concatenating instead of Summation, the idea is from PixelSNAL (https://arxiv.org/pdf/1712.09763.pdf)
                use_att = i in attention_res
                self.upscalers.append(ResidualBlock(in_channels=channels.pop() + curr_channels, out_channels=out_channels, dropout=dropout,
                                              time_emb_dim=time_emb_dim, num_classes=num_classes, activation=activation, norm=norm,
                                              num_groups=num_groups, use_attention=use_att)
                                )
                curr_channels = out_channels
            
            if i != 0:
                # increase resolution back
                self.upscalers.append(Upscale(in_channels=curr_channels))

        if len(channels) != 0:
            raise ValueError("No downscaling was applied - invalid configuration!")
        
        self.out_norm = get_norm(norm, initial_channels, num_groups)

        # to be used for reconstruction of error/noise
        self.out_conv = nn.Conv2d(initial_channels, img_channels, 3, padding=1)

    def forward(self, x, tstep_emb=None, yclass=None):
        # make sure the input resolution is a multiple of 2
        padding = self.image_pad
        if padding != 0:
            x = torchF.pad(x, (padding,) * 4)

        if self.time_embedding_layer is not None:
            if tstep_emb is None:
                raise ValueError("time conditioning was specified but tstep_emb is not passed")
            
            time_embeddings = self.time_embedding_layer(tstep_emb)
        else:
            time_embeddings = None
        
        if self.num_classes is not None and yclass is None:
            raise ValueError("class conditioning was specified but yclass is not passed")
        
        # first convolution to use
        x = self.initial_map(x)
        # from here is where skip connection will originate from
        skips = [x]

        # downsampling
        for layer in self.downscalers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_embeddings, yclass)
            else:
                # likely is just scaling layers they dont need timestep and yclass
                x = layer(x)
            
            skips.append(x)
        
        # gated convolution from PixelCNN++
        for layer in self.model_emb:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_embeddings, yclass)
            else:
                # likely is just scaling layers they dont need timestep and yclass
                x = layer(x)
        
        for layer in self.upscalers:
            if isinstance(layer, ResidualBlock):
                # residual connection between the encoder and decoder, by concatenation not addition.
                # the idea is from PixelSNAIL (https://arxiv.org/pdf/1712.09763.pdf)
                x = torch.cat([x, skips.pop()], dim=1)# channel dimension
                x = layer(x, time_embeddings, yclass)
            else:
                x = layer(x)

        x = self.activation(self.out_norm(x))
        
        # reconstructed perturbated/noisy image
        x = self.out_conv(x)
        
        # remove the padding if it was added.
        if self.image_pad != 0:
            return x[:, :, padding:-padding, padding:-padding]
        else:
            return x

