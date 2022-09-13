import torch
import torch.nn as nn
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ViP_S': _cfg(crop_pct=0.9),
    'ViP_M': _cfg(crop_pct=0.9),
    'ViP_L': _cfg(crop_pct=0.875),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim
        self.codim = dim
        self.mlp_c = nn.Linear(dim, self.codim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, self.codim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, self.codim, bias=qkv_bias)
        self.mlp_z = nn.Linear(dim, self.codim, bias=qkv_bias)
        

        self.reweight = Mlp(self.codim, self.codim // 3, self.codim * 4)
        
        self.proj = nn.Linear(self.codim, dim)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x):
        B, H, W, Z, C = x.shape
      
        S = C // self.segment_dim
        T = self.codim // H

        #print(C, self.segment_dim, H)

        h = x.reshape(B, H, W, Z, self.segment_dim, S).permute(0, 4, 3, 2, 1, 5).reshape(B, self.segment_dim, W, Z, H*S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, Z, H, T).permute(0, 4, 2, 3, 1, 5).reshape(B, H, W, Z, self.codim)

        w = x.reshape(B, H, W, Z,self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H, self.segment_dim, Z, W*S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, Z, W, T).permute(0, 1, 4, 3, 2, 5).reshape(B, H, W, Z, self.codim)

        z = x.reshape(B, H, W, Z,self.segment_dim, S).permute(0, 2, 1, 4, 3, 5).reshape(B, W, H, self.segment_dim, Z*S)
        z = self.mlp_w(z).reshape(B, W, H, self.segment_dim, Z, T).permute(0, 2, 1, 4, 3, 5).reshape(B, H, W, Z, self.codim)

        c = self.mlp_c(x)
        
        a = (h + w + z + c).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, self.codim, 4).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + z * a[2] + c * a[3]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn = WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # B, C, H, W, Z
        print(x)
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv3d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x) # B, C, H, W, Z
        x = x.permute(0, 2, 3, 4, 1)
        return x

def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3., qkv_bias=False, qk_scale=None, \
    attn_drop=0, drop_path_rate=0., skip_lam=1.0, mlp_fn = WeightedPermuteMLP, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,\
            attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn = mlp_fn))
        if kwargs['pos_embedding']=='PEG':
            if block_idx==0:
                blocks.append(PosCNN(in_chans = dim, embed_dim = dim))

    blocks = nn.Sequential(*blocks)

    return blocks

# From https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv3d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        cnn_feat = x.permute(0,4,1,2,3)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        #x = x.flatten(2).transpose(1, 2)
        x = x.permute(0,2,3,4,1)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class VisionPermutator3D(nn.Module):
    """ Vision Permutator
    """
    def __init__(self, layers, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
        embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.LayerNorm, mlp_fn = WeightedPermuteMLP, embed_layer=None, pos_embedding=None,
        checkpoint_path_2d='', device='cuda:0'):

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = embed_layer

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                    qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam,
                    mlp_fn = mlp_fn, pos_embedding = pos_embedding)
            #if pos_embedding=="PEG":
            #    stage.insert(1,PosCNN(in_chans = embed_dims[i], embed_dim = embed_dims[i]))
            network.append(stage)
            
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size))

        self.checkpoint_path_2d = checkpoint_path_2d
        self.device = device
        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

            

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        if self.checkpoint_path_2d!='':
            self.__load_2d_pretrained_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def __load_2d_pretrained_weight(self):
        pretrained_dict = torch.load(self.checkpoint_path_2d, map_location=self.device)
        # load partial dict_file (except for pos_embed and last layer)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,H,W,Z-> B,H,W,Z,C
        x = x.permute(0, 2, 3, 4, 1)
        return x

    def forward_tokens(self,x):
        for idx, block in enumerate(self.network):
            x = block(x)
        B, H, W, Z, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        
        x = self.forward_embeddings(x)
        # B, H, W, Z,  C -> B, N, C
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))


@register_model
def vip3d_s7(pretrained=False, **kwargs):
    layers = [4, 3, 8, 3]
    transitions = [True, False, False, False]
    segment_dim = [8, 4, 4, 4]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [192, 384, 384, 384]
    model = VisionPermutator3D(layers, embed_dims=embed_dims, patch_size=16, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP, **kwargs)
    model.default_cfg = default_cfgs['ViP_S']
    return model

@register_model
def vip3d_s14(pretrained=False, **kwargs):
    layers = [4, 3, 8, 3]
    transitions = [False, False, False, False]
    segment_dim = [8, 8, 8, 8]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [384, 384, 384, 384]
    model = VisionPermutator3D(layers, embed_dims=embed_dims, patch_size=16, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP, **kwargs)
    model.default_cfg = default_cfgs['ViP_S']
    return model


@register_model
def vip3d_m7(pretrained=False, **kwargs):
    # 55534632
    layers = [4, 3, 14, 3]
    transitions = [False, True, False, False]
    segment_dim = [8, 8, 4, 4]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 256, 512, 512]
    model = VisionPermutator3D(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP, **kwargs)
    model.default_cfg = default_cfgs['ViP_M']
    return model


@register_model
def vip3d_l7(pretrained=False, **kwargs):
    layers = [8, 8, 16, 4]
    transitions = [True, False, False, False]
    segment_dim = [8, 4, 4, 4]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 512, 512, 512]
    model = VisionPermutator3D(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP, **kwargs)
    model.default_cfg = default_cfgs['ViP_L']
    return model