from sub_model import Block as VitBlock, ViTill, LinearAttention2, bMlp
import warnings
import torch
import torch.nn as nn
import numpy as np
import random
import vit_encoder
from functools import partial

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Dinomaly(nn.Module):
    def __init__(self, weight=None, seed=42, encoder_name='dinov2reg_vit_base_14'):
        super(Dinomaly, self).__init__()
        setup_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Encoder 불러오기
        encoder = vit_encoder.load(encoder_name)
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise ValueError("Architecture must be one of: small, base, large.")
        
        # Bottleneck 생성
        bottleneck = []
        bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
        bottleneck = nn.ModuleList(bottleneck)

        bottleneck2 = []
        bottleneck2.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
        bottleneck2 = nn.ModuleList(bottleneck2)        
        # Decoder 생성: VitBlock들을 ModuleList에 추가
        decoder = []
        for i in range(8):
            blk = VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2
            )
            decoder.append(blk)
        decoder = nn.ModuleList(decoder)

        decoder2 = []
        for i in range(8):
            blk2 = VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2
            )
            decoder2.append(blk2)
        decoder2 = nn.ModuleList(decoder2)

        
        # ViTill 생성 (Dinomaly의 핵심 모델)
        self.model = ViTill(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            bottleneck2=bottleneck2,
            decoder2=decoder2,

            target_layers=target_layers,
            mask_neighbor_size=0,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder
        )
        self.model.to(self.device)
        
        # 학습할 파라미터 (예: bottleneck, decoder)
        self.trainable = nn.ModuleList([bottleneck, decoder])
        self.trainable2 = nn.ModuleList([bottleneck2, decoder2])
        
        # 만약 weight가 주어졌다면, Dinomaly 전체(state_dict)를 로드
        if weight is not None:
            try:
                self.load_state_dict(torch.load(weight), strict=True)
            except Exception as e:
                print("Dinomaly weight load Error:", e)
        
    def forward(self, x):

        return self.model(x)