import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights # 引入新版权重
from torchvision.ops import roi_align

class ViewEncoder(nn.Module):
    """
    视图编码器: ResNet18 -> 上采样到 128x128 (Stride=4)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # 使用 ResNet18 作为骨干 (修复警告写法)
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        resnet = models.resnet18(weights=weights)
        
        # 去掉最后两层 (GlobalAvgPool 和 FC)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 上采样层: 512(16x16) -> 64(128x128)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))

    def forward(self, x):
        x = self.backbone(x) # [B, 512, 16, 16]
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)      # [B, 64, 128, 128]
        return x

class SparseCrossAttention(nn.Module):
    """
    【修改版 - 软引导注意力】
    不再使用强制的 Geometric Mask 屏蔽非目标区域。
    允许模型查看全图，但通过 Loss 中的高斯热力图监督，
    训练模型"自觉"地关注 Y 轴对齐区域，同时保留对全局上下文的感知能力。
    """
    def __init__(self, d_model, img_size=(512, 512), feat_size=(128, 128)):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model ** 0.5
        
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        
        # 这里的 scale_y 仅保留用于未来可能的扩展
        self.scale_y = feat_size[0] / img_size[0]

    def forward(self, q_feat, kv_feat, rois=None):
        """
        q_feat: [B, N, d] - Query (圆的特征)
        kv_feat: [B, C, H, W] - Key/Value (左视图特征图)
        rois: [B, N, 3] - 保留接口兼容性，不再用于生成硬掩码
        """
        B, C, H, W = kv_feat.shape
        L = H * W
        kv_flat = kv_feat.view(B, C, L).permute(0, 2, 1) # [B, L, C]

        Q = self.to_q(q_feat)
        K = self.to_k(kv_flat)
        V = self.to_v(kv_flat)

        # 1. 计算原始注意力分数: (B, N, d) @ (B, d, L) -> (B, N, L)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        # === 核心修改 ===
        # 移除: scores = scores + geo_mask (不再强制屏蔽)
        # 模型必须自己学会看哪里，由 Loss 里的高斯真值进行引导
        
        attn_weights = F.softmax(scores, dim=-1) # [B, N, L] 全局 Softmax
        z = torch.bmm(attn_weights, V)
        
        return z, attn_weights

class ViewConsistencyNet(nn.Module):
    def __init__(self, d_model=256, roi_size=3):
        super().__init__()
        self.encoder = ViewEncoder()
        
        # 将 Encoder 输出 (64ch) 映射到 d_model
        self.feature_proj = nn.Conv2d(64, d_model, kernel_size=1)
        
        # === 全局错误分支 ===
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 输入: Front特征(256) + Left特征(256) = 512
        self.head_global_error = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2) # 0:正常, 1:多余线条
        )
        
        # === 局部圆 Query 生成 ===
        self.roi_size = roi_size
        feat_dim = d_model * roi_size * roi_size
        self.query_mlp = nn.Sequential(
            nn.Linear(feat_dim + 3, d_model), # feat + (x,y,r)
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # === 软引导 Attention ===
        self.cross_attn = SparseCrossAttention(d_model)

        # === 局部判定头 ===
        self.head_common = nn.Sequential(nn.Linear(2 * d_model, 256), nn.ReLU())
        self.head_has_proj = nn.Linear(256, 2)
        self.head_proj_ok = nn.Linear(256, 2)

    def make_queries(self, f_front, rois, img_size):
        B, C, H, W = f_front.shape
        H_orig, W_orig = img_size
        batch_queries = []
        scale = H / H_orig
        
        for b in range(B):
            if len(rois[b]) == 0:
                batch_queries.append(torch.zeros(0, C, device=f_front.device))
                continue
            
            # 1. 几何特征归一化
            g_i = rois[b].clone()
            g_i[:, 0] /= W_orig; g_i[:, 1] /= H_orig; g_i[:, 2] /= W_orig 

            # 2. ROI Align 提取视觉特征
            cx, cy, r = rois[b][:, 0], rois[b][:, 1], rois[b][:, 2]
            boxes = torch.stack([cx - r, cy - r, cx + r, cy + r], dim=1)
            
            f_i = roi_align(f_front[b].unsqueeze(0), [boxes], 
                            output_size=(self.roi_size, self.roi_size), 
                            spatial_scale=scale)
            f_i_flat = f_i.view(f_i.size(0), -1)
            
            combined = torch.cat([f_i_flat, g_i], dim=1)
            batch_queries.append(self.query_mlp(combined))
            
        return batch_queries

    def forward(self, img_front, img_left, rois):
        img_size = img_front.shape[-2:]
        
        # 1. 提取特征
        f_front = self.feature_proj(self.encoder(img_front))
        f_left = self.feature_proj(self.encoder(img_left))
        
        # 2. 计算全局错误 (Global Error)
        g_front = self.global_pool(f_front).flatten(1)
        g_left = self.global_pool(f_left).flatten(1)
        logits_global = self.head_global_error(torch.cat([g_front, g_left], dim=1))
        
        # 3. 生成局部圆 Query
        Q_list = self.make_queries(f_front, rois, img_size)
        
        outputs_has, outputs_ok, attn_maps = [], [], []
        
        # 4. 逐样本处理 Attention
        for b in range(len(rois)):
            Q = Q_list[b].unsqueeze(0) # [1, N, d]
            if Q.size(1) == 0:
                outputs_has.append(torch.zeros(0, 2).to(Q.device))
                outputs_ok.append(torch.zeros(0, 2).to(Q.device))
                attn_maps.append(torch.zeros(0, f_left.shape[2]*f_left.shape[3]).to(Q.device))
                continue
            
            # 软引导 Attention
            z_i, attn_w = self.cross_attn(Q, f_left[b:b+1], rois[b].unsqueeze(0))
            
            # Heads
            combined = torch.cat([Q, z_i], dim=-1)
            feat = self.head_common(combined)
            
            outputs_has.append(self.head_has_proj(feat).squeeze(0))
            outputs_ok.append(self.head_proj_ok(feat).squeeze(0))
            attn_maps.append(attn_w.squeeze(0)) # [N, L]
            
        return outputs_has, outputs_ok, logits_global, attn_maps
