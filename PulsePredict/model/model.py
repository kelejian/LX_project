import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import model.loss as module_loss
from torch_geometric.nn import MLP as PygMLP # 直接用PyG的MLP模块

# ==========================================================================================
# 基础组件定义 (Basic Components)
# ==========================================================================================

class ResMLPBlock(nn.Module):
    """
    残差 MLP 块
    
    用途: 用于深度编码器，增加网络深度的同时防止梯度消失和模型退化
    结构: Linear -> BN -> SiLU -> Dropout -> Linear -> BN -> SiLU -> Dropout + 残差连接
    
    参数:
        hidden_dim: 隐藏层维度
        dropout: Dropout 概率
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        输入: (B, hidden_dim)
        输出: (B, hidden_dim)
        """
        return x + self.block(x)

class SeedFeatureProjector(nn.Module):
    """
    基于 FiLM (Feature-wise Linear Modulation) 机制: '基函数生成 + 动态调制' 的种子投影器
    将高维时序生成任务解耦为两个正交的子空间：时序形态子空间（Temporal Shape Subspace） 和 物理强度子空间（Physical Intensity Subspace）
    结构:
    1. Basis Branch: PosEnc -> Conv -> Temporal Basis (学习波形的时序形态)
    2. Coeff Branch: Z -> MLP -> Scale & Shift (学习波形的物理参数)
    3. Fusion: Basis * (1 + Scale) + Shift
    parameters:
        z_dim: 输入全局特征维度
        output_len: 输出时序长度
        output_channels: 输出通道数
        pos_dim: 位置编码维度;偶数, 最好大于output_len,以满足DFT基的完备性
        proj_channels: 投影隐藏层维度
        dropout: Dropout 概率
    """
    def __init__(self, z_dim, output_len, output_channels, pos_dim=64, proj_channels=256, dropout=0.1):
        super().__init__()
        self.output_len = output_len
        self.pos_dim = pos_dim
        
        # 1. 固定正弦位置编码 (作为时序基底的种子)
        pe = self._generate_sinusoidal_pe(output_len, pos_dim)
        self.register_buffer('pos_embedding', pe) 

        # 2. 时序基生成分支 (Temporal Basis Branch)
        # 仅处理位置信息，卷积核较小，参数量少
        self.basis_net = nn.Sequential(
            nn.Conv1d(pos_dim, proj_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(proj_channels),
            nn.SiLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Conv1d(proj_channels, output_channels, kernel_size=3, padding=1, bias=True) 
            # 输出即为“标准基函数”
        )

        # 3. 动态调制分支 (Modulation Branch)
        # 处理全局特征 z，生成 Scale 和 Shift
        # 输出维度 = output_channels * 2 (一个给Scale，一个给Shift)
        self.modulator = nn.Sequential(
            nn.Linear(z_dim, proj_channels),
            nn.SiLU(inplace=True),
            nn.Linear(proj_channels, output_channels * 2)
        )

        self._init_weights()

    def _generate_sinusoidal_pe(self, length, d_model):
        """生成正弦位置编码"""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        Base = 500.0 # PE 的最低频率必须低于或等于信号的基频，1/Base <= 2*Pi / length -> Base >= length / (2*Pi)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(Base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.transpose(0, 1).unsqueeze(0)

    def _init_weights(self):
        """显式初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        
        # 特殊初始化：将调制分支的最后一层初始化为0
        # 使得初始状态下 Scale=0, Shift=0，输出完全由 Basis 决定，梯度流更稳定
        nn.init.constant_(self.modulator[-1].weight, 0)
        nn.init.constant_(self.modulator[-1].bias, 0)

    def forward(self, z):
        """
        z: (B, z_dim)
        return: (B, output_channels, output_len)
        """
        B = z.shape[0]
        
        # --- A. 生成时序基 (Basis) ---
        # 广播 PE: (1, P, L) -> (B, P, L) P: pos_dim, L: output_len
        pos = self.pos_embedding.expand(B, -1, -1)
        # 通过卷积生成基函数: (B, C, L) C: output_channels, L: output_len
        basis = self.basis_net(pos)
        
        # --- B. 生成调制系数 (Coefficients) ---
        # (B, Z) -> (B, 2*C)
        style = self.modulator(z)
        # 拆分为 Scale 和 Shift: (B, C)
        scale, shift = style.chunk(2, dim=1)
        
        # 调整形状以进行广播: (B, C, 1)
        scale = scale.unsqueeze(2)
        shift = shift.unsqueeze(2)
        
        # --- C. 融合 (Modulation) ---
        # Output = Basis * (1 + Scale) + Shift
        return basis * (1 + scale) + shift

class BiGRUBottleneck(nn.Module):
    """
    双向 GRU 时序瓶颈
    
    用途: 在低分辨率下注入时序演化逻辑，建立特征间的因果依赖关系
    结构: Bi-GRU -> Linear -> LayerNorm -> SiLU
    
    参数:
        input_dim: 输入特征维度
        hidden_dim: GRU 隐藏层维度
        output_dim: 输出特征维度
        gru_layers: GRU 层数
    """
    def __init__(self, input_dim, hidden_dim, output_dim, gru_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            bias=True  # GRU 门控机制需要 bias
        )
        # 双向 GRU 输出维度为 hidden_dim * 2
        self.proj = nn.Linear(hidden_dim * 2, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """
        输入: (B, input_dim, L)
        输出: (B, output_dim, L)
        """
        # 转换为 GRU 输入格式: (B, input_dim, L) -> (B, L, input_dim)
        x = x.permute(0, 2, 1)
        
        # GRU 处理: (B, L, input_dim) -> (B, L, hidden_dim*2)
        gru_out, _ = self.gru(x)
        
        # 投影: (B, L, hidden_dim*2) -> (B, L, output_dim)
        out = self.proj(gru_out)
        
        # Post-Norm 与激活
        out = self.norm(out)
        out = self.act(out)
        
        # 转换回通道优先格式: (B, L, output_dim) -> (B, output_dim, L)
        return out.permute(0, 2, 1)

class PixelShuffle1D(nn.Module):
    """
    一维亚像素卷积上采样（PixelShuffle）
    
    用途: 将通道维度重排为时间维度，实现高效上采样
    原理: (B, C*r, L) -> (B, C, L*r)，其中 r 为上采样倍率
    
    参数:
        upscale_factor: 上采样倍率
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        """
        输入: (B, C*upscale_factor, L)
        输出: (B, C, L*upscale_factor)
        """
        batch_size, channels, steps = x.size()
        r = self.upscale_factor
        
        if channels % r != 0:
            raise ValueError(f"输入通道数 {channels} 必须能被上采样倍率 {r} 整除")
        
        new_channels = channels // r
        
        # 重排: (B, C_out*r, L) -> (B, C_out, r, L) -> (B, C_out, L, r) -> (B, C_out, L*r)
        x = x.view(batch_size, new_channels, r, steps)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(batch_size, new_channels, steps * r)
        return x

def icnr_init(conv_layer, upscale_factor, initializer=nn.init.kaiming_normal_):
    """
    ICNR 初始化 (针对 1D 卷积)
    
    原理: 将权重初始化为一种特殊形态，使得 Conv1d + PixelShuffle 最初的行为等价于 Nearest Neighbor Upsampling。
    
    参数:
        conv_layer: 需要初始化的 nn.Conv1d 层
        upscale_factor: 上采样倍率
        initializer: 基础初始化函数 (默认 Kaiming Normal)
    """
    w = conv_layer.weight.data
    out_channels, in_channels, kernel_size = w.shape
    
    # 1. 计算“种子”权重的输出通道数 (C_out / r)
    if out_channels % upscale_factor != 0:
        raise ValueError("输出通道数必须能被上采样倍率整除")
        
    sub_kernel_out = out_channels // upscale_factor
    
    # 2. 生成低维种子权重 (shape: [C_out/r, C_in, K])
    kernel_shape = (sub_kernel_out, in_channels, kernel_size)
    w_seed = torch.zeros(kernel_shape)
    initializer(w_seed) # 使用标准方法初始化种子
    
    # 3. 沿输出通道维度进行复制扩展 (Repeat Interleave)
    # 效果: [w1, w2] -> [w1, w1, ..., w2, w2, ...] (每个重复 r 次)
    # 结合 PixelShuffle 的重排逻辑 (view -> permute -> flatten)，这确保了
    # 相邻的 r 个子像素拥有相同的权重，从而输出相同的值 (即最近邻插值)。
    w_new = w_seed.repeat_interleave(upscale_factor, dim=0)
    
    # 4. 赋值回权重
    conv_layer.weight.data.copy_(w_new)
    
    # 5. 处理 Bias: 建议初始化为 0，避免初始阶段引入通道间的直流偏差
    if conv_layer.bias is not None:
        nn.init.zeros_(conv_layer.bias)

class AdapLengthAlign1D(nn.Module):
    """
    自适应长度对齐层
    
    用途: 确保序列长度严格对齐到目标长度，处理上采样后的长度偏差
    策略:
        - 头部锚定: 保护 t=0 物理起点，不进行左侧裁剪/填充
        - 尾部填充: 使用 'replicate' 模式复制边界值，保持数值连续性
        - 尾部裁剪: 直接截断多余部分
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, target_length):
        """
        输入: (B, C, curr_length)
        输出: (B, C, target_length)
        """
        curr_length = x.shape[-1]
        diff = target_length - curr_length

        if diff == 0:
            return x
        
        if diff > 0:
            # 尾部填充: 右侧复制填充 diff 个时间步
            return F.pad(x, (0, diff), mode='replicate')
        else:
            # 尾部裁剪: 截断至目标长度
            return x[..., :target_length]

class ContextInjection(nn.Module):
    """
    基于 FiLM (Feature-wise Linear Modulation) 的上下文注入模块
    
    原理: x_out = (1 + scale(z)) * x_in + shift(z)
    优势: 
    1. 显式建模物理参数对波形的乘性调制(缩放)和加性调制(偏移)。
    2. 计算高效，避免了对拼接后的大通道特征图进行卷积。
    """
    def __init__(self, feature_channels, z_dim):
        """
        :param feature_channels: 输入特征图的通道数 (C)
        :param z_dim: 全局特征 z 的维度 (D)
        """
        super().__init__()
        
        # 定义两个投影层，分别用于生成 Scale (gamma) 和 Shift (beta)
        # 初始化为0，使得初始状态下呈现恒等变换 (Identity Mapping)，利于梯度传播
        self.scale_proj = nn.Linear(z_dim, feature_channels)
        self.shift_proj = nn.Linear(z_dim, feature_channels)
        
        self._init_weights()

    def _init_weights(self):
        # 显式将权重和偏置初始化为0
        # 这样初始输出 scale=0, shift=0 -> out = (1+0)*x + 0 = x
        nn.init.constant_(self.scale_proj.weight, 0)
        nn.init.constant_(self.scale_proj.bias, 0)
        nn.init.constant_(self.shift_proj.weight, 0)
        nn.init.constant_(self.shift_proj.bias, 0)

    def forward(self, x, z_prime):
        """
        输入:
            x: (B, C, L) - 局部特征图
            z_prime: (B, D) - 全局工况特征
        输出:
            (B, C, L) - 调制后的特征图
        """
        # 1. 计算调制参数
        # (B, D) -> (B, C) -> (B, C, 1) 以便广播
        scale = self.scale_proj(z_prime).unsqueeze(2)
        shift = self.shift_proj(z_prime).unsqueeze(2)
        
        # 2. 应用 FiLM 调制
        # 采用残差式缩放: (1 + scale) * x + shift
        return x * (1 + scale) + shift

class ResBlock1D(nn.Module):
    """
    一维残差块
    
    用途: 提取局部时序特征，通过残差连接增强梯度流动
    结构: Conv -> BN -> SiLU -> Conv -> BN + 残差连接 -> SiLU
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 通道数不匹配时使用 1x1 卷积调整
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        """
        输入: (B, in_channels, L)
        输出: (B, out_channels, L)
        """
        residual = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.silu(out, inplace=True)

class DeepRegressionHead(nn.Module):
    """
    深层回归头
    
    功能: 将特征空间映射到物理输出空间 (Mean & Variance)
    设计原则: 移除 BN 和 Dropout，保证回归数值的绝对尺度和连续性
    """
    def __init__(self, in_channels, out_channels=1, hidden_dim=64, num_layers=2):
        """
        :param in_channels: 输入特征通道数
        :param out_channels: 输出物理量通道数 (1或2)
        :param hidden_dim: 中间隐藏层通道数
        :param num_layers: 隐藏层(Conv+Act)的数量。
                           0 表示纯线性映射 (Conv1x1); 
                           >=1 表示 num_layers 个中间层 + 1 个输出层。
        """
        super().__init__()
        
        layers = []
        
        # 如果 num_layers > 0，构建非线性中间层
        if num_layers > 0:
            # 第一层: in_channels -> hidden_dim
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=True))
            layers.append(nn.SiLU(inplace=True))
            
            # 后续层: hidden_dim -> hidden_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=True))
                layers.append(nn.SiLU(inplace=True))
            
            # 最后一层: hidden_dim -> out_channels (1x1 Conv)
            layers.append(nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=True))
        else:
            # 如果 num_layers == 0，直接使用 1x1 卷积进行线性投影
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))

        self.net = nn.Sequential(*layers)
        
        # 显式初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                # 最后一层通常不需要激活函数增益，保持较小的初始值有助于回归稳定
                if m == self.net[-1]:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        输入: (B, in_channels, L)
        输出: (B, out_channels, L)
        """
        return self.net(x)


# ==========================================================================================
# 主模型定义 (Hybrid PulseCNN)
# ==========================================================================================
class HybridPulseCNN(BaseModel):
    """
    主模型定义
    架构: Deep Residual MLP -> Bi-GRU Bottleneck -> Trident Progressive Decoder

    参数:
        input_dim: 输入工况特征维度
        output_channels: 输出通道数（xyz 三轴）
        mlp_hidden_dim: MLP 编码器隐藏层维度
        seed_proj_channels: 种子投影器中间通道数
        seed_proj_pe_dim: 种子投影器位置编码维度
        gru_hidden_dim: GRU 隐藏层维度
        gru_layers: GRU 层数
        channel_configs: 各解码阶段的通道配置列表 [Stage1, Stage2, Stage3]
        output_lengths: 各解码阶段的输出长度列表 [L1, L2, L3]
        decoder_blocks_per_stage: [int, int, int], 控制各阶段 ResBlock 的堆叠数量。
        head_config: dict, 控制回归头的结构。
        GauNll_use: 是否使用高斯负对数似然损失（输出均值和方差）
    """
    def __init__(self, input_dim=3, output_channels=3, 
                 mlp_hidden_dim=256, 
                 seed_proj_channels=256,
                 seed_proj_pe_dim=64,
                 gru_hidden_dim=128,
                 gru_layers=1,
                 channel_configs=[128, 64, 32],
                 output_lengths=[37, 75, 150],
                 decoder_blocks_per_stage=[1, 1, 2],
                 head_config={'hidden_dim': 64, 'num_layers': 2}, 
                 GauNll_use=True):
        super().__init__()
        
        self.GauNll_use = GauNll_use
        self.output_lengths = output_lengths
        self.channel_configs = channel_configs
        
        # 确保 decoder_blocks_per_stage 长度正确
        if len(decoder_blocks_per_stage) != 3:
            raise ValueError("decoder_blocks_per_stage must have length 3 (for 3 stages).")
        
        # 自动计算各阶段上采样倍率
        self.upscale_factors = []
        for i in range(len(output_lengths) - 1):
            factor = output_lengths[i+1] / output_lengths[i]
            r = max(1, int(round(factor))) 
            self.upscale_factors.append(r)

        # ========================================================================
        # 1. 深度流形编码器
        # ========================================================================
        self.encoder_input = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim, bias=False),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.SiLU(inplace=True)
        )
        self.encoder_body = nn.Sequential(
            ResMLPBlock(mlp_hidden_dim, dropout=0.1),
            ResMLPBlock(mlp_hidden_dim, dropout=0.1)
        )
        self.z_dim = mlp_hidden_dim

        # ========================================================================
        # 2. 序列初始化与瓶颈
        # ========================================================================
        self.init_len = output_lengths[0]
        self.gru_input_dim = channel_configs[0]
        
        # 高效种子投影器
        self.seed_projector = SeedFeatureProjector(
            z_dim=mlp_hidden_dim,
            output_len=self.init_len,
            output_channels=self.gru_input_dim,
            pos_dim=seed_proj_pe_dim, 
            proj_channels=seed_proj_channels,
            dropout=0.1
        )

        # Bi-GRU 时序瓶颈
        self.bottleneck = BiGRUBottleneck(
            input_dim=self.gru_input_dim,
            hidden_dim=gru_hidden_dim,
            output_dim=self.gru_input_dim,
            gru_layers=gru_layers
        )
        
        # 长度对齐模块
        self.length_align = AdapLengthAlign1D()

        # ========================================================================
        # 3. Stage 1: 共享解码器（低分辨率）
        # ========================================================================
        self.s1_context = ContextInjection(channel_configs[0], self.z_dim)
        
        # 动态堆叠 ResBlocks
        s1_blocks = []
        for _ in range(decoder_blocks_per_stage[0]):
            s1_blocks.append(ResBlock1D(channel_configs[0], channel_configs[0]))
        self.s1_resblocks = nn.Sequential(*s1_blocks)
        
        # 动态构建 Head
        head_out_dim = 2 if GauNll_use else 1
        self.s1_head = DeepRegressionHead(
            channel_configs[0], 
            out_channels=output_channels * head_out_dim,
            hidden_dim=head_config.get('hidden_dim', 64),
            num_layers=head_config.get('num_layers', 2)
        )

        # ========================================================================
        # 4. Stage 2: 三叉分支（中等分辨率）
        # ========================================================================
        self.s2_branches = nn.ModuleDict()
        r1 = self.upscale_factors[0]
        
        for axis in ['x', 'y', 'z']:
            layers = nn.ModuleDict()
            
            # 上采样层
            layers['up_conv'] = nn.Conv1d(channel_configs[0], channel_configs[1] * r1, kernel_size=1, bias=True)
            icnr_init(layers['up_conv'], upscale_factor=r1) # ICNR 初始化
            layers['pixel_shuffle'] = PixelShuffle1D(upscale_factor=r1)
            
            # 平滑卷积层
            layers['smooth_conv'] = nn.Conv1d(channel_configs[1], channel_configs[1], kernel_size=3, padding='same', bias=False)
            layers['smooth_bn'] = nn.BatchNorm1d(channel_configs[1])
            layers['smooth_act'] = nn.SiLU(inplace=True)

            # 精炼模块
            layers['context'] = ContextInjection(channel_configs[1], self.z_dim)
            
            s2_blocks = []
            for _ in range(decoder_blocks_per_stage[1]):
                s2_blocks.append(ResBlock1D(channel_configs[1], channel_configs[1]))
            layers['resblocks'] = nn.Sequential(*s2_blocks) # 注意这里改名为 resblocks
            
            # Head
            layers['head'] = DeepRegressionHead(
                channel_configs[1], 
                out_channels=head_out_dim,
                hidden_dim=head_config.get('hidden_dim', 64),
                num_layers=head_config.get('num_layers', 2)
            )
            self.s2_branches[axis] = layers

        # ========================================================================
        # 5. Stage 3: 独立精炼（高分辨率）
        # ========================================================================
        self.s3_branches = nn.ModuleDict()
        r2 = self.upscale_factors[1]

        for axis in ['x', 'y', 'z']:
            layers = nn.ModuleDict()

            # 上采样层
            layers['up_conv'] = nn.Conv1d(channel_configs[1], channel_configs[2] * r2, kernel_size=1, bias=True)
            icnr_init(layers['up_conv'], upscale_factor=r2) # ICNR 初始化
            layers['pixel_shuffle'] = PixelShuffle1D(upscale_factor=r2)
            
            # 平滑卷积层
            layers['smooth_conv'] = nn.Conv1d(channel_configs[2], channel_configs[2], kernel_size=3, padding='same', bias=False)
            layers['smooth_bn'] = nn.BatchNorm1d(channel_configs[2])
            layers['smooth_act'] = nn.SiLU(inplace=True)

            # 精炼模块
            layers['context'] = ContextInjection(channel_configs[2], self.z_dim)
            
            s3_blocks = []
            for _ in range(decoder_blocks_per_stage[2]):
                s3_blocks.append(ResBlock1D(channel_configs[2], channel_configs[2]))
            layers['resblocks'] = nn.Sequential(*s3_blocks)
            
            # Head
            layers['head'] = DeepRegressionHead(
                channel_configs[2], 
                out_channels=head_out_dim,
                hidden_dim=head_config.get('hidden_dim', 64),
                num_layers=head_config.get('num_layers', 2)
            )
            self.s3_branches[axis] = layers

    def forward(self, x):
        """
        前向传播

        输入: (B, input_dim)
        输出: 
            - 若 GauNll_use=True: [(s1_mean, s1_var), (s2_mean, s2_var), (s3_mean, s3_var)]
            其中每个 tuple 的 mean 和 var 形状均为 (B, output_channels, output_lengths[i])
            
            - 若 GauNll_use=False: [s1_pred, s2_pred, s3_pred]
            其中每个 pred 是张量，形状为 (B, output_channels, output_lengths[i])

        Stage 输出长度:
            s1: output_lengths[0]
            s2: output_lengths[1]
            s3: output_lengths[2]
        """
        B = x.size(0)
        
        # ====================================================================
        # 1. 编码阶段: 提取全局特征
        # ====================================================================
        # (B, input_dim) -> (B, mlp_hidden_dim)
        z_prime = self.encoder_input(x)
        z_prime = self.encoder_body(z_prime)
        
        # ====================================================================
        # 2. 瓶颈阶段: 生成初始序列
        # ====================================================================
        # (B, mlp_hidden_dim) -> (B, channel_configs[0], output_lengths[0])
        seed_seq = self.seed_projector(z_prime) 
        # Bi-GRU 精炼: (B, channel_configs[0], output_lengths[0]) -> (B, channel_configs[0], output_lengths[0])
        f_s1_in = self.bottleneck(seed_seq)

        # ====================================================================
        # 3. Stage 1 解码: 共享低分辨率特征
        # ====================================================================
        # 上下文注入与残差精炼
        f_s1 = self.s1_context(f_s1_in, z_prime)
        f_s1 = self.s1_resblocks(f_s1)
        
        # 输出预测: (B, channel_configs[0], output_lengths[0]) -> (B, output_channels*(1or2), output_lengths[0])
        s1_out = self.s1_head(f_s1)
        s1_tuple = self._process_head_output(s1_out)

        # ====================================================================
        # 4. Stage 2 解码: 三叉戟中分辨率分支
        # ====================================================================
        s2_preds_list = []
        f_s2_feats = {}
        
        for axis in ['x', 'y', 'z']:
            layers = self.s2_branches[axis]
            
            # 上采样: (B, channel_configs[0], output_lengths[0]) -> 
            #         (B, channel_configs[1]*r1, output_lengths[0]) -> 
            #         (B, channel_configs[1], output_lengths[0]*r1)
            feat = layers['up_conv'](f_s1)      
            feat = layers['pixel_shuffle'](feat) 
            
            # 长度对齐: (B, channel_configs[1], ~output_lengths[1]) -> (B, channel_configs[1], output_lengths[1])
            feat = self.length_align(feat, self.output_lengths[1])
            
            # 平滑融合: 卷积扫过对齐边界，消除填充/裁剪伪影
            feat = layers['smooth_act'](layers['smooth_bn'](layers['smooth_conv'](feat)))
            
            # 上下文注入与残差精炼
            feat = layers['context'](feat, z_prime)
            feat = layers['resblocks'](feat)
            
            # 缓存特征用于下一阶段
            f_s2_feats[axis] = feat
            # 输出预测: (B, channel_configs[1], output_lengths[1]) -> (B, (1or2), output_lengths[1])
            s2_preds_list.append(layers['head'](feat))

        # 拼接三轴输出: (B, output_channels*(1or2), output_lengths[1])
        s2_out = torch.cat(s2_preds_list, dim=1)
        s2_tuple = self._process_head_output(s2_out)

        # ====================================================================
        # 5. Stage 3 解码: 独立高分辨率精炼
        # ====================================================================
        s3_preds_list = []
        
        for axis in ['x', 'y', 'z']:
            layers = self.s3_branches[axis]
            prev_feat = f_s2_feats[axis]
            
            # 上采样: (B, channel_configs[1], output_lengths[1]) -> 
            #         (B, channel_configs[2]*r2, output_lengths[1]) -> 
            #         (B, channel_configs[2], output_lengths[1]*r2)
            feat = layers['up_conv'](prev_feat)
            feat = layers['pixel_shuffle'](feat)
            
            # 长度对齐: (B, channel_configs[2], ~output_lengths[2]) -> (B, channel_configs[2], output_lengths[2])
            feat = self.length_align(feat, self.output_lengths[2])
            
            # 平滑融合
            feat = layers['smooth_act'](layers['smooth_bn'](layers['smooth_conv'](feat)))
            
            # 上下文注入与残差精炼
            feat = layers['context'](feat, z_prime)
            feat = layers['resblocks'](feat)
            
            # 输出预测: (B, channel_configs[2], output_lengths[2]) -> (B, (1or2), output_lengths[2])
            s3_preds_list.append(layers['head'](feat))

        # 拼接三轴输出: (B, output_channels*(1or2), output_lengths[2])
        s3_out = torch.cat(s3_preds_list, dim=1)
        s3_tuple = self._process_head_output(s3_out)

        return [s1_tuple, s2_tuple, s3_tuple]

    def _process_head_output(self, raw_out):
        """
        处理回归头输出，分离均值和方差
        
        输入: (B, output_channels*k, L)，k=1（仅均值）或 k=2（均值+方差）
        输出: 
            - GauNll_use=True: (mean, var)，元组，每个形状为 (B, output_channels, L)
            - GauNll_use=False: raw_out，张量，形状为 (B, output_channels, L)
        """
        if self.GauNll_use:
            # 通道布局: [Mx, Vx, My, Vy, Mz, Vz] -> 重塑为 [x:(M,V), y:(M,V), z:(M,V)]
            B, C, L = raw_out.shape
            reshaped = raw_out.view(B, 3, 2, L)
            mean = reshaped[:, :, 0, :]      # (B, output_channels, L)
            log_var = reshaped[:, :, 1, :]   # (B, output_channels, L)
            var = torch.exp(log_var)         # 转换为方差
            return (mean, var)
        else:
            return raw_out
    
    def get_metrics_output(self, model_output):
        """
        提取用于评估指标的模型输出
        
        返回: 最终阶段的均值预测 (B, output_channels, output_lengths[-1])
        """
        return model_output[-1][0] if self.GauNll_use else model_output[-1]
