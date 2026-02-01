''' This module includes the occupant injury prediction model. '''

import torch
import torch.nn as nn
# from torch_geometric.nn import MLP as PygMLP # 直接用PyG的MLP模块

class FeatureSELayer(nn.Module):
    """ 
    针对 1D 特征向量的 SE 门控模块。
    通过 Sigmoid 生成门控权重，动态调整融合后各模态特征的贡献度。
    """
    def __init__(self, channels, reduction=8):
        super(FeatureSELayer, self).__init__()
        reduced = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x shape: (batch_size, channels)
        return x * self.fc(x)
    
class BaseMLP(nn.Module):
    """
    自定义 MLP 模块，遵循 Linear -> BN -> SiLU -> Dropout 的现代最佳实践。
    最后一层保持线性 (Plain)，不进行归一化或激活，以保留特征分布。
    """
    def __init__(self, in_features, hidden_features, out_features, num_layers, dropout=0.0):
        super(BaseMLP, self).__init__()
        layers = []
        
        # 输入维度 -> 隐藏维度
        curr_in = in_features
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_in, hidden_features, bias=False)) # BN前不需要Bias
            layers.append(nn.BatchNorm1d(hidden_features))
            layers.append(nn.SiLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_in = hidden_features
            
        # 最后一层: 隐藏维度 -> 输出维度 (Plain Linear)
        layers.append(nn.Linear(curr_in, out_features))
        
        self.mlp = nn.Sequential(*layers)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        """
        用于 TemporalConvNet 中进行堆叠
        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            stride (int): 卷积步幅。
            dropout (float): Dropout 概率。
        """
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) // 2  # 保持输入输出长度一致

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False) # 无偏置配合BN
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.silu1 = nn.SiLU(inplace=True)
        if dropout > 0:
            self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.silu2 = nn.SiLU(inplace=True)

        # 如果输入输出通道数不同,使用 1x1 卷积调整维度
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu1(out)
        if hasattr(self, 'dropout1'):
            out = self.dropout1(out) # Post-Activation Dropout
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.silu2(out)
        return out
class ChannelAttention(nn.Module):
    """通道注意力模块，用于对不同方向的碰撞波形进行自适应加权
        也提供固定权重方案
    """
    def __init__(self, in_channels, fixed_weight=[0.7,0.2,0.1]):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        if fixed_weight is not None:
            assert len(fixed_weight) == in_channels, "fixed_weight长度必须等于in_channels"
            self.fixed_weight = torch.tensor(fixed_weight).view(1, in_channels, 1)  # (1, C, 1)
            self.fixed_weight = nn.Parameter(self.fixed_weight, requires_grad=False)  # 不更新权重
        else:
            self.fixed_weight = None

        # 共享的MLP
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_channels * 2, in_channels, 1, bias=True),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_channels * 2, in_channels, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

        # 用于记录整个epoch的注意力权重
        self.epoch_attention_weights = []

    def forward(self, x):
        # x: (B, C, L)
        if self.fixed_weight is not None:
            attention = self.fixed_weight.to(x.device)  # 使用固定权重
            self.epoch_attention_weights.append(attention.detach().cpu())
            return x * attention  # (B, C, L) * (1, C, 1)
        
        # 自适应计算注意力权重
        avg_out = self.fc1(self.avg_pool(x))  # (B, C, 1)
        max_out = self.fc2(self.max_pool(abs(x)))  # (B, C, 1)
        out = avg_out + max_out
        attention = self.sigmoid(out)  # (B, C, 1)

        # 记录当前batch的注意力权重
        self.epoch_attention_weights.append(attention.detach().cpu())
        
        return x * attention

    def get_epoch_attention_stats(self):
        """获取整个epoch的注意力权重统计信息"""
        if self.epoch_attention_weights:
            all_weights = torch.cat(self.epoch_attention_weights, dim=0)
            mean_weights = all_weights.mean(dim=0).squeeze(-1)  # (C,)
            std_weights = all_weights.std(dim=0).squeeze(-1)   # (C,)
            return mean_weights, std_weights
        return None, None
    
    def reset_epoch_records(self):
        """重置epoch记录，在每个epoch开始时调用"""
        self.epoch_attention_weights = []
    
    def get_epoch_attention_weights(self):
        """获取整个epoch的所有注意力权重"""
        if self.epoch_attention_weights:
            return torch.cat(self.epoch_attention_weights, dim=0).squeeze(-1)  # (Total_samples, C)
        return None

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, tcn_channels_list, Ksize_init=6, Ksize_mid=3, 
                 dropout=0.1, hidden=128, use_channel_attention=True, fixed_channel_weight=None,
                 use_attention_pooling=True):
        """
        损伤预测模型一部分, 负责提取X,Y加速度曲线特征(x_acc), 作为encoder一部分
        Args:
            use_attention_pooling (bool): 是否使用注意力池化替代全局平均池化。
        """
        super(TemporalConvNet, self).__init__()

        self.use_attention_pooling = use_attention_pooling

        # --- 1. 通道注意力 ---
        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            self.channel_attention = ChannelAttention(in_channels, fixed_weight=fixed_channel_weight)

        # --- 2. TCN 模块定义 ---
        kernel_sizes = [Ksize_init] + [Ksize_mid] * (len(tcn_channels_list)-1)

        # 确保参数列表长度一致
        assert len(tcn_channels_list) == len(kernel_sizes), \
            "参数列表长度必须一致:tcn_channels_list, kernel_sizes"
        # 确保kernel_sizes[0]为偶数, 其余为奇数
        assert kernel_sizes[0] % 2 == 0, "kernel_sizes[0]必须为偶数"
        if len(kernel_sizes) > 1:
            assert all([k % 2 == 1 for k in kernel_sizes[1:]]), "kernel_sizes[1:]必须为奇数"

        # 初始卷积层, 并进行一次下采样
        padding_init = (kernel_sizes[0] - 2) // 2  # 保持输入输出长度一致
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, tcn_channels_list[0], kernel_size=kernel_sizes[0], stride=2, padding=padding_init),  # 下采样
            nn.BatchNorm1d(tcn_channels_list[0]),
            nn.SiLU(inplace=True),
        )

        # 堆叠 TemporalBlock
        layers = []
        in_channels = tcn_channels_list[0]
        for i in range(len(tcn_channels_list)-1):
            out_channels = tcn_channels_list[i+1]
            kernel_size = kernel_sizes[i+1]
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
            in_channels = out_channels  # 更新输入通道数

        self.temporal_blocks = nn.Sequential(*layers)

        # --- 3. 池化层 ---
        
        # C_out 即 TCN 的最终输出通道数
        C_out = tcn_channels_list[-1] 

        if self.use_attention_pooling:
            # --- 方案 注意力池化 + 可学习 PE ---
            
            # (a) 定义TCN输出的时间步长度 (L_feat)
            tcn_output_length = 150 // 2
            
            # (b) 可学习的位置编码 (Learned PE)
            self.pos_embedding = nn.Embedding(
                num_embeddings=tcn_output_length, 
                embedding_dim=C_out
            )
            # 注册 position_ids 缓冲区
            self.register_buffer(
                'position_ids', 
                torch.arange(tcn_output_length).expand((1, -1))
            )
            self.pe_dropout = nn.Dropout(dropout) # 添加 Dropout

            # (c) 注意力权重计算网络 (attention_mlp)
            C_hidden_attn = C_out // 2 
            self.attention_mlp = nn.Sequential(
                nn.Conv1d(in_channels=C_out, out_channels=C_hidden_attn, kernel_size=1, bias=False),
                nn.BatchNorm1d(C_hidden_attn),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout), # 添加 Dropout
                nn.Conv1d(in_channels=C_hidden_attn, out_channels=1, kernel_size=1, bias=True)
            )
        else:
            # --- 原始 GAP 方案 ---
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # --- 4. 最终全连接层 ---
        # 无论哪种池化, 输出维度都是 (B, C_out), fc层保持不变
        self.fc = nn.Linear(C_out, hidden)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量,形状为 (B, C, L), C是通道数=3, L是序列长度=150

        Returns:
            torch.Tensor: 输出张量,形状为 (B, hidden)
        """
        # 1. 通道注意力
        if self.use_channel_attention:
            x = self.channel_attention(x)  # (B, C, L)
        
        # 2. 初始卷积
        x = self.initial_conv(x)  # (B, C_0, L/2)
        
        # 3. TCN 堆叠
        x = self.temporal_blocks(x)  # (B, C_out, L_feat), L_feat=75

        # 4. 池化
        if self.use_attention_pooling:
            # --- 方案3: 注意力池化 + 可学习 PE (含 Dropout) ---
            
            # (a) 获取当前特征长度 L_feat (应为 75)
            L_feat = x.size(2)
            
            # (b) 获取位置编码 (B, L_feat, C_out)
            pos_ids = self.position_ids[:, :L_feat].to(x.device)
            pos_embeds = self.pos_embedding(pos_ids)
            
            # *** 应用 PE Dropout ***
            # P_learn_dropout = P_learn * M_pe
            pos_embeds = self.pe_dropout(pos_embeds)
            
            # (c) 转换维度: (B, L_feat, C_out) -> (B, C_out, L_feat)
            pos_embeds = pos_embeds.permute(0, 2, 1)

            # (d) 注入 PE
            # F_pos = F + P_learn_dropout
            x_pos = x + pos_embeds # (B, C_out, L_feat)

            # (e) 计算注意力分数 (B, C_out, L_feat) -> (B, 1, L_feat)
            attention_scores = self.attention_mlp(x_pos)
            
            # (f) 归一化权重 (Softmax)
            # A = Softmax(S)
            attention_weights = torch.softmax(attention_scores, dim=2) 
            
            # (g) 加权求和 (用原始特征 x, 而非 x_pos)
            # F_weighted = F * A
            weighted_features = x * attention_weights
            
            # (h) 压缩维度 -> (B, C_out)
            # v = sum(F_weighted)
            x = torch.sum(weighted_features, dim=2)
        
        else:
            # --- 原始 GAP 方案 ---
            x = self.global_avg_pool(x)  # (B, C_out, 1)
            x = x.squeeze(-1)           # (B, C_out)

        # 5. 全连接层
        x = self.fc(x)  # (B, C_out) -> (B, hidden)
        
        return x
class DiscreteFeatureEmbedding(nn.Module):
    """
    对离散特征进行嵌入处理, 用于损伤预测模型的encoder
    输入: x_att_discrete (离散特征), num_classes_of_discrete (每个离散特征的类别数)
    输出: 嵌入后的特征向量 (concat 所有离散特征的嵌入向量)
    """
    def __init__(self, num_classes_of_discrete):
        """
        参数:
            num_classes_of_discrete (list): 每个离散特征的类别数, 预期输入 [2, 3] 对应 [is_driver_side, OT]。
        """
        super(DiscreteFeatureEmbedding, self).__init__()
        
        # --- 定义每个特征的 Embedding 维度 ---
        # 1. OT(3类)->8维: 赋予足够容量以解耦隐含的物理属性(质量/身高/刚度等), 并保证在MLP输入端的信号强度。
        # 2. is_driver_side(2类)->4维: 提升全局状态变量的表达能力, 且4/8均为2的幂次, 符合GPU内存对齐效率。
        target_dims = [4, 8]
        
        # 为每个离散特征创建嵌入层
        self.embedding_layers = nn.ModuleList()
        self.output_dim = 0 # 记录总输出维度供外部使用
        
        for i, num_classes in enumerate(num_classes_of_discrete):
            # 优先使用预设维度，若越界则回退到 num_classes - 1 (最低为1)
            dim = target_dims[i] if i < len(target_dims) else max(1, num_classes - 1)
            
            self.embedding_layers.append(nn.Embedding(num_classes, dim))
            self.output_dim += dim
        
    def forward(self, x_att_discrete):
        """
        对离散特征进行嵌入并拼接。

        参数:
            x_att_discrete (torch.Tensor): 离散特征张量,形状为 (B, num_discrete_features),B 是 batch size。
        
        返回:
            torch.Tensor: 嵌入后的特征向量,形状为 (B, sum(num_classes_of_discrete) - len(num_classes_of_discrete))。
        """
        embedded_features = []
        
        # 对每个离散特征进行嵌入
        for i, embedding_layer in enumerate(self.embedding_layers):
            # 提取第 i 个离散特征 (B, ) -> (B, num_classes - 1)
            feature = x_att_discrete[:, i]
            embedded_feature = embedding_layer(feature)
            embedded_features.append(embedded_feature)
        
        # 拼接所有嵌入后的特征 (B, sum(num_classes_of_discrete) - len(num_classes_of_discrete))
        output = torch.cat(embedded_features, dim=1)
        return output

class InjuryPredictModel(nn.Module):
    def __init__(self, num_classes_of_discrete, 
                 Ksize_init=8, Ksize_mid=3,
                 num_blocks_of_tcn=3,
                 tcn_channels_list=None, 
                 tcn_output_dim=128,
                 mlp_encoder_output_dim=128, 
                 mlp_decoder_output_dim=96,
                 mlpE_hidden=192, mlpD_hidden=160, 
                 num_layers_of_mlpE=3, num_layers_of_mlpD=3, 
                 dropout_MLP=0.1, dropout_TCN=0, 
                 use_channel_attention=True, fixed_channel_weight=None):
        """
        损伤预测模型初始化。

        参数:
            num_classes_of_discrete (list): 每个离散特征的类别数。
            Ksize_init (int): TCN 初始卷积核大小。
            Ksize_mid (int): TCN 中间卷积核大小。
            num_blocks_of_tcn (int): TCN 编码器的块数。
            tcn_channels_list (list or None): TCN 每个块的输出通道数列表。如果为 None,则根据 num_blocks_of_tcn 自动设置。
            tcn_output_dim (int): TCN 编码器的输出特征维度。
            mlp_encoder_output_dim (int): MLP 编码器的输出特征维度。
            mlp_decoder_output_dim (int): MLP 解码器的输出特征维度。
            num_layers_of_mlpE (int): MLP 编码器的层数。
            num_layers_of_mlpD (int): MLP 解码器的层数。
            mlpE_hidden (int): MLP 编码器的隐藏层维度。
            mlpD_hidden (int): MLP 解码器的隐藏层维度。
            dropout_MLP (float): MLP模块的Dropout 概率。
            dropout_TCN (float): TCN模块的Dropout 概率。
            use_channel_attention (bool): 是否使用通道注意力机制。
        """
        super(InjuryPredictModel, self).__init__()

        # 1. 离散特征嵌入层
        self.discrete_embedding = DiscreteFeatureEmbedding(num_classes_of_discrete)

        # TCN 编码器，处理 x_acc，现在支持通道注意力
        if tcn_channels_list is None:
            if num_blocks_of_tcn < 2:
                raise ValueError("num_blocks_of_tcn 必须大于等于 2")
            elif num_blocks_of_tcn >=2 and num_blocks_of_tcn <= 4:
                tcn_channels_list = [64, 128] + [256] * (num_blocks_of_tcn - 2)
            elif num_blocks_of_tcn >= 5:
                tcn_channels_list = [64, 128] + [256] * (num_blocks_of_tcn - 3) + [512]
        # else:
        #     if len(tcn_channels_list) != num_blocks_of_tcn:
        #         raise ValueError("tcn_channels_list 长度必须等于 num_blocks_of_tcn")
            
        #########################################
        # 2. TCN 波形编码器配置
        if fixed_channel_weight is not None and len(fixed_channel_weight) > 2: # 如果传入了针对 3 通道的 fixed_channel_weight，进行切片适配
            print(f"Warning: fixed_channel_weight length {len(fixed_channel_weight)} > 2. Truncating to 2 for X, Y channels.")
            fixed_channel_weight = fixed_channel_weight[:2]

        self.tcn = TemporalConvNet(
            in_channels=2, # 仅 X, Y 加速度
            tcn_channels_list=tcn_channels_list, 
            Ksize_init=Ksize_init, 
            Ksize_mid=Ksize_mid, 
            hidden=tcn_output_dim,
            dropout=dropout_TCN,
            use_channel_attention=use_channel_attention,
            fixed_channel_weight=fixed_channel_weight
        ) 
        #########################################
        
        #########################################
        # MLP 编码器，处理连续特征和离散特征的嵌入
        # 连续特征: 11个 (impact_velocity, impact_angle, overlap, LL1, LL2, BTF, LLATTF, AFT, SP, SH, RA)
        # 离散特征: 维度由 DiscreteFeatureEmbedding.output_dim 提供 (4 + 8 = 12)
        num_continuous_features = 11 # 20260123 增加了 座椅高度 特征
        mlp_encoder_input_dim = num_continuous_features + self.discrete_embedding.output_dim
        # 3. MLP 编码器定义
        self.mlp_encoder = BaseMLP(
            in_features=mlp_encoder_input_dim,
            hidden_features=mlpE_hidden,
            out_features=mlp_encoder_output_dim,
            num_layers=num_layers_of_mlpE,
            dropout=dropout_MLP
        )
        ###################################

        # 4. 高级特征融合层 (Fusion Layer)
        # 融合向量 = [TCN特征, MLP编码特征, 原始标量特征(Skip Connection)]
        fusion_dim = tcn_output_dim + mlp_encoder_output_dim + mlp_encoder_input_dim
        
        self.fusion_norm = nn.LayerNorm(fusion_dim) 
        self.fusion_se = FeatureSELayer(fusion_dim, reduction=8)

        # 5. MLP 解码器，解码出最终特征
        self.mlp_decoder = BaseMLP(
            in_features=fusion_dim,
            hidden_features=mlpD_hidden,
            out_features=mlp_decoder_output_dim,
            num_layers=num_layers_of_mlpD,
            dropout=dropout_MLP
        )

        # 6. 解码后共享激活层 (Post-Decoder Block)
        # BaseMLP 输出是线性的，在分叉到 Heads 之前，统一做一次 BN+Activation
        # 这作为"共享表示层"，提取对所有损伤任务通用的非线性特征
        self.post_decoder_block = nn.Sequential(
            nn.BatchNorm1d(mlp_decoder_output_dim),
            nn.SiLU(inplace=True)
        )

        # 7. 独立预测头 (Prediction Heads)
        # 输入已经是 Activation 后的特征，所以 Head 的第一个 Linear 是有效的变换
        # 结构: Linear -> BN -> SiLU -> Linear(Plain)
        self.HIC_head = nn.Sequential(
            nn.Linear(mlp_decoder_output_dim, mlp_decoder_output_dim // 2, bias=False),
            nn.BatchNorm1d(mlp_decoder_output_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_decoder_output_dim // 2, 1)
        )
        self.Dmax_head = nn.Sequential(
            nn.Linear(mlp_decoder_output_dim, mlp_decoder_output_dim // 2, bias=False),
            nn.BatchNorm1d(mlp_decoder_output_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_decoder_output_dim // 2, 1)
        )
        self.Nij_head = nn.Sequential(
            nn.Linear(mlp_decoder_output_dim, mlp_decoder_output_dim // 2, bias=False),
            nn.BatchNorm1d(mlp_decoder_output_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_decoder_output_dim // 2, 1)
        )

    def forward(self, x_acc, x_att_continuous, x_att_discrete):
        """
        参数:
            x_acc (torch.Tensor): 碰撞波形数据，形状为 (B, 2, 150)。
            x_att_continuous (torch.Tensor): 连续特征，形状为 (B, 11)。
            x_att_discrete (torch.Tensor): 离散特征，形状为 (B, 2)。

        返回:
            predictions: 预测的 HIC, Dmax, Nij 值，形状为 (B, 3)。
            encoder_output: 编码器的输出，形状为 (B, tcn_output_dim + mlp_encoder_output_dim)。
            decoder_output: 解码器的输出，形状为 (B, mlp_decoder_output_dim)。
        """
        # 1. 特征编码
        x_discrete_embedded = self.discrete_embedding(x_att_discrete) # (B, discrete_emb_dim)
        x_features = torch.cat([x_att_continuous, x_discrete_embedded], dim=1) # (B, raw_dim)
        
        x_mlp_encoded = self.mlp_encoder(x_features) # (B, mlp_encoder_output_dim)
        x_tcn_encoded = self.tcn(x_acc)              # (B, tcn_output_dim)
        # 2. 特征融合 (Encoder Out + Skip Connection)
        # 拼接: [TCN特征, MLP编码特征, 原始输入特征] , 归一化 + SE 注意力重加权
        fusion_vec = torch.cat([x_tcn_encoded, x_mlp_encoded, x_features], dim=1) # (B, fusion_dim)

        fusion_vec = self.fusion_norm(fusion_vec) # 层归一化
        fusion_vec = self.fusion_se(fusion_vec) # 动态调整各部分特征的权重，输出 (B, fusion_dim)

        # 3. 解码
        decoder_output_linear = self.mlp_decoder(fusion_vec) # (B, dec_dim), Linear output

        # 4. 共享非线性变换
        shared_features = self.post_decoder_block(decoder_output_linear) # (B, dec_dim), Activated output

        # 5. 多任务预测
        HIC_pred = self.HIC_head(shared_features) # (B, 1)
        Dmax_pred = self.Dmax_head(shared_features) # (B, 1)
        Nij_pred = self.Nij_head(shared_features) # (B, 1)
        
        predictions = torch.cat([HIC_pred, Dmax_pred, Nij_pred], dim=1) # (B, 3)

        # 返回 encoder_output (用于潜在的蒸馏或分析), 这里定义为 TCN+MLP 的拼接
        encoder_output = torch.cat([x_tcn_encoded, x_mlp_encoded], dim=1) # (B, tcn_output_dim + mlp_encoder_output_dim)

        return predictions, encoder_output, decoder_output_linear