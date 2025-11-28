#import torch.nn.functional as F
import timeit
import torch
from typing import Iterable
from torch import nn
from torch import Tensor
import numpy as np
from typing import List
from einops import rearrange, einsum, reduce, repeat
import math
from jaxtyping import Bool, Float, Int

class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int, #`sequence_length` is at most `context_length`.
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = context_length
        self.d_ff = d_ff
        self.num_head = num_heads
        self.theta = rope_theta
        self.num_layers = num_layers
        self.d_model = d_model

        self.token_embeddings = Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,d_model,d_model,d_in=d_model,max_seq_len=context_length,theta=rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings.forward(in_indices)
        for i in range(self.num_layers):
            x = self.layers[i].forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        # x = softmax(x, -1) # 注意在adapter中，要求的是没有正则化的next-word概率输出
        return x

class TransformerBlock(nn.Module):
    def __init__(self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        d_k:int, 
        d_v:int,
        d_in:int,
        max_seq_len: int, 
        theta: float
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_kh = d_model // num_heads
        self.d_vh = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.d_in = d_in

        self.attn = nn.Module()
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_k=d_k,
            d_v=d_v,
            d_in=d_in,
            with_rope=True,
            max_seq_len=max_seq_len,
            theta=theta
        )
        self.ln1 = nn.Module()
        self.ln1 = RMSNorm(d_model=d_model)

        self.ln2 = nn.Module()
        self.ln2 = RMSNorm(d_model=d_model)

        self.ffn = nn.Module()  # 外层子模块：ffn
        self.ffn = SwiGLU(d_model=d_model,d_ff=d_ff)


    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"]
    )-> Float[Tensor, " batch sequence_length d_model"]:
        x1 = self.ln1.forward(in_features)
        x2 = self.attn.forward(x1)
        x3 = in_features + x2
        x4 = self.ln2.forward(x3)
        x5 = self.ffn.forward(x4)
        out = x3 + x5
        return out

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        d_in: int,
        with_rope: bool = False,
        max_seq_len: int = -1,
        theta: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_kh = d_model // num_heads  # 单头 QK 维度
        self.d_vh = d_model // num_heads  # 单头 V 维度

        # 线性投影：注意 Linear 的定义是 (in_features, out_features)，
        # 你这里的 Linear 应该是一个自定义类，其 weight 形状是 [out, in]
        self.q_proj = Linear(d_k, d_in)  # [d_k, d_in]
        self.k_proj = Linear(d_k, d_in)
        self.v_proj = Linear(d_v, d_in)
        self.output_proj = Linear(d_model, d_v)

        # 显式初始化（可以按需要换初始化方式）
        self.q_proj.weight = nn.Parameter(torch.randn(d_k, d_in))
        self.k_proj.weight = nn.Parameter(torch.randn(d_k, d_in))
        self.v_proj.weight = nn.Parameter(torch.randn(d_v, d_in))
        self.output_proj.weight = nn.Parameter(torch.randn(d_model, d_v))

        self.theta = theta
        self.max_seq_len = max_seq_len
        self.with_rope = with_rope

        if self.with_rope:
            # RoPE 作用在单头维度 d_kh 上
            self.rope = RotaryPositionalEmbedding(
                theta=self.theta,
                d_k=self.d_kh,
                max_seq_len=self.max_seq_len,
            )

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        """
        in_features: (..., seq_len, d_in)
        token_positions: (..., seq_len) 或 None
        """
        device = in_features.device
        seq_len = in_features.size(-2)
        if token_positions is not None:
            assert seq_len == token_positions.size(-1)

        d_in = in_features.size(-1)
        # 你可以加一个断言，确保配置合理：
        # assert d_in == self.d_model, "期望 d_in == d_model"

        # 计算 Q, K, V
        Q = einsum(
            self.q_proj.weight,
            in_features,
            "d_k d_in, ... seq_len d_in -> ... seq_len d_k",
        )
        Q = rearrange(Q, "... seq_len (h d_kh) -> ... h seq_len d_kh", h=self.num_heads)

        K = einsum(
            self.k_proj.weight,
            in_features,
            "d_k d_in, ... seq_len d_in -> ... seq_len d_k",
        )
        K = rearrange(K, "... seq_len (h d_kh) -> ... h seq_len d_kh", h=self.num_heads)

        V = einsum(
            self.v_proj.weight,
            in_features,
            "d_v d_in, ... seq_len d_in -> ... seq_len d_v",
        )
        V = rearrange(V, "... seq_len (h d_vh) -> ... h seq_len d_vh", h=self.num_heads)

        # 提取 batch-like 维度（如 batch, num_heads, 以及可能的前置维度）
        *batch_dims, _, _ = Q.shape  # (..., h, seq_len, d_kh)
        batchlike_str_dict = {f"batchlike{i}": dim for i, dim in enumerate(batch_dims)}
        batchlike_str = " ".join([f"batchlike{i}" for i in range(len(batch_dims))])

        # 构造 token_positions：统一在 in_features.device 上
        if token_positions is None:
            token_pos_1D = torch.arange(seq_len, device=device)  # (seq_len,)
            token_positions = repeat(
                token_pos_1D,
                "seq_len -> " + batchlike_str + " seq_len",
                **batchlike_str_dict,
            )
        else:
            # 迁移到正确的 device + dtype
            if token_positions.dtype != torch.long:
                token_positions = token_positions.long()
            token_positions = token_positions.to(device)

        # RoPE（如果启用）
        if self.with_rope:
            rope_Q = self.rope(Q, token_positions)
            rope_K = self.rope(K, token_positions)
        else:
            rope_Q, rope_K = Q, K

        # 生成下三角掩码：True=允许注意力，False=被屏蔽
        base_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=0,
        )  # [seq, seq]

        mask = repeat(
            base_mask,
            "q k -> " + batchlike_str + " q k",
            **batchlike_str_dict,
        )  # (..., h, seq, seq)

        # 调用 scaled dot-product attention
        # 注意：这里假设 scaled_dot_product_attention 返回的是输出 value，
        # 如果你的实现返回 (out, attn_weights)，这里要 unpack 一下。
        attn_out = scaled_dot_product_attention(
            rope_Q, rope_K, V, mask=mask
        )  # (..., h, seq, d_vh)

        # 如果你的 scaled_dot_product_attention 实际写的是：
        #     return out, attn
        # 那这里要改为：
        # attn_out, attn_weights = scaled_dot_product_attention(...)

        # 合并 heads
        attention_merged = rearrange(
            attn_out, "... h seq d_vh -> ... seq (h d_vh)"
        )  # (..., seq, d_v)

        out = einsum(
            self.output_proj.weight,
            attention_merged,
            "d_model d_v, ... d_v -> ... d_model",
        )
        return out

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean = 0, std = std, a = -3 * std, b = 3 * std)

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, # i.e., vocab_size
        embedding_dim: int, # i.e., dmodel
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None 
    ):
        '''num_embeddings: Size of the vocabulary 
        embedding_dim: Dimension of the embedding vectors,   
        device: Device to store the parameters on  
        dtype: Data type of the parameters  '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        std = 1
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean = 0, std = std, a = -3, b = 3)

    
    def forward(
        self, 
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        '''Lookup the embedding vectors for the given token IDs.'''
        # return type (batch_size, sequence_length)
        # 原始 token_ids 的形状是 (B, T)（B=batch_size，T=sequence_length），替换后每个位置从标量（ID）变成了向量（d_model 维），因此整体形状扩展为 (B, T, d_model)。
        # 输入 token_ids 是形状为 (batch_size, sequence_length) 的二维张量，例如 (32, 10)（32 个样本，每个样本 10 个 token）。
        # 当执行 self.embedding_matrix[token_ids] 时，PyTorch 会按以下逻辑处理：
        # 对 token_ids 中的每个元素（即每个 token ID），从嵌入矩阵中取出对应的行向量（形状 (d_model,)）。
        # 保持 token_ids 自身的维度结构不变，仅将每个元素替换为对应的嵌入向量。
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, # Hidden dimension of the model
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.'''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # performing RMSNorm ... 
        mean_square = reduce(x ** 2, 'batch_size sequence_length d_model -> batch_size sequence_length 1', 'mean' ) # 表示用求平均的方式压缩d_model这一维, 不能直接消去这一维，否则下一步不匹配
        rms = torch.sqrt(mean_square + self.eps) # Python 标准库的 math.sqrt 是为单个数值设计的，只能接收一个标量（如 3.14），无法处理 PyTorch 张量（即使是单元素张量）。所以这里不能使用math.sqrt; PyTorch 的 torch.sqrt 是为张量设计的，支持对张量中的每个元素逐元素计算平方根，且保留张量的形状。
        result = x * self.weight / rms
        # Return the result in the original dtype 
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int,

        d_ff: int | None = None,  # 允许显式传入d_ff，测试时使用
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        # 若未显式传入d_ff，则自动计算：(8/3)*d_model 并向上取整为64的倍数
        if d_ff is None:
            d_ff_candidate = (8 / 3) * d_model
            # 向上取整到最近的64的倍数
            d_ff = ((math.ceil(d_ff_candidate / 64)) * 64)
        self.d_ff = d_ff

        self.w1 = Linear(in_features=d_model, out_features=d_ff) # 根据linear类的定义，应该是dout在前
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)
        self.w1.weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2.weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3.weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w1.weight, mean=0, std=1, a=-3, b=3)
        nn.init.trunc_normal_(self.w2.weight, mean=0, std=1, a=-3, b=3)
        nn.init.trunc_normal_(self.w3.weight, mean=0, std=1, a=-3, b=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1x = einsum(x, self.w1.weight.T,'... d_model, d_model d_ff-> ... d_ff')
        SiLU_of_W1x = W1x * torch.sigmoid(W1x)
        W3x = einsum(x, self.w3.weight.T,'... d_model, d_model d_ff-> ... d_ff')
        SiLU_of_W1x_times_W3x = SiLU_of_W1x * W3x
        res = einsum(SiLU_of_W1x_times_W3x, self.w2.weight.T, '... d_ff, d_ff d_model -> ... d_model')
        return res

# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(
#         self, 
#         theta: float,
#         d_k: int,  # query/key 的维度
#         max_seq_len: int,  # 最大序列长度
#         device=None
#     ):
#         super().__init__()
#         self.theta = theta
#         self.d_k = d_k
#         self.max_seq_len = max_seq_len
#         self.device = device or torch.device('cpu')

#         # 确保 d_k 是偶数（RoPE 要求特征按对处理）
#         d = d_k if d_k % 2 == 0 else d_k + 1  
#         self.d = d

#         # 步骤1：计算分母（θ^(2k-2)/d）
#         base = torch.full((1, d // 2), theta, device=self.device)
#         exp = torch.linspace(0, d - 2, d // 2, device=self.device) / d
#         denominator = torch.pow(base, exp)

#         # 步骤2：计算分子（位置 i）
#         numerator = torch.arange(0, max_seq_len, device=self.device)

#         # 步骤3：计算 θ_ik = i / 分母 → 形状: (max_seq_len, d//2)
#         theta_ik = numerator.unsqueeze(1) / denominator
#         cos_theta = torch.cos(theta_ik)
#         sin_theta = torch.sin(theta_ik)
#         self.cos: Tensor
#         self.sin: Tensor
#         # 注册为 buffer（非可学习参数）
#         self.register_buffer('cos', cos_theta, persistent=False)
#         self.register_buffer('sin', sin_theta, persistent=False)

#     def forward(
#         self, 
#         in_query_or_key: torch.Tensor,  # 形状: (... sequence_length d_k)
#         token_positions: torch.Tensor   # 形状: (... sequence_length)
#     ) -> torch.Tensor:
#         """
#         对输入的查询或键张量应用 Rotary Position Embedding
        
#         参数:
#             in_query_or_key: 输入张量，形状为 (... sequence_length d_k)
#             token_positions: 每个 token 的位置索引，形状为 (... sequence_length)
        
#         返回:
#             应用 RoPE 后的张量，形状与输入一致
#         """
#         # 提取维度信息
#         *batch_dims, seq_len, d_k = in_query_or_key.shape
#         assert d_k == self.d_k, "输入特征维度与初始化时的 d_k 不匹配"

#         # 确保 token_positions 与输入的 sequence_length 一致
#         assert token_positions.shape[-1] == seq_len, "token_positions 的序列长度与输入不匹配"

#         # 处理输入维度（若 d_k 是奇数，先补零使维度为偶数）
#         if self.d != d_k:
#             in_padded = torch.nn.functional.pad(in_query_or_key, (0, self.d - d_k))
#         else:
#             in_padded = in_query_or_key

#         # 拆分特征维度为 (d//2, 2)，按对处理
#         in_reshaped = in_padded.reshape(*batch_dims, seq_len, self.d // 2, 2)

#         # 提取当前 token 位置对应的 cos 和 sin（利用高级索引）
#         # token_positions 的形状是 (... sequence_length)，需扩展为与 in_reshaped 兼容的形状
#         # 当执行 self.cos[token_positions] 时，PyTorch 的高级索引会根据 token_positions 中的每个位置索引，从 self.cos 中提取对应行，最终得到形状为 (... sequence_length, d//2) 的张量，实现 “按 token 位置提取旋转系数” 的逻辑。
#         cos = self.cos[token_positions]  # 形状: (... sequence_length d//2)
#         sin = self.sin[token_positions]  # 形状: (... sequence_length d//2)

#         # 应用旋转矩阵：[a, b] * [[cos, -sin], [sin, cos]] = [a*cos - b*sin, a*sin + b*cos]
#         a, b = in_reshaped[..., 0], in_reshaped[..., 1]
#         rotated_a = a * cos - b * sin
#         rotated_b = a * sin + b * cos

#         # 重组为原始维度
#         # 把两个形状为 (..., seq, d//2) 的张量合并为形状 (..., seq, d//2, 2) 的张量（每个特征对的两个分量在最后一维聚合）
#         rotated = torch.stack([rotated_a, rotated_b], dim=-1)
#         rotated = rearrange(rotated, '... seq d2 two -> ... seq (d2 two)') # 不能把two改成2
#         # rotated = rotated.reshape(*batch_dims, seq_len, self.d)
#         # 直接将 rotated_a 和 rotated_b 按最后一维堆叠，再重组维度
#         # rotated = rearrange([rotated_a, rotated_b], ' ... seq d2 2 -> ... seq (d2 2)')

#         # 若原始 d_k 是奇数，裁剪回原维度
#         if self.d != d_k:
#             rotated = rotated[..., :d_k]

#         return rotated
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        theta: float,
        d_k: int,        # query/key 的维度
        max_seq_len: int # 最大序列长度
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 确保 d_k 是偶数（RoPE 要求特征按对处理）
        d = d_k if d_k % 2 == 0 else d_k + 1  
        self.d = d

        # 这里默认在 CPU 上建表，之后通过 model.to(device) 或 forward 里的 .to() 迁移
        base = torch.full((1, d // 2), theta)               # [1, d//2]
        exp = torch.linspace(0, d - 2, d // 2) / d          # [d//2]
        denominator = torch.pow(base, exp)                  # [1, d//2]

        numerator = torch.arange(0, max_seq_len)            # [max_seq_len]

        # θ_ik = i / 分母 → 形状: (max_seq_len, d//2)
        theta_ik = numerator.unsqueeze(1) / denominator     # [max_seq_len, d//2]
        cos_theta = torch.cos(theta_ik)
        sin_theta = torch.sin(theta_ik)

        # 注册为 buffer（非可学习参数），这样 model.to(device) 时会自动迁移
        self.register_buffer('cos', cos_theta, persistent=False)  # [max_seq_len, d//2]
        self.register_buffer('sin', sin_theta, persistent=False)  # [max_seq_len, d//2]

    def forward(
        self, 
        in_query_or_key: torch.Tensor,  # 形状: (..., seq_len, d_k)
        token_positions: torch.Tensor   # 形状: (..., seq_len) 或 (seq_len,)
    ) -> torch.Tensor:
        """
        对输入的查询或键张量应用 Rotary Position Embedding
        
        参数:
            in_query_or_key: 输入张量，形状为 (..., seq_len, d_k)
            token_positions: 每个 token 的位置索引，形状为 (..., seq_len) 或 (seq_len,)
        
        返回:
            应用 RoPE 后的张量，形状与输入一致
        """
        # 提取维度信息
        *batch_dims, seq_len, d_k = in_query_or_key.shape
        assert d_k == self.d_k, f"输入特征维度 {d_k} 与初始化时的 d_k={self.d_k} 不匹配"

        # 确保 token_positions 与输入的 sequence_length 一致
        assert token_positions.shape[-1] == seq_len, (
            f"token_positions 的序列长度 {token_positions.shape[-1]} 与输入 {seq_len} 不匹配"
        )

        device = in_query_or_key.device

        # token_positions 需要是 long，并且放到和输入同一个 device 上
        if token_positions.dtype != torch.long:
            token_positions = token_positions.long()
        token_positions = token_positions.to(device)

        # 将 RoPE 表也迁移到输入所在设备（即使你忘记 model.to(device)，这里也兜底）
        cos = self.cos
        sin = self.sin
        if cos.device != device:
            cos = cos.to(device)
            sin = sin.to(device)

        # 处理输入维度（若 d_k 是奇数，先补零使维度为偶数）
        if self.d != d_k:
            in_padded = torch.nn.functional.pad(in_query_or_key, (0, self.d - d_k))
        else:
            in_padded = in_query_or_key

        # 拆分特征维度为 (d//2, 2)，按对处理
        # in_reshaped: (..., seq_len, d//2, 2)
        in_reshaped = in_padded.reshape(*batch_dims, seq_len, self.d // 2, 2)

        # 提取当前 token 位置对应的 cos 和 sin
        # self.cos: [max_seq_len, d//2]
        # token_positions: (..., seq_len)
        # 高级索引后 cos/sin 形状为 (..., seq_len, d//2)
        cos = cos[token_positions]
        sin = sin[token_positions]

        # 应用旋转矩阵：
        # [a, b] * [[cos, -sin], [sin, cos]] = [a*cos - b*sin, a*sin + b*cos]
        # in_reshaped: (..., seq_len, d//2, 2)
        a = in_reshaped[..., 0]  # (..., seq_len, d//2)
        b = in_reshaped[..., 1]  # (..., seq_len, d//2)

        # 这里 a/b 与 cos/sin 形状相同，按元素相乘即可
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos

        # 重组为原始维度
        # rotated: (..., seq_len, d//2, 2)
        rotated = torch.stack([rotated_a, rotated_b], dim=-1)
        rotated = rearrange(rotated, '... seq d2 two -> ... seq (d2 two)')  # (..., seq_len, d)

        # 若原始 d_k 是奇数，裁剪回原维度
        if self.d != d_k:
            rotated = rotated[..., :d_k]

        return rotated



def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    pre_softmax = einsum(Q,K, '... queries d_k, ... keys d_k -> ... queries keys') / math.sqrt(d_k)
    
    if mask is not None:
        assert pre_softmax.shape == mask.shape
        #pre_softmax[~mask] = - torch.inf # 使用Pytorch智能索引，在mask里每一个为false的位置，pre_softmax都变成负无穷
        # 最好避免原地操作，而改用mask_fill
        masked = pre_softmax.masked_fill(~mask, -torch.inf)
    else:
        masked = pre_softmax
    # Softmax 作用于 Q^T K 的最后一个维度（即 m 维度，对应键的序列长度），目的是对每个查询（n 中的每个元素），在所有键的位置上计算归一化的注意力权重（使得每个查询对应的权重和为 1）
    res = einsum(softmax(masked, dim=-1), V, '... queries keys , ... keys d_v -> ... queries d_v') 
    # masked是' ... queries keys'的，对于最后一维求softmax，就是对任意一个query，输出注意力都是归一化的。Softmax 对 Key 维（而非 Query 维）归一化，是为了让每个 Query 都能得到一个‘针对所有 Key 的归一化权重分’
    # value 必须等于 keys，这是因为每一个query对应keys个键，做内积就是为了知道某一个query与所有键的相似度，进而指导输出注意力的分配，即value的分配，所以一个key就对应一个value
    return res

def softmax(
    in_features: Float[Tensor, " ..."],
    dim: int
    ) -> Float[Tensor, " ..."]:
    in_dtype = in_features.dtype
    in_features = in_features.to(torch.float32)
    max_entry = torch.max(in_features, dim=dim, keepdim=True).values # 应该是从 '... dim ...' -> '... 1 ...'
    subtracted = torch.sub(in_features, max_entry)
    exp = torch.exp(subtracted)
    exp_sum = torch.sum(exp, dim=dim, keepdim=True)
    res = exp / exp_sum
    res = res.to(in_dtype)
    return res

def SiLU(in_features:Float[Tensor, "..."])->Float[Tensor,"..."]:
    return in_features * torch.sigmoid(in_features)


# 测试函数
# def test_swiglu_load_state_dict():
#     # 1. 配置测试参数（模拟 TransformerBlock 中的初始化参数）
#     d_model = 512  # 模型维度
#     d_ff = 1344    # 与 (8/3)*512=1365.333 向上取整到64倍数一致（1344是64*21，实际可调整，此处仅为测试）
#     device = torch.device("cpu")  # 可改为 "cuda" 测试GPU
#     dtype = torch.float32

#     # 2. 模拟 TransformerBlock 中 ffn 权重初始化
#     ffn = nn.Module()
#     # 初始化 ffn.w1.weight（形状：(d_model, d_ff) = (512, 1344)）
#     ffn.w1 = nn.Module()
#     ffn.w1.weight = nn.Parameter(torch.randn(d_model, d_ff, device=device, dtype=dtype))  # 用随机值模拟测试权重
#     # 初始化 ffn.w2.weight（形状：(d_ff, d_model) = (1344, 512)）
#     ffn.w2 = nn.Module()
#     ffn.w2.weight = nn.Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype))
#     # 初始化 ffn.w3.weight（形状：(d_model, d_ff) = (512, 1344)）
#     ffn.w3 = nn.Module()
#     ffn.w3.weight = nn.Parameter(torch.randn(d_model, d_ff, device=device, dtype=dtype))

#     # 3. 模拟 TransformerBlock 中构建 ffn_weights（键名：w1_weight/w2_weight/w3_weight）
#     ffn_weights = {
#         'w1_weight': ffn.w1.weight.T,
#         'w2_weight': ffn.w2.weight.T,
#         'w3_weight': ffn.w3.weight.T,
#     }

#     # 4. 实例化 SwiGLU
#     swiglu = SwiGLU(
#         d_model=d_model,
#         d_ff=d_ff,  # 显式传入d_ff，与 TransformerBlock 一致
#         device=device,
#         dtype=dtype
#     )

#     # 5. 保存 SwiGLU 初始化时的权重（用于后续对比，验证是否被覆盖）
#     init_w1 = swiglu.w1_weight.data.clone()
#     init_w2 = swiglu.w2_weight.data.clone()
#     init_w3 = swiglu.w3_weight.data.clone()

#     # 6. 执行 load_state_dict（核心测试步骤）
#     try:
#         swiglu.load_state_dict(ffn_weights, strict=True)
#         print("✅ load_state_dict 执行成功（无键名不匹配错误）")
#     except Exception as e:
#         assert False, f"❌ load_state_dict 执行失败：{str(e)}"

#     # 7. 校验权重加载结果（核心校验项）
#     print("\n=== 权重加载校验 ===")
#     # 校验1：w1_weight 被正确覆盖（与 ffn.w1.weight 完全一致）
#     w1_match = torch.allclose(swiglu.w1_weight.data, ffn.w1.weight.T.data)
#     assert w1_match, "❌ w1_weight 加载失败（与 ffn.w1.weight 不一致）"
#     print("✅ w1_weight 加载成功")

#     # 校验2：w2_weight 被正确覆盖（与 ffn.w2.weight 完全一致）
#     w2_match = torch.allclose(swiglu.w2_weight.data, ffn.w2.weight.T.data)
#     assert w2_match, "❌ w2_weight 加载失败（与 ffn.w2.weight 不一致）"
#     print("✅ w2_weight 加载成功")

#     # 校验3：w3_weight 被正确覆盖（与 ffn.w3.weight 完全一致）
#     w3_match = torch.allclose(swiglu.w3_weight.data, ffn.w3.weight.T.data)
#     assert w3_match, "❌ w3_weight 加载失败（与 ffn.w3.weight 不一致）"
#     print("✅ w3_weight 加载成功")

#     # 校验4：加载后的权重与初始化权重不同（确保确实被覆盖）
#     init_w1_diff = not torch.allclose(swiglu.w1_weight.data, init_w1)
#     init_w2_diff = not torch.allclose(swiglu.w2_weight.data, init_w2)
#     init_w3_diff = not torch.allclose(swiglu.w3_weight.data, init_w3)
#     assert init_w1_diff and init_w2_diff and init_w3_diff, "❌ 权重未被覆盖（与初始化值一致）"
#     print("✅ 加载的权重成功覆盖初始化权重")

#     # 校验5：权重形状正确（匹配 SwiGLU 定义的形状）
#     assert swiglu.w1_weight.shape == (d_ff, d_model), f"❌ w1_weight 形状错误：预期 ({d_ff}, {d_model})，实际 {swiglu.w1_weight.shape}"
#     assert swiglu.w2_weight.shape == (d_model, d_ff), f"❌ w2_weight 形状错误：预期 ({d_model}, {d_ff})，实际 {swiglu.w2_weight.shape}"
#     assert swiglu.w3_weight.shape == (d_ff, d_model), f"❌ w3_weight 形状错误：预期 ({d_ff}, {d_model})，实际 {swiglu.w3_weight.shape}"
#     print("✅ 所有权重形状符合预期")

#     # 8. 额外校验：SwiGLU 前向传播正常（确保加载权重后功能不受影响）
#     test_input = torch.randn(2, 10, d_model, device=device, dtype=dtype)  # (batch, seq_len, d_model)
#     try:
#         output = swiglu(test_input)
#         assert output.shape == (2, 10, d_model), f"❌ 前向传播输出形状错误：预期 (2,10,{d_model})，实际 {output.shape}"
#         print("✅ 前向传播正常，输出形状符合预期")
#     except Exception as e:
#         assert False, f"❌ 前向传播失败：{str(e)}"

#     print("\n🎉 所有测试通过！SwiGLU.load_state_dict 功能正常")

if __name__ == "__main__":
    batch_size = 4
    # vocab_size = 5
    # inputs = torch.rand(batch_size, vocab_size)
    # targets =  torch.randint(0,vocab_size-1,(batch_size,))

    # print(cross_entropy(1000*inputs, targets))
    