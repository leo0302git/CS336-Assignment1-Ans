from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

# 注释以下内容可以避免inport影响scalene的性能分析
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
# 新增
import regex as re
from collections import defaultdict
import json
import time
from pathlib import Path
import pathlib
import pickle
from typing import Iterable, Iterator
import tempfile
import numpy as np
import tempfile
import multiprocessing
from torch import nn
# uv test 可用相对导入
from cs336_basics.my_tokenizer import pre_tokenization, pre_tokenization_para, pre_tokenization_para_pro, Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.model import Linear, Embedding, RMSNorm, get_device, SwiGLU, RotaryPositionalEmbedding, softmax, scaled_dot_product_attention, MultiheadSelfAttention, TransformerBlock, TransformerLM, SiLU
from cs336_basics.training import AdamW, cross_entropy, lr_cosine_schedule, gradient_clipping, get_batch, save_checkpoint, load_checkpoint

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    device = get_device()
    linear = Linear(in_features=d_in, out_features=d_out, device=device, dtype=torch.float32)
    #linear.weight = torch.nn.Parameter(weights)  # 恢复权重原始形状
    weight_to_load = {
        'weight': weights
    }
    linear.load_state_dict(weight_to_load)
    res = linear.forward(x=in_features)
    return res



def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    device = get_device()
    emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    emb.weight = torch.nn.Parameter(weights) 
    res = emb.forward(token_ids=token_ids)
    return res


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_model=d_model,d_ff=d_ff)
    weights = {
        'w1.weight': w1_weight, # 其实是内部Linear类的weight，根据Linear类的定义，out_feature维度在前
        'w2.weight': w2_weight,
        'w3.weight': w3_weight
    }
    swiglu.load_state_dict(weights)
    return swiglu.forward(in_features)

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q=Q, V=V, K=K, mask=mask)

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha_weights = {
        'q_proj.weight':q_proj_weight,
        'k_proj.weight':k_proj_weight,
        'v_proj.weight':v_proj_weight,
        'output_proj.weight':o_proj_weight
    }
    d_k = q_proj_weight.size(-2)
    d_in = q_proj_weight.size(-1)
    d_v = v_proj_weight.size(-2)
    max_seq_len = in_features.size(-2)
    mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, d_in=d_in, max_seq_len=max_seq_len, with_rope=False, theta=-1)
    mha.load_state_dict(mha_weights,strict=True)
    return mha.forward(in_features=in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha_weights = {
        'q_proj.weight':q_proj_weight,
        'k_proj.weight':k_proj_weight,
        'v_proj.weight':v_proj_weight,
        'output_proj.weight':o_proj_weight
    }
    d_k = q_proj_weight.size(-2)
    d_in = q_proj_weight.size(-1)
    d_v = v_proj_weight.size(-2)
    max_seq_len = in_features.size(-2)
    mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, d_in=d_in, max_seq_len=max_seq_len, theta=theta, with_rope=True) 
    mha.load_state_dict(mha_weights,strict=True)
    return mha.forward(in_features=in_features,token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta=theta,d_k=d_k,max_seq_len=max_seq_len)
    return rope.forward(in_query_or_key=in_query_or_key, token_positions=token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, nn.Parameter],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input. 相当于MHA中的d_in
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    d_k = weights['attn.q_proj.weight'].size(-2)
    d_in = weights['attn.q_proj.weight'].size(-1)
    d_v = weights['attn.v_proj.weight'].size(-2)
    transformer_block = TransformerBlock(d_model=d_model, num_heads=num_heads,d_ff=d_ff, d_k=d_k, d_v=d_v, d_in=d_in, max_seq_len=max_seq_len, theta=theta)
    transformer_block.load_state_dict(weights,strict=True)
    return transformer_block.forward(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int, #`sequence_length` is at most `context_length`.
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformerlm = TransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta)

    transformerlm.load_state_dict(weights)
    return transformerlm.forward(in_indices)

    
    
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSNorm(d_model=d_model,eps=eps)
    weight = {
        'weight': weights
    }
    rmsnorm.load_state_dict(weight)
    return rmsnorm.forward(in_features)




def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return SiLU(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    return get_batch(dataset=dataset, batch_size=batch_size, context_length=context_length, device=device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features=in_features, dim=dim)

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs=inputs, targets=targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    gradient_clipping(parameters=parameters, max_l2_norm=max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return lr_cosine_schedule(t = it, alpha_max=max_learning_rate, alpha_min=min_learning_rate, Tw=warmup_iters, Tc=cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    it, _ = load_checkpoint(src, model, optimizer)
    return it


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    tokenizer = Tokenizer(vocab=vocab, merges=merges,special_tokens=special_tokens)
    return tokenizer
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #print('enter run_train_bpe: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start_time = time.time()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    COMPILE_PAT = re.compile(PAT)
    # pretokenization

    pre_tokens = defaultdict(int)
    chunk_idx = 0
    num_processes = 15
    proc_list = []
    queue = multiprocessing.Queue()
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # p = multiprocessing.Process(target=pre_tokenization_para, args=(chunk, chunk_idx, COMPILE_PAT, special_tokens, queue))
        p = multiprocessing.Process(
        target=pre_tokenization_para_pro,
        args=(
            input_path,  # 传文件路径（字符串，支持跨进程传递）
            start,            # 分块起始位置
            end,              # 分块结束位置
            chunk_idx,
            COMPILE_PAT,      # 预编译的正则（全局变量，或传参）
            special_tokens,
            queue
        )
        )
            
        proc_list.append(p)
        p.start()
        chunk_idx += 1

    c = 0
    while c < len(proc_list):
        chunk_idx, pre_tokens_part = queue.get()
        #print(f'No. {chunk_idx} value returned.')
        for t in pre_tokens_part:
            pre_tokens[t] += pre_tokens_part[t]
        c += 1
    
    for p in proc_list:
        p.join()

    # pre_tokens_path = 'D:/CollegeLife/self_learning/CS336 2025/assignment1-basics/data/pre_tokens_owt.pkl'
    # with open (pre_tokens_path, 'rb') as f:
    #     pre_tokens = pickle.load(f)
    # Path(pre_tokens_path).parent.mkdir(parents=True, exist_ok=True)
    # # 序列化保存
    # with open(pre_tokens_path, "wb") as f:
    #     pickle.dump(pre_tokens, f)
    
    pre_token_time = time.time()
    # print('num_process: ', num_processes,'\npre_tokenization time: ', pre_token_time - start_time)
    # init vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(0,256)}
    new_idx = len(vocab)
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[new_idx] = token_bytes
            new_idx += 1

    merges: list[tuple[bytes, bytes]] = []
    # 初始化 pair_cnt
    pair_cnt = defaultdict(int)
    #pair_to_tokens = defaultdict(list) # key为字节对，value为list，list中的元素是这个字节对所在的pre_token，比如key=(b't',b'h'),value = [(b't',b'h',b'e'),(b't',b'h',b'y'),...]
    # for token in pre_tokens:
    #     j = 0
    #     while j < len(token) - 1: 
    #         pair = (token[j], token[j+1])
    #         pair_cnt[pair] += pre_tokens[token]
    #         if token not in pair_to_tokens[pair]: pair_to_tokens[pair].append(token) # 避免重复。这是关键！
    #         j += 1
    # 关键新增：为每个字节对的列表维护一个临时set，用于O(1)去重判断
    pair_token_set = defaultdict(set)  # key: 字节对，value: 已加入列表的token集合（临时用）

    for token in pre_tokens:
        # 优化1：提前缓存token的出现次数和长度，减少循环内重复操作
        token_count = pre_tokens[token]
        token_len = len(token)
        
        # 优化2：用for循环替代while循环，简化逻辑并减少条件判断
        for j in range(token_len - 1):
            pair = (token[j], token[j + 1])
            
            # 1. 统计字节对计数（原有逻辑不变）
            pair_cnt[pair] += token_count
            
            # 2. 去重添加token到pair_to_tokens[pair]（核心优化）
            if token not in pair_token_set[pair]:  # O(1)判断，替代原列表O(n)判断
                #pair_to_tokens[pair].append(token)  # 仍添加到列表，适配其他代码
                pair_token_set[pair].add(token)     # 同步记录到set，避免后续重复添加

    # del pair_token_set
    init_pair_token_set_time = time.time()
    # print('init_pair_token_set_time:' , init_pair_token_set_time - pre_token_time)
    while len(vocab) < vocab_size:
        if len(pair_cnt.values()) == 0:  
            #print('pair count break')
            break # every token is merged into single byte sequence. Fully tokenized
        max_count = max(pair_cnt.values())
        candidates = [pair for pair in pair_cnt if pair_cnt[pair] == max_count]
        max_pair = max(candidates)
        #print('当前最大对：', max_pair, 'new_idx', new_idx)
        byte1, byte2 = max_pair
        # enlarge the vocab and merges
        vocab[new_idx] = byte1 + byte2
        # print(new_idx)
        new_idx += 1
        merges.append(max_pair)
        # merge the pre_tokens
        # pre_tokens cannot be changed during iteration
        token_changes = []
        # for token in pair_to_tokens[max_pair]:
        for token in pair_token_set[max_pair]:
            new_token = []
            j = 0
            while j < len(token):
                if j + 1 < len(token) and (token[j], token[j+1]) == max_pair:
                    new_token.append(vocab[new_idx-1])
                    j += 2
                else:
                    new_token.append(token[j])
                    j += 1
            token_changes.append((token, tuple(new_token), pre_tokens[token]))
        # 批量统计待删除/新增的 pair 计数（减少重复操作）
        broken_pair_batch = defaultdict(int)  # {broken_pair: 总减少量}
        add_pair_batch = defaultdict(int)     # {add_pair: 总增加量}
        # 记录需同步到 pair_to_tokens 的 (pair, token) 关系
        to_remove = defaultdict(set)  # {pair: 待删除的 token}
        to_add = defaultdict(set)     # {pair: 待新增的 token}

        for change in token_changes:
            t_old, t_new, cnt = change
            # 缓存高频值（避免循环内重复查字典/计算长度）
            t_old_len = len(t_old)
            t_new_len = len(t_new)
            t_old_cnt = cnt  # pre_tokens[t_old] 就是 cnt，无需重复访问
            
            # 1. 处理旧 token：收集待删除的 pair 和计数
            for i in range(t_old_len - 1):
                broken_pair = (t_old[i], t_old[i+1])
                # 批量累计计数（一次循环完成，无需逐次减）
                broken_pair_batch[broken_pair] += t_old_cnt
                # 记录待删除的 token（用 set 去重，避免重复删除）
                to_remove[broken_pair].add(t_old)
            
            # 2. 处理新 token：收集待新增的 pair 和计数
            for j in range(t_new_len - 1):
                add_pair = (t_new[j], t_new[j+1])
                # 批量累计计数
                add_pair_batch[add_pair] += t_old_cnt  # 新 token 计数=旧 token 计数
                # 记录待新增的 token（set 去重）
                to_add[add_pair].add(t_new)
            
            # 3. 更新 pre_tokens（原有逻辑，仅缓存值）
            pre_tokens[t_new] = cnt
            del pre_tokens[t_old]

        for pair, total_sub in broken_pair_batch.items():
            # 更新计数（一次减法，替代多次循环减法）
            pair_cnt[pair] -= total_sub
            # 移除 token（set 操作 O(1)，替代 list 的 O(n) remove）
            tokens_to_del = to_remove[pair]
            pair_token_set[pair].difference_update(tokens_to_del)
            # 计数≤0时，直接删除该 pair（避免后续无效操作）
            if pair_cnt[pair] <= 0:
                del pair_cnt[pair]
                # del pair_to_tokens[pair]
                del pair_token_set[pair]

        for pair, total_add in add_pair_batch.items():
            # 更新计数（一次加法，替代多次循环加法）
            pair_cnt[pair] += total_add
            # 新增 token（set 去重 O(1)）
            tokens_to_add = to_add[pair]
            pair_token_set[pair].update(tokens_to_add)

    # print('loop time: ', time.time() - init_pair_token_set_time)
    return (vocab, merges)






if __name__ == '__main__':
    special_tokens = ["<|endoftext|>"]
    special_tokens = None
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    OpenWebText_vocab_path = './data/OpenWebText_vocab_32000.pkl'
    OpenWebText_merges_path = './data/OpenWebText_merges_32000.pkl'
    dump_path = [OpenWebText_vocab_path, OpenWebText_merges_path]
    with open(OpenWebText_vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(OpenWebText_merges_path, "rb") as f:
        merges = pickle.load(f)
    
    tokenizer = Tokenizer(vocab=vocab, merges=merges,special_tokens=special_tokens,num_processes=4)
    # test Tokenizer.encode()
    # try_txt = 'the cat ate'

    # try_txt = ''

    # try_txt = " the at the "

    # try_txt = "Héllò hôw <|endoftext|><|endoftext|> are ü? Héllò 🙃<|endoftext|>"
    # try_txt_token = tokenizer.encode(try_txt)
    # print(try_txt_token)

    # test Tokenizer.encode_iterable()
    # encode_txt_path = './data/TinyStoriesV2-GPT4-valid.txt'
    # with open(encode_txt_path, 'r', encoding='utf-8') as f:
    #     token_generator = tokenizer.encode_iterable(f)
    #     TinyStoriesV2_valid_token = []
    #     for token_id in token_generator:
    #         TinyStoriesV2_valid_token.append(token_id)
    #         # print(token_id, vocab[token_id])
    
    # test Tokenizer.decode()
    # try_txt_decoded = tokenizer.decode(try_txt_token)
    # print('try_txt_decoded:\n', try_txt_decoded)
    # assert try_txt_decoded == try_txt
    


    # test: test_train_bpe.py
    # from common import FIXTURES_PATH, gpt2_bytes_to_unicode
    # # from t_train_bpe_need import pre_tokenization
    # FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"
    # input_path = FIXTURES_PATH / "corpus.en"
    # vocab, merges = run_train_bpe(
    #     input_path=input_path,
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )
    # # Path to the reference tokenizer vocab and merges
    # reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    # reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # # Compare the learned merges to the expected output merges
    # gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    # with open(reference_merges_path, encoding="utf-8") as f:
    #     gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
    #     reference_merges = [
    #         (
    #             bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
    #             bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
    #         )
    #         for merge_token_1, merge_token_2 in gpt2_reference_merges
    #     ]
    # i = 0
    # while i < len(merges):
    #     if merges[i] != reference_merges[i]: 
    #         pass
    #     i += 1
    # assert merges == reference_merges

    # # Compare the vocab to the expected output vocab
    # with open(reference_vocab_path, encoding="utf-8") as f:
    #     gpt2_reference_vocab = json.load(f)
    #     reference_vocab = {
    #         gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
    #         for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
    #     }
    # # Rather than checking that the vocabs exactly match (since they could
    # # have been constructed differently, we'll make sure that the vocab keys and values match)
    # assert set(vocab.keys()) == set(reference_vocab.keys())
    # assert set(vocab.values()) == set(reference_vocab.values())
    # input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    # vocab, merges = run_train_bpe(
    #     input_path=input_path,
    #     vocab_size=1000,
    #     special_tokens=["<|endoftext|>"],
    # )

    # # Check that the special token is not in the vocab
    # vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    # for word_bytes in vocabs_without_specials:
    #     assert b"<|" not in word_bytes
    # print(snapshot)
    # snapshot.assert_match(
    #     {
    #         "vocab_keys": set(vocab.keys()),
    #         "vocab_values": set(vocab.values()),
    #         "merges": merges,
    #     },
    # )
