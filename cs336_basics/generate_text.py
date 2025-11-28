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
from cs336_basics.model import TransformerLM
from cs336_basics.my_tokenizer import Tokenizer
import pickle
import os
from cs336_basics.training import load_checkpoint
import json

def temp_scaled_softmax(
    in_features: Float[Tensor, " ..."],
    dim: int,
    tau: Float = 1
    ) -> Float[Tensor, " ..."]:
    in_dtype = in_features.dtype
    in_features = in_features.to(torch.float32)
    in_features = in_features / tau
    max_entry = torch.max(in_features, dim=dim, keepdim=True).values # 应该是从 '... dim ...' -> '... 1 ...'
    subtracted = torch.sub(in_features, max_entry)
    exp = torch.exp(subtracted)
    exp_sum = torch.sum(exp, dim=dim, keepdim=True)
    res = exp / exp_sum
    res = res.to(in_dtype)
    return res

def top_p_sample(
    prob: Float[Tensor, 'batch_size sequence_length vocab_size'],
    thresh: Float,
    tau: Float
) -> Int[Tensor, 'batch_size sequence_length']:
    b = prob.size(0)
    normalized_prob = temp_scaled_softmax(prob, -1, tau) # (b, seq, vocab)
    sorted_prob, indices = torch.sort(normalized_prob, dim=-1, descending=True) # (b, seq, vocab)
    cum_sum = torch.cumsum(sorted_prob, dim=-1) # (b, seq, vocab)
    nucleus = cum_sum < thresh
    #print(nucleus)
    nucleus = nucleus | torch.roll(nucleus, shifts=1, dims= -1)
    #print(nucleus)
    chosen_prob = sorted_prob * nucleus.float()
    #print(chosen_prob)
    # chosen_prob_sum = torch.sum(chosen_prob,dim=-1,keepdim=True) # (b, seq, 1)
    # normalized_chosen_prob = chosen_prob / (chosen_prob_sum + 1e-8)  # (b, seq, vocab)
    # print(normalized_chosen_prob)
    # normalized_chosen_prob
    chosen_prob_flat = torch.flatten(chosen_prob,start_dim=0,end_dim=1)
    chosen_indices = torch.multinomial(chosen_prob_flat,num_samples=1,replacement=False)  # (b, seq) multinomial会自动概率归一化，只要输入加起来大于零且有限即可
    chosen_indices = rearrange(chosen_indices, '(b s) 1 -> b s 1', b = b)
    #print(chosen_indices) 
    sampled_tokens = torch.gather(indices, dim=-1, index=chosen_indices)
    sampled_tokens = rearrange(sampled_tokens, 'b s 1 -> b s')
    return sampled_tokens

def top_p_sample_next_token(
    prob: Float[Tensor, 'vocab_size'],
    thresh: Float,
    tau: Float
) -> Int[Tensor, '1']:
    normalized_prob = temp_scaled_softmax(prob, -1, tau) # (vocab,)
    sorted_prob, indices = torch.sort(normalized_prob, dim=-1, descending=True) # (b, seq, vocab)
    cum_sum = torch.cumsum(sorted_prob, dim=-1) # (vocab,)
    nucleus = cum_sum < thresh
    nucleus = nucleus | torch.roll(nucleus, shifts=1, dims= -1)
    if nucleus[0] == False: nucleus[0] = True
    chosen_prob = sorted_prob * nucleus.float()
    chosen_prob_sum = torch.sum(chosen_prob,dim=-1,keepdim=True) # (1,)
    chosen_prob = chosen_prob / (chosen_prob_sum + 1e-8)  # ( vocab)
    chosen_indices = torch.multinomial(chosen_prob, num_samples=1,replacement=False)  #multinomial会自动概率归一化，只要输入加起来大于零且有限即可
    sampled_tokens = torch.gather(indices, dim=-1, index=chosen_indices)
    return sampled_tokens


def decode_next_tokens(
    token_prob: Tensor,
    max_len: int,
    tau: Float,
    top_p_thresh: Float
):
    i = 0
    while i < max_len:
        top_p_sample(token_prob, thresh=top_p_thresh, tau=tau)

def generating_text(
    promt,
    tokenizer: Tokenizer,
    model: TransformerLM,
    thresh = 0.8,
    context_length = 64,
    tau = 0.5, # 大于1时概率分布更加平坦
    end_sign = '<|endoftext|>'
):

    device = next(model.parameters()).device
    end_bytes = end_sign.encode()

    end_token = tokenizer.vocab_inv[end_bytes]
    token = torch.tensor(tokenizer.encode(promt),dtype=torch.long, device=device) # list[int] -> Tensor[" batch_size sequence_length"]

    i = 0
    next_tokens = []
    while len(token) < min(context_length, model.max_seq_len):
        # print(len(token))
        token = rearrange(token, 'seq_len -> 1 seq_len') # 生出batch这一维
        next_tokens_prob = model.forward(token) # "1 sequence_length" -> "1 seq_len vocab_size"
        prob = next_tokens_prob[0][-1] # 只取最后一个token对应的next token prob, (vocab_size,)
        next_token = top_p_sample_next_token(prob=prob, thresh=thresh,tau=tau) # (1,)
        next_tokens.append(next_token.item())
        token = token.squeeze(0) #(seq_len,)
        # next_token = next_token.squeeze() #(1,) -> (0)
        token = torch.concat([token, next_token], dim=0)
        i = i + 1
        if end_token == next_token : break
    return tokenizer.decode(token.tolist()), tokenizer.decode(next_tokens)


def try_temp_scaled_softmax():
    inputs = torch.rand(2,3)
    res1 = temp_scaled_softmax(inputs, -1, 1)
    res2 = temp_scaled_softmax(inputs, -1, 0.1)
    res3 = temp_scaled_softmax(inputs, -1, 2)
    # 可以看出res2的分布更加尖锐，res3的分布更加平坦
    print('res1\n', res1)
    print('res2\n', res2)
    print('res3\n', res3)    

def try_top_p_sample():
    inputs = torch.rand(2,3,5)
    print(top_p_sample_next_token(inputs,tau=0.5, thresh=0.8))

def gen_from_model(
    vocab_path,
    merges_path,
    ckpt_folder_path,
    max_ans_len,
    tau=0.9,
    thresh=0.8
):
    model_json_path = os.path.join(ckpt_folder_path,'training_state.json')
    model_path = os.path.join(ckpt_folder_path,'training_latest.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f'Using vocab from: {vocab_path}')
    print(f'Using merges from: {merges_path}')
    print(f'Using model from: {model_path}')
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)

    special_tokens = ['<|endoftext|>']
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
        num_processes=4
    )

    with open(model_json_path, "r") as f:
        data = json.load(f)

    hp = data["hyperparams"]
    context_length = hp["context_length"]
    d_model        = hp["d_model"]
    num_layers     = hp["num_layers"]
    num_heads      = hp["num_heads"]
    d_ff           = hp["d_ff"]
    rope_theta     = hp["rope_theta"]

    print("context_length =", context_length)
    print("d_model        =", d_model)
    print("num_layers     =", num_layers)
    print("num_heads      =", num_heads)
    print("d_ff           =", d_ff)
    print("rope_theta     =", rope_theta)
    print("vocab size     =", tokenizer.vocab_len)

    # === Step 1: 构建“干净”的模型，结构必须和训练时一致 ===
    model = TransformerLM(
        vocab_size=tokenizer.vocab_len,
        context_length=context_length,   # 要和训练时的一致
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(device)

    _ , _ = load_checkpoint(
        src=model_path,
        model=model,
        optimizer=None,      # 推理不用 optimizer
        map_location=device
    )
    model.eval()  # 切换到 eval 模式
    print("Input 'quit' to exit dialog.")
    while True:
        promt = input('User > ')
        if promt == 'quit': break
        if not promt.strip():
            continue
        _ , ans = generating_text(
            promt=promt,
            context_length=max_ans_len,
            tokenizer=tokenizer,
            model=model,
            tau=tau,
            thresh=thresh
        )
        print('Ans  >', ans, end='\n\n')

if __name__ == "__main__":
    TinyStories_vocab_path = './data/tinystories_vocab_10000.pkl'
    TinyStories_merges_path = './data/tinystories_merges_10000.pkl'
    owt_vocab_path = './data/OpenWebText_vocab_32000.pkl'
    owt_merges_path = './data/OpenWebText_merges_32000.pkl'
    ckpt_folder_path_tiny = './data/trained_models/2025-11-26-11-11-21-tiny-val-loss-1.3861'
    ckpt_folder_path_owt = './data/trained_models/2025-11-26-17-16-28-owt-val-loss-3.94'
    model_choice = input(' 1: tiny\t 2: owt\nChoose model: ')
    if model_choice == '1':
        vocab_path = TinyStories_vocab_path
        merges_path = TinyStories_merges_path
        ckpt_folder_path=ckpt_folder_path_tiny
    elif model_choice == '2':
        vocab_path=owt_vocab_path
        merges_path=owt_merges_path
        ckpt_folder_path=ckpt_folder_path_owt
    else:
        print('No choice match.')
        assert False
    gen_from_model(
        vocab_path=vocab_path,
        merges_path=merges_path,
        ckpt_folder_path=ckpt_folder_path,
        max_ans_len=256,
        tau=1,
        thresh=0.8
    )
