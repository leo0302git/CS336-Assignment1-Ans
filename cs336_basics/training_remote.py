from collections.abc import Callable, Iterable 
from typing import Optional 
import torch 
from torch import Tensor
from torch import nn
import math  
import os
import sys
import matplotlib

# 在 Linux 且没有 DISPLAY（典型 headless 容器）时，强制用 Agg
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")  # 要在 import pyplot 之前调用

import matplotlib.pyplot as plt
import json
from jaxtyping import Bool, Float, Int
from typing import Iterable, Iterator
import numpy.typing as npt
import numpy as np
import random
import typing
import pickle
from cs336_basics.model import TransformerLM
from cs336_basics.my_tokenizer import Tokenizer
import time
from einops import einsum, repeat, reduce, rearrange
import wandb
from wandb.sdk.wandb_run import Run


class SGD(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3): 
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr} 
        super().__init__(params, defaults)  
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups:  
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]: 
                if p.grad is None: continue  
                state = self.state[p] # Get state associated with p. 
                t = state.get("t", 0) # Get iteration number from the state, or initial value. 
                grad = p.grad.data # Get the gradient of loss with respect to p.  
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. 
                state["t"] = t + 1 # Increment iteration number.  return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01): 
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)
        # 接下来对于每一个可学习参数，初始化其动量
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['t'] = 1 # 迭代次数
                self.state[p]['m'] = torch.zeros_like(p.data) # 一阶动量
                self.state[p]['v'] = torch.zeros_like(p.data) # 二阶动量
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure()
        for group in self.param_groups:  
            lr = group["lr"] # Get the learning rate.
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]: 
                if p.grad is None: continue  
                state = self.state[p] # Get state associated with p. 
                t = state.get("t") # Get iteration number from the state, or initial value. 
                grad = p.grad.data # Get the gradient of loss with respect to p.  
                self.state[p]['m'] = beta1 * self.state[p]['m'] + (1-beta1) * grad
                self.state[p]['v'] = beta2 * self.state[p]['v'] + (1-beta2) * grad**2
                lr_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data -= lr_t * self.state[p]['m'] / (torch.sqrt(self.state[p]['v']) + eps)
                p.data -= lr * weight_decay * (p.data)
                state["t"] = t + 1 # Increment iteration number.  return loss

def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],  # 模型输出的logits（未经过softmax）
    targets: Int[Tensor, "batch_size"]              # 目标标签索引（0 <= targets < vocab_size）
) -> Float[Tensor, ""]:
    # 对每个样本的logits减去其最大值（核心数值稳定化步骤）
    # 沿词汇表维度（最后一维）计算最大值，保持维度以便广播
    max_logit = torch.max(inputs, dim=-1, keepdim=True).values  # shape: (batch_size, 1)
    inputs_shifted = inputs - max_logit  # 消除指数爆炸，shape: (batch_size, vocab_size)
    
    # 计算所有logits的指数和（softmax的分母）
    exp_sum = torch.sum(torch.exp(inputs_shifted), dim=-1)  # shape: (batch_size,) 当有极大数出现时，exp_sum为1
    
    # 提取目标位置的logit（已偏移）
    # 生成行索引：[0, 1, ..., batch_size-1]
    row_indices = torch.arange(inputs.size(0), device=inputs.device)
    target_logits_shifted = inputs_shifted[row_indices, targets]  # shape: (batch_size,)
    
    # 计算负对数似然：-log(exp(target) / sum(exp(all))) = log(sum) - target
    neg_log_likelihood = torch.log(exp_sum) - target_logits_shifted  # shape: (batch_size,)
    
    # 对批次取平均（与PyTorch默认行为一致）
    return torch.mean(neg_log_likelihood)

def adamwAccounting():
    vocab_size = 50257  
    vocab = vocab_size
    context_length = 1024  
    seq = context_length
    num_layers = 48  
    d_model = 1600 
    num_heads = 25  
    d_ff = 6400
    B = 1024
    steps = 400 * 1000
    A100_FLOPs_per_sec = 19.5e12
    A100_MFU = 0.5
    A100_num = 1
    params = 2 * d_model * vocab_size + num_layers * (4 * d_model^2 + 3 * d_ff * d_model + 2 * d_model) + d_model
    forward_prop_FLOPs = 2 * B * seq * d_model * vocab + num_layers * (8 * B * seq * d_model^2 + 4 * B * seq^2 * d_model + 6 * B * seq * d_model * d_ff) 
    activation = B * seq * d_model + num_layers * (7 * seq * d_model + 2 * B * seq^2 + 3 * B * seq * d_ff) + B * seq * d_model + B * seq * vocab_size + B * seq 
    update_FLOPs = 12 * params
    bp_FLOPs = 2 * forward_prop_FLOPs
    FLOPs_per_step = forward_prop_FLOPs + update_FLOPs + bp_FLOPs
    print(f'FLOPs_per_step: {FLOPs_per_step / (1024**4)} tera FLOPs')
    print(f'forward FLOPs (%): {forward_prop_FLOPs/FLOPs_per_step * 100}%')
    print(f'update FLOPs (%): {update_FLOPs/FLOPs_per_step *100}%')
    print(f'backprop FLOPs (%): {bp_FLOPs/FLOPs_per_step *100}%')

    total_day = steps * FLOPs_per_step / (A100_FLOPs_per_sec * A100_MFU * A100_num) / (60 * 60 * 24)
    print(f'Training {steps} steps on {A100_num} A100, with MFU = {A100_MFU}, needs {total_day:.2e} days.')

def try_adamw():
    lr_list = [1e1, 1e2, 1e3]
    iter_num = 10
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    
    # 用于存储每个lr对应的loss历史
    loss_histories = []
    for lr in lr_list:
        opt = AdamW([weights], lr=lr)  # 注意这里lr参数要使用循环变量lr
        weights.data = 5 * torch.randn((10, 10))  # 每次换lr时重置weights
        loss_history = []
        for t in range(iter_num):  
            opt.zero_grad()
            loss = (weights**2).mean()
            loss_history.append(loss.cpu().item())
            loss.backward()
            opt.step()
        loss_histories.append(loss_history)

    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    # 绘制每个lr对应的图
    for i, lr in enumerate(lr_list):
        plt.figure(figsize=(8, 5))
        plt.plot(range(iter_num), loss_histories[i])
        plt.xlabel('迭代次数')
        plt.ylabel('Loss')
        plt.title(f'学习率为{lr}时的Loss变化')
        plt.grid(True)
        path = f'./fig/loss_lr_{lr}.png'
        plt.savefig(path)  # 保存图片，也可以用plt.show()直接显示
        # plt.show()
        plt.close()

def lr_cosine_schedule(
    t, 
    alpha_max,
    alpha_min,
    Tw,
    Tc
) -> Float:
    if t < Tw: alpha_t = t / Tw * alpha_max
    elif t > Tc: alpha_t = alpha_min
    else:
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - Tw) / (Tc - Tw) * math.pi)) * (alpha_max - alpha_min)
    return alpha_t

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float
) -> None:
    eps = 1e-6
    grads = []
    for p in parameters:
        if p.grad is None: continue
        grad  = p.grad.flatten() # shape like torch.Size([n])
        #print('grad dim:' ,grad.shape)
        grads.append(grad)
    if len(grad) == 0: return
    #print('all grads: ', grads) # a list consist of torch.Size([n1]), torch.Size([n1])...
    all_grads = torch.cat(grads) # shape like torch.Size([n1+n2+...])
    #print('all_grads:', all_grads.shape)
    all_norm = torch.norm(all_grads, p=2)
    if all_norm > max_l2_norm:
        scale_factor = max_l2_norm / (all_norm + eps)
        for p in parameters:
            if p.grad is not None: p.grad.mul_(scale_factor)

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
):
    # print(dataset)
    # print('ndarray的维度: ', dataset.ndim)
    # print('ndarray的形状: ', dataset.shape)
    # print('ndarray的元素数量: ', dataset.size)
    # print('ndarray中的数据类型: ', dataset.dtype)
    # print(type(dataset))
    max_start_point = len(dataset) - context_length - 1
    assert max_start_point > 0, 'max_start_point < 0'
    # 不能用Tensor()初始化，会默认是float32，导致后面不能做索引
    res1 = torch.empty(batch_size, context_length, dtype=torch.long, device=device)
    res2 = torch.empty(batch_size, context_length, dtype=torch.long, device=device)
    # 生成范围在 [0, max_start_point] 之间的随机整数
    for i in range(batch_size):
        start = random.randint(0, max_start_point)
        res1[i] = torch.tensor(dataset[start:start+context_length],dtype=torch.long,device=device)
        res2[i] = torch.tensor(dataset[start+1:start+context_length+1],dtype=torch.long, device=device)
        # 都是(batch, seq_len)形式 
    res = (res1, res2)
    return res

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    elapsed_time = 0.0
):
    model_dict = model.state_dict()
    opt_dict = optimizer.state_dict()
    to_save = {
        'model_params': model_dict,
        'optimizer_params': opt_dict,
        'iteration': iteration,
        'elapsed_time': elapsed_time
    }
    torch.save(to_save, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location=None,

):
    to_load = torch.load(src, map_location=map_location)
    model.load_state_dict(to_load['model_params'])
    if optimizer is not None:
        optimizer.load_state_dict(to_load['optimizer_params'])
    it = to_load['iteration']
    elapsed_time = to_load.get('elapsed_time', 0.0)
    return it, elapsed_time

def try_gradient_clipping():
    params = nn.ParameterList([
        nn.Parameter(torch.rand(4,5) * 2)
        for _ in range(2)
    ])
    for p in params:
        p.grad = torch.rand_like(p)
    gradient_clipping(params, 1)

def try_get_batch():
    array_b = np.arange(0, 100)
    get_batch(array_b,32,6,'cpu')

import os

def print_param_brief(params, to_text=True, path: str | None = None):
    # 关键修复：将生成器转换为列表，确保可重复迭代
    params_list = list(params)  # 生成器 → 列表，后续所有操作基于此列表
    
    # 1. 终端打印参数详情
    print("="*50)
    print("优化器中的可训练参数详情：")
    print("="*50)
    total_params = 0
    for name, param in params_list:  # 迭代列表（可重复）
        if param.requires_grad:
            shape = param.shape
            dtype = param.dtype
            num_params = param.numel()
            total_params += num_params
            print(f"参数名称: {name:30} | 形状: {shape} | 数据类型: {dtype} | 元素个数: {num_params:,}")
    print("="*50)
    print(f"可训练参数总数: {total_params:,}")
    print("="*50)

    # 2. 写入文件（若需要）
    if to_text and path is not None:
        file_path = os.path.join(path, 'params.txt')
        # 用 'w' 模式（覆盖写入），避免多次调用时内容重复（若需追加可保留 'a'）
        with open(file=file_path, mode='w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("优化器中的可训练参数详情：\n")
            f.write("="*50 + "\n")
            total_params_file = 0
            for name, param in params_list:  # 再次迭代列表（无数据损耗）
                if param.requires_grad:
                    shape = param.shape
                    dtype = param.dtype
                    num_params = param.numel()
                    total_params_file += num_params
                    f.write(f"参数名称: {name:30} | 形状: {shape} | 数据类型: {dtype} | 元素个数: {num_params:,}\n")
            f.write("="*50 + "\n")
            f.write(f"可训练参数总数: {total_params_file:,}\n")
            f.write("="*50 + "\n")


def log_training_process(text, log_path, print_to_console=True):
    """将训练过程日志写入文件，并可选输出到终端"""
    if print_to_console:
        print(text)
    with open(log_path, 'a') as f:
        f.write(text + '\n')

def save_training_state(
    state_dict_path, 
    hyperparams, 
    loss_history=None, 
    lr_history=None, 
    step_history=None
):
    """保存训练超参数和状态到字典"""
    state = {
        "hyperparams": hyperparams,
        "loss_history": loss_history or [],
        "lr_history": lr_history or [],
        "step_history": step_history or []
    }
    with open(state_dict_path, 'w') as f:
        json.dump(state, f, indent=2)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def training(
    token_path,
    vocab_size,
    context_length=4*64,  # 推荐配置
    d_model=128,        # 推荐配置
    num_layers=4,       # 推荐配置
    num_heads=8,        # 推荐配置（128/8=16，每个头维度合理）
    d_ff=4 * 128,       # 4*d_model标准配置
    rope_theta=10000,
    iter_num=10000,     # 推荐配置（足够训练）
    alpha_max=3e-4,     # 最大学习率（推荐）
    alpha_min=1e-5,     # 最小学习率（推荐）
    max_l2_norm=1.0,    # 梯度裁剪的最大L2范数（推荐）
    betas=(0.9, 0.999),  # 适配NLP的AdamW动量
    batch_size=16,      # 推荐配置
    checkpoint_path='.\\data\\checkpoints',
    resume = True,
    last_time: str = 'No last time',
    logger: Run | None = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Using device: {device}")


    # 初始化日志和状态字典
    Tw = math.floor(0.1 * iter_num)
    Tc = math.floor(0.8 * iter_num)
    checkpoint_period = math.floor(iter_num / 20)
    if resume and last_time != 'No last time':
        time_stamp = last_time
    else: time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    sub_folder = os.path.join(checkpoint_path, time_stamp)
    os.makedirs(sub_folder, exist_ok=True)
    
    # 日志文件路径
    log_path = os.path.join(sub_folder, 'training_process.txt')
    # 状态字典路径
    state_dict_path = os.path.join(sub_folder, 'training_state.json')
    
    # 初始化训练状态记录
    loss_history = []
    lr_history = []
    step_history = []
    
    # 记录超参数
    hyperparams = {
        "token_path": token_path,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta,
        "iter_num": iter_num,
        "alpha_max": alpha_max,
        "alpha_min": alpha_min,
        "max_l2_norm": max_l2_norm,
        "betas": betas,
        "batch_size": batch_size,
        "checkpoint_path": checkpoint_path,
        "resume": resume,
        "resume_path": last_time
    }
    
    # 初始化日志
    log_text = f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
    log_text += "="*50 + "\n"
    log_text += "Hyperparameters:\n"
    for k, v in hyperparams.items():
        log_text += f"  {k}: {v}\n"
    log_text += "="*50 + "\n"
    log_training_process(log_text, log_path)

    if logger is not None:
        # 同步超参数到 WandB（网页端可查看）
        logger.config.update(hyperparams)
        # 记录实验目录（方便关联本地文件）
        logger.config.update({"experiment_folder": sub_folder})
        log_training_process(f"WandB initialized: syncing to run {logger.id}", log_path)

    token = np.load(file=token_path, mmap_mode='r')
    
    # 初始化模型和优化器
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(device)
    opt = AdamW(
        params=model.parameters(),
        lr=alpha_max,
        betas=betas,
        eps=1e-8,
        weight_decay=0.01
    )
    
    # 打印参数详情
    # print_param_brief(model.named_parameters(), to_text=True, path=sub_folder)
    total_elapsed_time  = 0.0
    if resume:
        print('Resume training!')
        if last_time == 'No last time': 
            print('No last time!')
            return
        latest_ckpt = os.path.join(checkpoint_path,last_time,'training_latest.pt')
        t, total_elapsed_time  = load_checkpoint(latest_ckpt, model, opt, map_location=device)
        # 恢复训练状态（如果存在）
        if os.path.exists(state_dict_path):
            with open(state_dict_path, 'r') as f:
                state = json.load(f)
                loss_history = state.get("loss_history", [])
                lr_history = state.get("lr_history", [])
                step_history = state.get("step_history", [])
            if logger is not None:
                log_training_process(f"WandB resuming: loading {len(step_history)} historical steps", log_path)
    else:
        t = 0
    
    start_time = time.time()
    log_training_process('Start training!\n', log_path)
    
    while t < iter_num:
        x, y = get_batch(
            dataset=token,
            batch_size=batch_size,
            context_length=context_length,
            device=device
        )
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred_y = model(x)
            pred_y_flatten = rearrange(pred_y, 'batch seq_len vocab -> (batch seq_len) vocab')
            y_flatten = rearrange(y, 'batch seq_len -> (batch seq_len)')
            loss = cross_entropy(pred_y_flatten, y_flatten)
        scaler.scale(loss).backward()
        
        scaler.unscale_(opt)
        gradient_clipping(
            parameters=model.parameters(),
            max_l2_norm=max_l2_norm
        )
        
        current_lr = lr_cosine_schedule(
            t=t,
            alpha_max=alpha_max,
            alpha_min=alpha_min,
            Tw=Tw,
            Tc=Tc
        )
        for group in opt.param_groups:
            group['lr'] = current_lr
        
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        
        # 记录训练状态
        loss_history.append(loss.item())
        lr_history.append(current_lr)
        step_history.append(t)

        # 1. 计算核心时间指标
        current_time = time.time()
        elapsed_time = current_time - start_time + total_elapsed_time   # 已运行时间（秒）
        completed_steps = t + 1  # 已完成步数（t从0开始，t=0代表完成1步）
        
        # 2. 计算预估剩余时间（避免首次打印时除以0）
        if completed_steps == 0:
            estimated_remaining = 0  # 理论上不会触发，防止极端情况
        else:
            avg_step_time = elapsed_time / completed_steps  # 单步平均耗时（秒）
            remaining_steps = iter_num - completed_steps    # 剩余步数
            estimated_remaining = avg_step_time * remaining_steps  # 预估剩余时间（秒）
        
        # 3. 格式化当前时间、已运行时间、预估剩余时间
        current_time_str = time.strftime('%m-%d %H:%M:%S', time.localtime(current_time))
        elapsed_time_str = format_time(elapsed_time)
        estimated_remaining_str = format_time(estimated_remaining)
        
        # 4. 拼接日志文本（含当前时间、已运行时间、预估剩余时间）
        log_text = (
            f"[Iter {t:5d}] Loss: {loss.item():.4f} | Current LR: {current_lr:.6f} | "
            f"Now: {current_time_str} | Elapsed: {elapsed_time_str} | "
            f"Estimated Remaining: {estimated_remaining_str}"
        )
        print_to_console = True if t % checkpoint_period == 0 else False
        log_training_process(log_text, log_path,print_to_console=print_to_console)
        if logger is not None:
            try:
                # 同步核心指标（支持网页端可视化）
                logger.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/total_elapsed_time_s": elapsed_time,
                    "train/avg_step_time_ms": avg_step_time * 1000,  # 毫秒级更直观
                })  # step 与迭代步数对齐，确保可视化连贯
                
            except Exception as e:
                # 网络异常时仅记录本地日志，不中断训练
                error_msg = f"WandB sync failed (will retry later): {str(e)}"
                print(error_msg)
                log_training_process(error_msg, log_path,print_to_console=False)
        # 定期保存日志和状态
        if t % checkpoint_period == 0 or t == iter_num - 1: 
            # 保存状态字典
            save_training_state(
                state_dict_path,
                hyperparams,
                loss_history,
                lr_history,
                step_history
            )
            # 保存模型 checkpoint
            latest_ckpt = os.path.join(sub_folder, 'training_latest.pt')
            iter_ckpt = os.path.join(sub_folder, f'training_iter_{t}.pt')
            save_checkpoint(model, opt, t, latest_ckpt, elapsed_time)
            save_checkpoint(model, opt, t, iter_ckpt, elapsed_time)
        t += 1
    
    end_time = time.time()
    log_text = f"Done! Time: {(end_time-start_time)/3600:.2f} hours\n"
    log_text += f"Ended at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    log_training_process(log_text, log_path)
    
    # 最终保存状态字典
    save_training_state(
        state_dict_path,
        hyperparams,
        loss_history,
        lr_history,
        step_history
    )


def try_training():
    special_tokens = ["<|endoftext|>"]
    OpenWebText_vocab_path = './data/OpenWebText_vocab_32000.pkl'
    OpenWebText_merges_path = './data/OpenWebText_merges_32000.pkl'
    TinyStories_vocab_path = './data/tinystories_vocab_10000.pkl'
    TinyStories_merges_path = './data/tinystories_merges_10000.pkl'
    with open(OpenWebText_vocab_path, "rb") as f:
        owt_vocab = pickle.load(f)
    with open(OpenWebText_merges_path, "rb") as f:
        owt_merges = pickle.load(f)
    with open(TinyStories_vocab_path, "rb") as f:
        tiny_vocab = pickle.load(f)
    with open(TinyStories_merges_path, "rb") as f:
        tiny_merges = pickle.load(f)
    owt_tokenizer = Tokenizer(vocab=owt_vocab, merges=owt_merges,special_tokens=special_tokens,num_processes=4)
    tiny_tokenizer = Tokenizer(vocab=tiny_vocab, merges=tiny_merges,special_tokens=special_tokens,num_processes=4)
    batch_size = 32
    iter_num = 40000
    content_length = 256
    total_tokens_processed = batch_size * iter_num * content_length # should be close to 327,680,000
    print('total tokens processed:', total_tokens_processed)
    training(
        token_path='.\\data\\tinystories_train_token_merged.npy',
        vocab_size=tiny_tokenizer.vocab_len,
        context_length=content_length,
        d_ff=1344,
        d_model=512,
        num_layers=4,
        num_heads=16,
        batch_size=batch_size,
        checkpoint_path = '.\\data\\checkpoints',
        iter_num=iter_num,
        resume=False
    )

def try_training_on_minitext():
    special_tokens = ["<|endoftext|>"]
    TinyStories_vocab_path = '../data/tinystories_vocab_10000.pkl'
    TinyStories_merges_path = '../data/tinystories_merges_10000.pkl'
    with open(TinyStories_vocab_path, "rb") as f:
        tiny_vocab = pickle.load(f)
    with open(TinyStories_merges_path, "rb") as f:
        tiny_merges = pickle.load(f)
    tiny_tokenizer = Tokenizer(vocab=tiny_vocab, merges=tiny_merges,special_tokens=special_tokens,num_processes=4)

    token_merged_out_path = '../data/self-test_merged.npy'

    batch_size = 16
    iter_num = 10000
    content_length = 256
    
    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="leoself",
        # Set the wandb project where this run will be logged.
        project="test",
        # Track hyperparameters and run metadata.
        config={
            "content length": content_length,
            'batch size': batch_size,
            "epochs": iter_num,
        },
        settings=wandb.Settings(init_timeout=40),
    )
    # print(type(run))
    total_tokens_processed = batch_size * iter_num * content_length # should be close to 327,680,000
    print('total tokens processed:', total_tokens_processed)
    training(
        token_path=token_merged_out_path,
        vocab_size=tiny_tokenizer.vocab_len,
        context_length=content_length,
        d_ff=1344,
        d_model=512,
        num_layers=4,
        num_heads=16,
        alpha_max=3e-4,
        alpha_min=1e-5,
        batch_size=batch_size,
        checkpoint_path = '../data/checkpoints',
        iter_num=iter_num,
        resume=False,
        logger=run
    )

def try_wandb():
    # Start a new wandb run to track this script.
    # os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="leoself",
        # Set the wandb project where this run will be logged.
        project="test",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "epochs": 10,
        },
        settings=wandb.Settings(init_timeout=40),
    )
    # Simulate training.
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        # acc = 1 - 2**-epoch - random.random() / epoch - offset
        # loss = 2**-epoch + random.random() / epoch + offset
        acc = epoch + 0.1
        print(acc)
        # Log metrics to wandb.
        run.log({"acc": acc})

    # Finish the run and upload any remaining data.
    run.finish()

if __name__ == '__main__':
    try_training_on_minitext()
    #try_wandb()





