# 在 Python 中，@dataclass 是一个装饰器，用于简化类的定义，特别是那些主要用于存储数据的类
from dataclasses import dataclass 
import torch.nn as nn
import torch 
from torch.nn import functional as F
import inspect
import os
import math
import time
import tiktoken
#from hellaswag import render_example, iterate_examples # hellaswag 是一个特定的数据集模块，通常用于评估语言模型的常识推理能力

@dataclass
class GPTConfig: #基本配置参数类
    block_size:int = 1024  # 最大序列长度
    vocab_size:int = 50257   # 50000个字节对编码（BPE）合并后的token，加上额外的256个字节级别的token，以及一个特殊的 <|endoftext|>标记，用来表示文本的结束
    n_layer:int = 12  # 模型中Transformer层的数量
    n_head:int = 12  # 每个Transformer层中多头注意力机制的头数
    n_embed:int = 768  # 词嵌入的维度，默认值为768，这个维度决定了每个token会被映射到一个多少维的空间中

class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        # 确保嵌入维度 n_embed 能够被头数 n_head 整除。这是为了确保每个注意力头可以均匀分配到相同的嵌入维度
        assert config.n_embed % config.n_head == 0
        # 定义一个线性层 c_attn，用于计算所有头的键（key）、查询（query）和值（value）。输入是嵌入维度 n_embed，输出是三倍的嵌入维度 3 * config.n_embed，这是因为我们要计算三个不同的向量集（Q、K、V）
        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed)

        # 输出投影
        #定义另一个线性层 c_proj，用于将注意力机制的输出投影回原始的嵌入维度 n_embed
        self.c_proj = nn.Linear(config.n_embed,config.n_embed)
        # 设置一个属性 NANOGPT_SCALE_INIT 为1，这可能用于特定的权重初始化策略
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # 保存配置中的头数 n_head 和嵌入维度 n_embed 为类的属性，用于后续的计算
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    # 前向传播方法
    def forward(self, x):
        # 获取输入张量 x 的形状，并将其分解为批量大小 B、序列长度 T 和嵌入维度 C
        B,T,C = x.size()

        # 使用 c_attn 层计算 Q、K、V 向量，并将它们拆分成三个张量 q、k 和 v，每个张量的形状为 (B, T, C)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x) # qkv 的形状将是 (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embed, dim=2)  # 将 qkv 沿着维度 dim=2 分割成三个大小为 self.n_embed 的张量

        # 重新排列 Q、K、V 张量的维度，使其形状变为 (B, nh, T, hs)，其中 nh 是头数，hs 是每个头的大小。这一步是为了让每个头都能够独立地计算注意力。
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float("inf") )
        # att = F.softmax(att,dim = -1)
        # y = att @ v # (B,nh,T,T) x (B,nh,T,hs)
        # 使用 F.scaled_dot_product_attention 函数计算缩放点积注意力。这里的 is_causal=True 表示这是一个因果注意力机制，即每个位置只能看到前面的位置，而不能看到后面的位置，这是为了保持模型的自回归特性
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        # 将注意力机制的结果重新组合成原来的形状 (B, T, C)，即将所有头的输出拼接在一起
        # y.transpose(1, 2) 会交换 y 的第1维度和第2维度，从而将形状从 (B, nh, T, hs) 变为 (B, T, nh, hs)
        # contiguous 方法确保张量在内存中是连续存储的。这对于后续的操作（如 view）是非常重要的，因为 view 需要张量在内存中是连续的。(如果不是连续存储的，会返回一个新的连续存储的)
        # view 方法用于重塑张量的形状
        # 通过这种方式，我们可以将所有头的输出重新组装在一起，形成一个统一的输出张量，以便后续的处理
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # 使用 c_proj 层对注意力机制的结果进行投影，回到原始的嵌入维度 C
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed,4*config.n_embed)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
            

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    def forward(self,x):
         x = x + self.attn(self.ln_1(x))
         x = x + self.mlp(self.ln_2(x))
         return x

class GPT(nn.Module): #定义了一个继承自 torch.nn.Module 的新类 GPT。这意味着 GPT 类将是一个PyTorch模型类

    def __init__(self,config): #定义了 GPT 类的构造函数，接受一个 config 参数，该参数应该是前面定义的 GPTConfig 类的一个实例
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),  # wte: 单词嵌入（word token embedding），是一个 nn.Embedding 层，用于将词汇表中的单词映射到一个固定维度的向量空间。
            wpe = nn.Embedding(config.block_size,config.n_embed), # wpe: 位置嵌入（position embedding），也是一个 nn.Embedding 层，用于给每个位置赋予一个特定的向量，帮助模型理解序列中的位置信息。
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # h: 一个 nn.ModuleList，包含了多个 Block 类型的对象，每个 Block 对应一个Transformer层。
            ln_f = nn.LayerNorm(config.n_embed) # ln_f: 最终的层归一化（Layer Normalization）层，用于对最后一层的输出进行归一化。
        ))
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False) # 分类器：将Transformer的最后一层输出映射回词汇表大小的维度，用于预测下一个单词的概率分布
        
        # 实施权重共享方案，使语言模型头部的权重与单词嵌入层的权重相同
        self.transformer.wte.weight = self.lm_head.weight  
        
        #初始化参数
        self.apply(self._init_weights) #调用 apply 函数，对模型的所有参数应用 _init_weights 方法进行初始化

    def _init_weights(self,module):
        # 根据模块类型初始化权重：
        #     如果模块是 nn.Linear 类型，则使用正态分布初始化权重，并且如果模块具有 NANOGPT_SCALE_INIT 属性，则调整标准差。
        #     如果模块是 nn.Embedding 类型，则同样使用正态分布初始化权重。
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    # 前向传播逻辑，接受输入 idx（一个包含序列索引的张量）和可选的目标 targets
    def forward(self, idx, targets=None):
        # 获取输入的形状，并确保序列长度 T 不超过配置中定义的最大序列长度 block_size
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward the sequence of length {T},block size is only {self.config.bloak_size}"

        # 计算位置嵌入和 token嵌入，并将它们相加得到输入的最终嵌入表示
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb

        # 依次通过每个Transformer块对输入进行处理
        for block in self.transformer.h:
            x = block(x)
        
        # 对输出进行最终的层归一化，并通过语言模型头部(分类器)得到每个位置上的词汇预测概率分布
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 如果提供了目标张量，则计算交叉熵损失作为模型的损失函数
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # 计算的是B*T元素的平均损失
        
        return logits, loss
    
    # 用于从预训练模型加载权重
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        
        # 确保了 model_type 是一个有效的选项，并从 transformers 库中加载预训练模型
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 根据不同的预训练模型类型设置相应的配置参数
        # n_layer, n_head and n_embed are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # 创建一个新的 GPT 模型实例，并获取其状态字典
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        
        # 从Hugging Face库加载对应的预训练模型，并获取其状态字典
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 遍历预训练模型的状态字典，确保所有参数名称和形状一致，并复制权重。对于某些特定的权重矩阵，需要转置后再复制
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # 返回加载了预训练权重的新模型实例
        return model 
    
    # 配置优化器
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # 筛选出需要梯度更新的所有参数，并根据参数维度决定是否应用权重衰减
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # 根据参数列表创建优化器组，并打印参数数量信息（如果是在主进程中）
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 根据设备类型选择是否使用融合版本的AdamW优化器，并创建优化器实例
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)   # 加载UN16 numpy文件
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)  # 转换为torch.long
    return ptt

# 分布式数据加载器
# 改写数据加载器，使每个进程都能获取属于自己的数据块，使其处理数据的不同部分
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # with open('/amax/xuhu/project/Build-Nanopgpt/input.txt', 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)

        # get the shard filenames
        data_root = "/amax/xuhu/project/Build-Nanopgpt/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

        # self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        #self.current_position = self.B *self.T * self.process_rank
    
    # 重置数据加载器
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device(自动检测设备)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # 如果 CUDA 不可用，则检查是否有 MPS 设备可用。MPS 主要用于 Apple 的硬件，特别是 macOS 上的设备。通过 hasattr(torch.backends, "mps") 检查 torch.backends 模块中是否存在 mps 属性，然后调用 torch.backends.mps.is_available() 来确认 MPS 是否可用。如果 MPS 可用，则将 device 设置为 "mps"
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# 启用TF32
torch.set_float32_matmul_precision('high')

# 梯度累计
total_batch_size = 524288 # 2**19, ~0.5M = 50万, in number of tokens
B = 16 # micro batch size：控制在一次前向后向中处理多少token和行
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)  # 524288//16*1024*8=4

# 每个进程都会打印一遍，总共八遍，为了保持只打印一次，我们只在主进程上打印
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# print("I'm GPU ",ddp_rank)
# print("Bye!")
# import sys; sys.exit(0)  #退出


train_loader = DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split="train" )

# # get a data batch
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()
# data = text[:1000] # first 1,000 characters
# tokens = enc.encode(data)
# B,T = 4,32
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)

# create model
model = GPT(GPTConfig(vocab_size=50304))  # 修改为2的幂形式，可以加快计算
model.to(device)
# 它允许用户通过简单的 API 调用来启用 Just-In-Time (JIT) 编译器和优化器，从而提高模型的执行效率
model = torch.compile(model)

#将模型包装到 DDP 容器中
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
# 375M =375*1,000,000= 375 * 10^6;375*10^6/2^19 = 715  
warmup_steps = 715
# 每一步执行2**19=524288个tokens, ~0.5M，想要执行 10B=10e9=10^10 tokens，10^10/2^19= 19073
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)
        # 使用BF16，只会前向传播在一些矩阵运算时使用，例如layerNorm、Softmax等仍然使用TF32
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps -1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG) # 每个rank都将拥有存储在所有rank的梯度的平均值
    
    #对模型的梯度进行裁剪，以防止梯度爆炸（gradient explosion）
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step() #更新参数

    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0  # time difference in miliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f}| lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | token/sec:{tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)  #退出

# PREFIX TOKEN

