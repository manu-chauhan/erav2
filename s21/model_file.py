import torch
import tiktoken
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # this is max sequence len
    vocab_size: int = 50304  # 50257 # total vocab including 256 bytes + 1 special token (<|endoftext|>) and 1000-257 BPE merges
    n_layer: int = 12  # number of layers
    n_head: int = 12  # total number of attention heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_head = config.n_head
        n_embd = config.n_embd

        assert n_embd % n_head == 0

        # query, key, value prjections all combined
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)

        # output projection, after `v` is already multiplied with attention_scores
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.c_proj.NANOGPT_SCALE_INIT = 1

        block_size = config.block_size

        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence_len, embedding_dim (n_embd)
        # total dim = n_head * head_size
        # example GPT2 has 12 heads with each hs = 64 thus C= 12*64 = 768

        qkv = self.c_attn(x)  # get combined qkv matix B, T, n_embd * 3(768*3=2304)

        q, k, v = qkv.split(self.n_embd, dim=2)  # each item gets n_embd size, dimension against two

        # b, seq, n_embd -> b, seq, n_heads, head_size -> b, n_heads, seq_len, head_size
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # final-> bs, n_heads, seq_len, mini-n_head_embd

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # # print(f"shape of q: {q.shape}... shape of k : {k.shape}")

        # attn = (q @ k.transpose(-2, -1))/(math.sqrt(k.shape[-1]))

        # # apply masked fill at places where mask ==0, remember tril is lower triangle
        # attn = attn.masked_fill(mask = self.bias[ : , : , :T, :T] == 0, value=float('-inf'))

        # attn = F.softmax(attn, dim=-1)

        # y = attn @ v # B, n_heads, T/seq, T @ B, n_heads, T/Seq, head_size) -> B, n_heads, T, head_size

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention

        # transpose y to merge all n_heads. B, n_heads, T, head_size -> transpose B, T, n_heads, head_size -> view B, T, Channel_size/n_emb 768 
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # out projection, B, T, C -> B, T, C
        y = self.c_proj(y)

        return y

    def generate(self, prompt):
        if not isinstance(prompt, str) or len(prompt) == 0:
            return "Say something!"


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd)
            ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()  # batch , seq_len

        # check if incoming seq_len of idx is within limits
        assert T <= self.config.block_size, f"Cannot proceed as your Sequence len : {T} is more than {self.config.block_size}"

        # forward for token and position encodings
        # shape (T)
        pos = torch.arange(0, T, dtype=torch.int32, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # position embds of shape (T, n_embd)
        token_emb = self.transformer.wte(idx)  # token embds of shape (Batch, T/seq_len, n_embd)

        x = pos_emb + token_emb

        # now forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # pass through final layernorm
        x = self.transformer.ln_f(x)

        # pass through final LM_HEAD
        logits = self.lm_head(x)  # shape (Batch_size, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded len : {len(self.tokens)}')
        # print(f'1 epoch = {len(self.tokens)//(B*T)} batches ')
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + (B * T) + 1]
        y = buf[1:].view(B, T)
        x = buf[:-1].view(B, T)

        self.current_position += (B * T * self.num_processes)

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


def get_model():
    model = GPT(GPTConfig())
    return model

# cuda = torch.cuda.is_available()
# torch.set_float32_matmul_precision('high')

# max_lr = 6e-4
# min_lr = 0.1 * max_lr
# warmup_steps = 10
# max_steps = 5000

# def get_lr(iteration):
#     if iteration < warmup_steps:
#         return max_lr * (iteration + 1) / warmup_steps
#     if iteration > max_steps:
#         return min_lr

#     decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)

#     assert 0<= decay_ratio <= 1

#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
#     return min_lr + coeff * (max_lr - min_lr)


# model = GPT(GPTConfig()).to(device=device)

# model = torch.compile(model, mode='default')

# if ddp:
#     print("\n\n====================================\nDDP")
#     model = DDP(module=model,device_ids=[ddp_local_rank])

# raw_model = model.module if ddp else model

# # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
# optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)


# total_batch_size = 524288

# B = 16
# T = 1024

# assert total_batch_size % (B * T * ddp_world_size) == 0, "just to make sure total batch size is divisible by B*T"

# grad_accumulation_steps = total_batch_size // (B * T * ddp_world_size)

# if master_process:
#     print(f"\nGradient accumulation steps needed with B: {B} and T: {T} for total batch size: {total_batch_size} = {grad_accumulation_steps}")
#     print(f"total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

# # torch.cuda.amp.autocast(enabled=True)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)

# start= time.time()

# for step in range(max_steps):
#     t0 = time.time()
#     optimizer.zero_grad()

#     loss_mini = 0.0
#     for micro_step in range(grad_accumulation_steps):
#         x, y = train_loader.next_batch()
#         x, y = x.to(device=device), y.to(device)
#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             logits, loss = model(x, y)
#             # if i == 0:
#             #     assert logits.dtype == torch.bfloat16
#             #     assert loss.dtype == torch.float32
#             #     assert model.transformer.wte.weight.dtype == torch.float32

#         loss = loss/grad_accumulation_steps
#         loss_mini += loss.detach()
#         if ddp:
#             model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)
#         loss.backward()
#     if ddp:
#         dist.all_reduce(loss_mini, op=dist.ReduceOp.AVG)
#     if master_process and step%50==0 and step > 100:
#             print(f"saving at: {step}")            
#             checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
#             checkpoint = {
#                 'model': raw_model.state_dict(),
#                 'config': raw_model.config,
#                 'step': step
#             }
#             torch.save(checkpoint, checkpoint_path)
#     # grad clip
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#     lr = get_lr(step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr 

#     optimizer.step()
#     torch.cuda.synchronize()

#     t1 = time.time() 
#     dt = (t1 - t0) 
#     tokens_per_sec = (train_loader.B * train_loader.T * grad_accumulation_steps * ddp_world_size) / (dt)
#     if master_process:
#         # print happens via CPU, hence wait (synchronize GPU)
#         print(f'step : {step+1} | loss: {loss_mini.item()} | lr: {lr:.7f} | dt: {dt* 1000:.2f} ms | tokens/sec: {tokens_per_sec:_.6f} | norm: {norm:.2f}')


# end = time.time()
# print("final loss: ", loss*grad_accumulation_steps)
# print(f"total time: {end - start} seconds")
# torch.save(model.state_dict(), "5k-run-new-DDP.pt")

# if ddp:
#     destroy_process_group()
