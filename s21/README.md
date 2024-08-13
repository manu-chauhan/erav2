# GPT Training from scratch

- Training a GPT model from scratch on Shakespeare data

- Multiple optimizations:
  - flash attention from Pytorch
  - torch.compile (tried default, max-autotune)
  - power of two for vocab size 50257 to 50304
  - Optimizer
  - grad clipping 
  - cosine LR
  - gradient accumulation
  
- Ran training for 5000 steps on `input.txt` -> did not converge to < 0.09

- Final run via DDP on cloud with 2 x GPUs with Gradient accumulation, loss went below 0.09 around epoch 394

  