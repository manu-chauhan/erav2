import torch
from transformers import GPT2Tokenizer
import gradio as gr
import tiktoken
import model_file
from dataclasses import dataclass
import time
import os
import torch.nn.functional as F

num_return_sequences = 1
max_length = 200


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = tiktoken.get_encoding("gpt2")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

device = torch.device(device)

try:
    model = model_file.get_model().to(device)
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "model_00350.pt"), map_location=device)
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

examples = [
    "Who are you?",
    "Write a Shakespeare short poem.",
    "Tell me a joke.",
    "What is the meaning of life?",
]


def chat_fn(message, history):
    # Tokenize
    print(f"message: {message}")
    tokens = tokenizer.encode(message)
    tokens = torch.tensor(tokens, dtype=torch.int32)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    while x.size(1) < max_length:
        # forward pass through model to get logits
        with torch.no_grad():
            logits = model(x)[0]  # batch_size, T, vocab_size
            logits = logits[:, -1, :]  # get last position logits B, vocab_size

            # calculate probabilities
            probs = F.softmax(logits, dim=-1)

            # doing topk here, HF defafult is 50
            # topk is (5, 50), top_indices is (5, 50) too
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            # sampling a token from topk
            ix = torch.multinomial(input=topk_probs, num_samples=1)  # (B, 1)

            # gather corresponding indices
            xcol = torch.gather(input=topk_indices, dim=-1, index=ix)
            # append to the seq
            x = torch.cat([x, xcol], dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)

        yield decoded + "\n"


gr.ChatInterface(chat_fn, examples=examples,
                 title="GPT2 trained from scratch on Shakespeare dataset").launch()

