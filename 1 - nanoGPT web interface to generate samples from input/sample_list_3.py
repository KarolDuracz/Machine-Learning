
"""
Interactive sampling from a trained nanoGPT model.

Behavior:
- Read a prompt (multi-line; end with an empty line).
- For that prompt, produce LINES lines.
- Each line contains SAMPLES_PER_LINE independent samples (each TOKENS_PER_SAMPLE tokens long),
  separated by the literal "<eot>".
- Each sample is generated from the original prompt (no chaining between samples).
"""
import os
import pickle
from contextlib import nullcontext
import sys
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# CONFIG: change these to control behavior
init_from = 'resume'       # 'resume' to load from out_dir, or 'gpt2-*' variants
out_dir = 'out'            # used when init_from == 'resume'

LINES = 20                 # number of output lines to produce per prompt
SAMPLES_PER_LINE = 20      # number of independent samples per line (10,20,30,...)
TOKENS_PER_SAMPLE = 30     # how many new tokens each sample generates
SEPARATOR = "<eot>"        # literal separator between samples on the same line
# -----------------------------------------------------------------------------

temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'  # 'cpu' or 'cuda' (or 'cuda:0', etc.)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())  # allow overrides from CLI/config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if 'cuda' in device:
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError("init_from must be 'resume' or start with 'gpt2'")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# load encoding/metadata if present
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

def read_multiline_prompt():
    """Read prompt lines until an empty line. Return None to quit."""
    print("Enter prompt (finish with an empty line). Submit an empty prompt to quit.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            return None
        if line == "":
            break
        lines.append(line)
    if len(lines) == 0:
        return None
    return "\n".join(lines)

# Interactive loop
try:
    while True:
        start = read_multiline_prompt()
        if start is None:
            print("No prompt — exiting.")
            break

        # encode prompt and create tensor (single example)
        start_ids = encode(start)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # shape (1, prompt_len)
        prompt_len = x.shape[1]

        # generate: for each of LINES, produce SAMPLES_PER_LINE independent samples
        with torch.no_grad():
            with ctx:
                for line_idx in range(LINES):
                    out_texts = []
                    # generate each sample from the original prompt (no chaining)
                    for sample_idx in range(SAMPLES_PER_LINE):
                        # call generate on the single prompt; model.generate returns shape (1, prompt_len + TOKENS_PER_SAMPLE)
                        y = model.generate(x, TOKENS_PER_SAMPLE, temperature=temperature, top_k=top_k)
                        full_ids = y[0].tolist()
                        new_ids = full_ids[prompt_len:]  # only newly generated tokens
                        out_texts.append(decode(new_ids))
                    # join and print samples for this line, separated by the literal "<eot>"
                    print(SEPARATOR.join(out_texts))
                print("---------------")  # divider after all LINES for this prompt

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting.")
