
"""
Interactive sampling: type a prompt on one line, script fills the rest of the line.
Each printed completion is: <your prompt><generated continuation>.
Completions on the same printed line are separated by the literal "<eot>".
"""
import os
import pickle
from contextlib import nullcontext
import sys
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# CONFIG
init_from = 'resume'       # 'resume' or 'gpt2-*'
out_dir = 'out'

LINES = 20                 # number of printed output lines per prompt
SAMPLES_PER_LINE = 20      # how many independent completions per printed line
TOKENS_PER_SAMPLE = 30     # how many tokens to generate per completion
SEPARATOR = "<eot>"        # literal separator between completions on the same printed line
# -----------------------------------------------------------------------------

temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'  # or 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())  # allow CLI/config overrides
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

# load encoder/decoder (meta.pkl if available, otherwise gpt2)
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

def read_single_line_prompt():
    """Read a prompt from the user (single line). Return None to quit."""
    try:
        s = input("Prompt (single line, empty to quit): ")
    except EOFError:
        return None
    if s is None or s == "":
        return None
    return s

# interactive loop: read one-line prompt, produce LINES lines each with SAMPLES_PER_LINE completions
try:
    while True:
        prompt = read_single_line_prompt()
        if prompt is None:
            print("Exiting.")
            break

        # encode the prompt and create single-example tensor
        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # shape (1, prompt_len)
        prompt_len = x.shape[1]

        # For each printed line, generate SAMPLES_PER_LINE independent completions from the original prompt.
        # We use batched generation per printed line for speed: repeat the prompt SAMPLES_PER_LINE times.
        with torch.no_grad():
            with ctx:
                for line_idx in range(LINES):
                    # batch the same prompt SAMPLES_PER_LINE times
                    xb = x.repeat(SAMPLES_PER_LINE, 1)  # shape (SAMPLES_PER_LINE, prompt_len)
                    y = model.generate(xb, TOKENS_PER_SAMPLE, temperature=temperature, top_k=top_k)
                    # y shape: (SAMPLES_PER_LINE, prompt_len + TOKENS_PER_SAMPLE)
                    completions = []
                    for i in range(SAMPLES_PER_LINE):
                        full_ids = y[i].tolist()
                        new_ids = full_ids[prompt_len:]  # newly generated tokens
                        completion = decode(new_ids)
                        # print full line sample as prompt + completion
                        completions.append(prompt + completion)
                    # join samples on same printed line using the literal "<eot>"
                    print(SEPARATOR.join(completions))
                print("---------------")  # divider after all lines for this prompt

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting.")
