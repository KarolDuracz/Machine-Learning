# app.py
import os
import pickle
from contextlib import nullcontext
import argparse
import torch
import tiktoken
from flask import Flask, render_template, request, jsonify

# import your GPT model & config
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# defaults (you can override via CLI)
DEFAULT_OUT_DIR = 'out'
DEFAULT_INIT_FROM = 'resume'     # 'resume' or 'gpt2-*'
# sampling defaults (can be overridden from the web UI)
DEFAULT_LINES = 10
DEFAULT_SAMPLES_PER_LINE = 5
DEFAULT_TOKENS_PER_SAMPLE = 30
SEPARATOR = "<eot>"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 200
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
parser.add_argument("--init_from", type=str, default=DEFAULT_INIT_FROM)
parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
parser.add_argument("--no_compile", action="store_true", help="don't torch.compile even if enabled")
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

out_dir = args.out_dir
init_from = args.init_from
device_arg = args.device
device = torch.device("cpu" if "cpu" in device_arg else "cuda")
use_compile = False if args.no_compile else False  # default off; set True if you want compile

# Model loading (adapted from your script)
print("Loading model... (this may take a while)")
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError("init_from must be 'resume' or start with 'gpt2'")

model.eval()
model.to(device)
if use_compile:
    try:
        model = torch.compile(model)
    except Exception as e:
        print("torch.compile failed, continuing without compile:", e)

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
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# prepare no-op autocast context like your original script
device_type = 'cuda' if device.type == 'cuda' else 'cpu'
dtype = 'float16' if (device.type == 'cuda' and torch.cuda.is_available()) else 'float32'
ptdtype = {'float32': torch.float32, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print("Model loaded and ready.")

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    JSON input:
    {
      "prompt": "Hello",
      "lines": 10,
      "samples": 5,
      "tokens": 30,
      "temperature": 0.8,
      "top_k": 200
    }
    JSON output:
    {
      "results": [ "line0joined-with-<eot>-samples", "line1...", ... ]
    }
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if prompt is None or prompt.strip() == "":
        return jsonify({"error": "empty prompt"}), 400

    lines = int(data.get("lines", DEFAULT_LINES))
    samples = int(data.get("samples", DEFAULT_SAMPLES_PER_LINE))
    tokens = int(data.get("tokens", DEFAULT_TOKENS_PER_SAMPLE))
    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    top_k = int(data.get("top_k", DEFAULT_TOP_K))

    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    prompt_len = x.shape[1]

    completions_all = []
    try:
        with torch.no_grad():
            with ctx:
                # generate line-by-line, batching the samples per line for speed
                for _ in range(lines):
                    xb = x.repeat(samples, 1)  # batch the same prompt
                    y = model.generate(xb, tokens, temperature=temperature, top_k=top_k)
                    for i in range(samples):
                        full_ids = y[i].tolist()
                        new_ids = full_ids[prompt_len:]
                        completion = decode(new_ids)
                        completions_all.append(prompt + completion)
    except Exception as e:
        return jsonify({"error": f"generation failed: {str(e)}"}), 500

    # turn flat completions_all into lines of joined samples
    results = []
    idx = 0
    for _ in range(lines):
        row = SEPARATOR.join(completions_all[idx:idx + samples])
        results.append(row)
        idx += samples

    return jsonify({"results": results})

if __name__ == "__main__":
    # Avoid Flask reloader accidentally re-loading models twice
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)