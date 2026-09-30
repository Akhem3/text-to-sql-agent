import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
filename = "qwen2.5-coder-14b-instruct-q4_k_m.gguf"

# On charge le tokenizer depuis le repo non-GGUF officiel qui possède tous les fichiers propres
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-14B-Instruct", trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    gguf_file=filename,
    device_map="auto",
    dtype=torch.bfloat16,
)