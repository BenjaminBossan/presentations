"""Run memory estimation on multiple models and output as an org table

```bash
# run with the default models
python org-table.py
# specify models to check, LoRA rank, dtype
python org-table.py --model_ids "meta-llama/Meta-Llama-3-8B,google/gemma-2-9b" --rank 32 --dtype int4
```

"""

import argparse

from utils import main


def null(*args, **kwargs):
    pass


def print_org_table(column_names, data):
    header = "|" + "|".join(column_names) + "|"
    print(header)
    divider = "|" + "+".join("-" for _ in column_names) + "|"
    print(divider)
    for row in data:
        row_str = "|" + "|".join(str(cell) for cell in row) + "|"
        print(row_str)


def create_org_table(model_ids, rank, dtype):
    results = []
    for model_id in model_ids.split(","):
        model_id = model_id.strip()
        result = main(model_id, rank=rank, dtype=dtype, sink=null)
        memory_ft = result["memory required for full fine-tuning"]
        memory_lora = result["memory required for LoRA fine-tuning"]
        results.append([model_id, memory_ft, memory_lora])

    column_names = ["Model", f"Full fine-tuning ({dtype})", f"LoRA fine-tuning (rank {rank})"]
    print_org_table(column_names, results)


DEFAULTS = """meta-llama/Meta-Llama-3-8B,
meta-llama/Meta-Llama-3-70B,
mistralai/Mistral-7B-v0.3,
Qwen/Qwen2-1.5B,
Qwen/Qwen2-72B,
google/gemma-2-9b,
google/gemma-2-27b"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ids", type=str, help="Model names, separated by comma (on Hugging Face)", default=DEFAULTS)
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA adapter")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type, one of float32, float16, bfloat16, int8, int4")
    args = parser.parse_args()
    create_org_table(args.model_ids, rank=args.rank, dtype=args.dtype)
