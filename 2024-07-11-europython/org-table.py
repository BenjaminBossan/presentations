"""Run memory estimation on multiple models and output as an org table

Requires to install accelerate and tabulate:

```bash
python -m pip install accelerate tabulate
```

Run the script like so:

```bash
# run with the default models
python org-table.py
# specify models to check, LoRA rank, dtype
python org-table.py --model_ids "meta-llama/Meta-Llama-3-8B,google/gemma-2-9b" --rank 32 --dtype int4
# change output format to GitHub-flavored markdown
python org-table.py --tablefmt github
```

"""

import argparse

from tabulate import tabulate

from utils import main


DEFAULTS = """meta-llama/Meta-Llama-3-8B,
meta-llama/Meta-Llama-3-70B,
mistralai/Mistral-7B-v0.3,
Qwen/Qwen2-1.5B,
Qwen/Qwen2-72B,
google/gemma-2-9b,
google/gemma-2-27b"""


def null(*args, **kwargs):
    pass


def create_org_table(model_ids, rank, dtype, tablefmt):
    results = []
    for model_id in model_ids.split(","):
        model_id = model_id.strip()
        result = main(model_id, rank=rank, dtype=dtype, sink=null)
        memory_ft = result["memory required for full fine-tuning"]
        memory_lora = result["memory required for LoRA fine-tuning"]
        results.append([model_id, memory_ft, memory_lora])

    column_names = ["Model", f"Full fine-tuning ({dtype})", f"LoRA fine-tuning (rank {rank})"]
    print(tabulate(results, headers=column_names, tablefmt=tablefmt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ids", type=str, help="Model names, separated by comma (on Hugging Face)", default=DEFAULTS)
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA adapter")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type, one of float32, float16, bfloat16, int8, int4")
    parser.add_argument("--tablefmt", type=str, default="orgtbl", help="Table format for tabulate (default: orgtbl)")
    args = parser.parse_args()
    create_org_table(args.model_ids, rank=args.rank, dtype=args.dtype, tablefmt=args.tablefmt)
