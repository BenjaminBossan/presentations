import argparse
import json
import sys
import warnings
from collections import defaultdict

from accelerate.commands.estimate import create_empty_model
from accelerate.utils.other import convert_bytes

# suppress all warnings
warnings.filterwarnings("ignore")

dtype_to_bytes_linear = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
# no quantization if not Linear, assume 16 bit instead
dtype_to_bytes_other = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 2, "int4": 2}
LORA = "lora"


def get_num_params(param):  # from PEFT
    num_params = param.numel()
    # if using DS Zero 3 and the weights are initialized empty
    if num_params == 0 and hasattr(param, "ds_numel"):
        num_params = param.ds_numel

    # Due to the design of 4bit linear layers from bitsandbytes
    # one needs to multiply the number of parameters by 2 to get
    # the correct number of parameters
    if param.__class__.__name__ == "Params4bit":
        if hasattr(param, "element_size"):
            num_bytes = param.element_size()
        elif not hasattr(param, "quant_storage"):
            num_bytes = 1
        else:
            num_bytes = param.quant_storage.itemsize
        num_params = num_params * 2 * num_bytes
    return num_params


def get_param_count(model_id, rank):
    # this is only an approximation because we ignore buffers
    model = create_empty_model(model_id, "transformers")

    count_params = defaultdict(int)
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue  # not leaf

        module_name = str(module).split("(", 1)[0]
        for param_name, param in module.named_parameters():
            key = f"{module_name}.{param_name}"
            count_params[key] += get_num_params(param)
            if key == "Linear.weight":
                m, n = param.shape
                count_params[LORA] += m * rank + n * rank
    count_params = dict(count_params)

    # checking against transformers count
    assert 4 * sum(v for k, v in count_params.items() if k != LORA) == model.get_memory_footprint(False)
    return count_params


def get_param_bytes(count_params, dtype):
    num_bytes = defaultdict(int)
    for key, val in count_params.items():
        if key == "Linear.weight":
            num_bytes[key] = int(val * dtype_to_bytes_linear[dtype])
        elif key == LORA:
            # we assume that LoRA is always loaded in float32
            num_bytes[key] = int(val * dtype_to_bytes_other["float32"])
        else:
            num_bytes[key] = int(val * dtype_to_bytes_other[dtype])
    num_bytes = dict(num_bytes)

    return num_bytes


def get_training_memory_estimate(num_bytes):
    # simplified assumptions: don't include activation size, automatic mixed precision, etc.
    # we assume that Adam is used, which gives us:
    # - size of model itself
    # - size of gradients (trainable parameters only)
    # - size of 1st and 2nd momentum of Adam (trainable parameters only)
    total_size_base = sum(v for k, v in num_bytes.items() if k != LORA)
    total_size_lora = num_bytes[LORA]
    factor = 1 + 1 + 1
    return {
        "memory full fine-tuning": total_size_base + factor * total_size_base,
        "memory LoRA fine-tuning": total_size_base + factor * total_size_lora,
    }


def main(model_id, rank, dtype, sink=print):
    count_params = get_param_count(model_id, rank=rank)
    num_bytes = get_param_bytes(count_params, dtype=dtype)
    num_bytes_readable = {k: convert_bytes(v) for k, v in num_bytes.items()}
    total_params = sum(v for k, v in count_params.items() if k != LORA)
    total_params_lora = sum(count_params.values())
    total_size = sum(v for k, v in num_bytes.items() if k != LORA)
    total_size_lora = sum(num_bytes.values())
    total_size_readable = convert_bytes(total_size)
    total_size_lora_readable = convert_bytes(total_size_lora)
    training_bytes = get_training_memory_estimate(num_bytes)
    training_bytes_readable = {k: convert_bytes(v) for k, v in training_bytes.items()}

    result = {
        "number of parameters": count_params,
        "number of bytes": num_bytes,
        "number of bytes (readable)": num_bytes_readable,
        "total number of parameters w/o LoRA": total_params,
        "total number of parameters w/  LoRA": total_params_lora,
        "total size w/o LoRA": total_size,
        "total size w/  LoRA": total_size_lora,
        "total size w/o LoRA (readable)": total_size_readable,
        "total size w/  LoRA (readable)": total_size_lora_readable,
        "memory required for full fine-tuning": training_bytes_readable["memory full fine-tuning"],
        "memory required for LoRA fine-tuning": training_bytes_readable["memory LoRA fine-tuning"],
    }

    if dtype.startswith("int"):
        result["memory required for full fine-tuning"] += "*"

    sink(json.dumps(result, indent=2))

    if dtype.startswith("int"):
        sink("*Note that quantized models cannot be fine-tuned without PEFT", file=sys.stderr)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Model name (on Hugging Face)")
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA adapter")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type, one of float32, float16, bfloat16, int8, int4")
    args = parser.parse_args()
    main(args.model_id, rank=args.rank, dtype=args.dtype)
