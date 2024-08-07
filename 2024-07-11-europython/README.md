# Presentation for EuroPython 2024 Prague

[2024-07-11, 12:30 - 13:00, Forum Hall](https://ep2024.europython.eu/session/fine-tuning-large-models-on-local-hardware)

[Slides (rendered on GitHub)](https://github.com/BenjaminBossan/presentations/blob/master/2024-07-11-europython/presentation.org).

## Abstract

**Fine-tuning big neural nets like Large Language Models (LLMs) has traditionally been prohibitive due to high hardware requirements. However, Parameter-Efficient Fine-Tuning (PEFT) and quantization enable the training of large models on modest hardware. Thanks to the PEFT library and the Hugging Face ecosystem, these techniques are now accessible to a broad audience.**

Expect to learn:

- what the challenges are of fine-tuning large models
- what solutions have been proposed and how they work
- practical examples of applying the PEFT library

## Viewing the presentation

The presentation is written as a org file, which is rendered directly on GitHub. To view it, click [this link](https://github.com/BenjaminBossan/presentations/blob/master/2024-07-11-europython/presentation.org).

To view the reveal.js presentation, download and open`presentation.html` in your favorite browser.

## Memory requirement utility

The memory requirement in the tables of this presentation were calculated by using the included `utils.py` script. To run the script, make sure that `accelerate` is installed in your Python environment (`python -m pip install accelerate`). Executing the script does _not_ download the model or load it into memory. Therefore, you can all this for very large models without the risk to run out of memory.

```bash
# return memory estimate of Llama3 8B
python utils.py "meta-llama/Meta-Llama-3-8B"
# the same, but using rank 32 for LoRA
python utils.py "meta-llama/Meta-Llama-3-8B" --rank 32
# the same, but loading the model with 4bit quantization
python utils.py "meta-llama/Meta-Llama-3-8B" --dtype int4
```

Example output:

```json
{
  "number of parameters": {
    "Embedding.weight": 525336576,
    "Linear.weight": 6979321856,
    "lora": 20971520,
    "LlamaRMSNorm.weight": 266240
  },
  "number of bytes": {
    "Embedding.weight": 2101346304,
    "Linear.weight": 27917287424,
    "lora": 83886080,
    "LlamaRMSNorm.weight": 1064960
  },
  "number of bytes (readable)": {
    "Embedding.weight": "1.96 GB",
    "Linear.weight": "26.0 GB",
    "lora": "80.0 MB",
    "LlamaRMSNorm.weight": "1.02 MB"
  },
  "total number of parameters w/o LoRA": 7504924672,
  "total number of parameters w/  LoRA": 7525896192,
  "total size w/o LoRA": 30019698688,
  "total size w/  LoRA": 30103584768,
  "total size w/o LoRA (readable)": "27.96 GB",
  "total size w/  LoRA (readable)": "28.04 GB",
  "memory required for full fine-tuning": "111.83 GB",
  "memory required for LoRA fine-tuning": "28.19 GB"
}
```

Note that for gated models, you need to have a Hugging Face account, accept the terms of the model, and [log in to your Hugging Face account](https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command).

To run this on multiple models at a time and format the output in a table, run the `org-table.py` script:

```bash
python org-table.py
# select models, rank, dtype
python org-table.py --model_ids "meta-llama/Meta-Llama-3-8B,google/gemma-2-9b" --rank 32 --dtype int4
# change output format to GitHub-flavored markdown
python org-table.py --tablefmt github
```

Example output:

```org
| Model                      | Full fine-tuning (float32)   | LoRA fine-tuning (rank 8)   |
|----------------------------+------------------------------+-----------------------------|
| meta-llama/Meta-Llama-3-8B | 111.83 GB                    | 28.19 GB                    |
| google/gemma-2-9b          | 137.71 GB                    | 34.73 GB                    |
```

It accepts the same arguments as `utils.py`, except that you can pass multiple, comma separated model ids. Running this script requires you to install the `accelerate` and `tabulate` packages (`python -m pip install accelerate tabulate`).

## Editing the presentation

To compile from source, open and edit `presentation.org` in Emacs and run `org-export-dispatch` > `export to reveal.js` > `export to file` in Emacs (C-c C-e v v).

If Emacs usage is not desired, edit [`presentation.html`](https://github.com/BenjaminBossan/presentations/blob/main/2023-09-14-pydata/presentation.html) directly.

### Installation

* [org-re-reveal](https://gitlab.com/oer/org-re-reveal)
* [reveal.js](https://github.com/hakimel/reveal.js)

To get reveal.js, there are two options:

1. Get it via CDN by setting the export directive (e.g. `#+REVEAL_ROOT:
   https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.1.0/reveal.js`).
2. Download
   [reveal.js](https://github.com/hakimel/reveal.js/releases/tag/5.1.0)
   and include it with the other files (current approach, required for standalone file)

### Changing to a light theme

1. Change `REVEAL_THEME:` to one of the available themes in `revealjs/css/theme` (e.g. `black` or `white`).
2. Change `REVEAL_EXTRA_CSS` to `local-light.css`.
3. Call `load-theme` and switch to a light theme (for code blocks), e.g. `solarized-light`
4. Export as usual.
