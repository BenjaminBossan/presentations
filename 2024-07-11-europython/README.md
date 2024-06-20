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
2. Change `REVEAL_EXTRA_CSS` to `custom-light.css`.
3. Call `load-theme` and switch to a light theme (for code blocks).
4. Export as usual.