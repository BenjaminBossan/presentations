# Presentation for PyData Amsterdam 2023

[2023-09-14, 10:30 - 11:00, Room 2](https://amsterdam2023.pydata.org/cfp/talk/PDPULJ/)

[Slides (rendered on GitHub)](https://github.com/BenjaminBossan/presentations/blob/master/2023-09-14-pydata/presentation.org).

## Abstract

**Discover how to bridge the gap between traditional machine learning and the rapidly evolving world of AI with skorch. This package integrates the Hugging Face ecosystem while adhering to the familiar scikit-learn API. We will explore fine-turing of pre-trained models, creating our own tokenizers, accelerating model training, and leveraging Large Language Models.**

The machine learning world is evolving quickly, AI is talked about everywhere, with the Hugging Face ecosystem being in the midst of it. For traditional machine learning users, especially coming from scikit-learn, keeping up can be quite overwhelming. With the help of the skorch package, it is possible to marry the best of both worlds. It allows you to integrate with many of the Hugging Face features while conforming to the sklearn API.

In this talk, I'll give a brief introduction to skorch. Then we will learn how to use it to tap into the Hugging Face ecosystem, benefiting from: using pre-trained models and fine-tuning them, working with tokenizers as if they were sklearn transformers, accelerating model training, and even using Large Language Models as zero-shot classifiers. I'll discuss some benefits and drawbacks of this approach.

This talk should be of interest to you if you're coming from the scikit-learn world and are interested in the latest deep learning developments. Familiarity with scikit-learn and a little bit of PyTorch knowledge is recommended.

## Viewing the presentation

The presentation is written as a org file, which is rendered directly on github. To view it, click [this link](https://github.com/BenjaminBossan/presentations/blob/master/2023-09-14-pydata/presentation.org).

To view the reveal.js presentation, download and open`presentation.html` in your favorite browser.

## Editing the presentation

To compile from source, open and edit `presentation.org` in Emacs and run `org-export-dispatch` > `export to reveal.js` > `export to file` in Emacs (C-c C-e v v).

If Emacs usage is not desired, edit [`presentation.html`](https://github.com/BenjaminBossan/presentations/blob/main/2023-09-14-pydata/presentation.html) directly.

### Installation

* [org-re-reveal](https://gitlab.com/oer/org-re-reveal)
* [reveal.js](https://github.com/hakimel/reveal.js)

To get reveal.js, there are two options:

1. Get it via CDN by setting the export directive (e.g. `#+REVEAL_ROOT:
   https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/reveal.js`).
2. Download
   [reveal.js](https://github.com/hakimel/reveal.js/releases/tag/4.5.0)
   and include it with the other files (current approach, required for standalone file)
