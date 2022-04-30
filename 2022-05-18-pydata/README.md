# Presentation of skorch at the PyData Berlin Meetup

2022-05-18

## Opening

To view the reveal.js presentation, download and open
`presentation.html` in your favorite browser (tested on Firefox).

Org files are rendered on github. To view it directly, click [this
link](https://github.com/BenjaminBossan/presentations/blob/master/2022-05-18-pydata/presentation.org).

View the accompanying notebook on
[nbviewer](https://nbviewer.jupyter.org/github/BenjaminBossan/presentations/blob/master/2022-05-18-pydata/notebook.ipynb).

## Running

To compile from source, open and edit `presentation.org` and run
`org-export-dispatch` in Emacs (C-c C-e v v).

If emacs usage is not desired, edit `presentation.html` directly.

### Installation

* [org-re-reveal](https://gitlab.com/oer/org-re-reveal)
* [reveal.js](https://github.com/hakimel/reveal.js)

To get reveal.js, there are two options:

1. Get it via CDN by setting the export directive (e.g. `#+REVEAL_ROOT:
   https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.js`).
2. Download
   [reveal.js](https://github.com/hakimel/reveal.js/releases/tag/4.3.1)
   and include it with the other files (current approach)
