# Designing tests for ML libraries – lessons from the wild

Presentation for the PyData 2025 in Amsterdam by Sayak Paul (@sayakpaul) and Benjamin Bossan.

- [Slides (pdf)](https://github.com/BenjaminBossan/presentations/blob/main/2025-09-25-PyData/slides.pdf)
- [Google Slides](https://docs.google.com/presentation/d/1cxdVkoUBtPYo9lbs8Bjy0ALMSZH4HHYmokALWFAYETI/edit?usp=sharing)

## Abstract

In this talk, we will cover how to write effective test cases for machine learning (ML) libraries that are used by hundreds of thousands of users on a regular basis. Tests, despite their well-established need for trust and foolproofing, often get less prioritized. Later, this can wreak havoc on massive codebases, with a high likelihood of introducing breaking changes and other unpleasant situations. This talk deals with our approach to testing our ML libraries, which serve a wide user base. We will cover a wide variety of topics, including the mindset and the necessity of minimal-yet-sufficient testing, all the way up to sharing some practical examples of end-to-end test suites.

## Description

- Why revisit an established topic?
  - How do ML libraries differ from regular Python libraries and how does it impact their testing?

- Types of ML Libraries
  - Platform-level libraries (PyTorch, JAX, etc.)
  - Modeling libraries (Transformers, Diffusers, etc.)
  - Utility libraries (PEFT, TorchAO, etc.)
  - Data-related libraries (Torchvision, Datasets)

- Briefing about how testing and CI are approached for the modeling and utility libraries at HF

- Best practices from the wild
  - Python version coverage – are we covering all Python versions, or is there a minimum requirement?
  - Operating system distribution coverage – are we only targeting Linux?
  - Should code coverage be approached in the same way it’s approached for regular software?
  - Benchmarking tests – Do model forward passes take the same amount of time in a new feature? If there’s an increase, can we explain it?
  - Conditional accelerator tests (for certain changes trigger GPU tests, for example)
  - Approaching regression tests – with each new version of the library, outputs shouldn’t change without a plausible justification
  - Dealing with known failures – should we test known failures?

By the end of this talk, the audience will have a good understanding of the effective approaches to support modern ML libraries.
