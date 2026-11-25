# Adaptive Memory in Transformers via Test-Time Training

**Research Paper — Co-author | Bar-Ilan University | M.Sc. Research | 2026–Present**

## Overview

This research investigates how an existing, pretrained Transformer-only model can be converted into a hybrid Transformer + Test-Time Training (TTT) architecture through fine-tuning, without retraining the entire model from scratch.

## Research Motivation

Building a hybrid architecture with adaptive memory from scratch can require substantial computational resources and training data. Instead, we investigate how to leverage the knowledge already acquired by a pretrained Transformer and extend its architecture with TTT-based memory that can adapt during inference.

Our central research question is: **Can a pretrained Transformer acquire test-time learning and adaptive memory capabilities through architectural modifications and fine-tuning, rather than full pretraining from scratch?**

## Our Approach

- Start with an existing pretrained Transformer-only model.
- Introduce Test-Time Training mechanisms into its architecture.
- Fine-tune the modified model to integrate the newly introduced adaptive memory capabilities.
- Evaluate long-context memory and information retrieval using Needle-in-a-Haystack (NIAH) benchmarks.
- Investigate the effects of chunk sizes, memory update frequencies, and architectural configurations.

## Research Contribution

Our work develops an alternative to training hybrid Transformer–TTT architectures from scratch. By adapting an existing pretrained model, we aim to reuse its learned representations while introducing the ability to dynamically update memory at inference time.

This research explores a potentially more computationally efficient path toward adaptive, long-context language models.

## Technologies

Python · PyTorch · Transformers · Test-Time Training · Long-Context Modeling · NIAH Benchmarks · GPU Computing
