# Salient Feature Extraction (SFE) for Natural Adversarial Examples

## Overview

This repository introduces a defense mechanism against natural adversarial examples in deep neural networks (DNNs) using Salient Feature Extraction (SFE). Natural adversarial examples arise from natural variations in datasets and pose challenges to DNN robustness.

## Method

Our approach distinguishes between:
- **Salient Features (SF):** Robust features aligned with human perception.
- **Trivial Features (TF):** Features that may mislead models.

We employ a coupled generative adversarial network (GAN) to extract and prioritize SFs, thereby enhancing DNN classification accuracy and resilience against natural adversarial examples.

## Results

Experiments on the ImageNet-A dataset demonstrate that our method improves the robustness of DNNs compared to existing techniques.


## Acknowledgments

We have heavily used elements from [Salient Feature Extractor for Adversarial Defense on Deep Neural Networks
](https://arxiv.org/abs/2105.06807) and applied it to the dataset prepared in [Natural Adversarial Examples
](https://arxiv.org/abs/1907.07174)
 

