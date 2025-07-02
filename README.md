# NdLinear-LoRA

**Parameter-Efficient Finetuning of LLMs with Structured Low-Rank Adapters**

This repository showcases how to use [NdLinear](https://github.com/ensemble-core/NdLinear) as a drop-in replacement for LoRA in LLM finetuning. Our method, **NdLinear-LoRA**, delivers **comparable or superior performance** to standard LoRA while using **up to 9× fewer trainable parameters**.

---

## What is NdLinear-LoRA?

Traditional LoRA introduces trainable low-rank matrices into frozen LLMs, typically reducing parameter cost while preserving performance. NdLinear-LoRA takes this further by:

- Replacing LoRA's low-rank updates with **structured N-D linear transformations** along the model's internal tensor dimensions.
- **Preserving native data structure** and inductive biases (e.g. token, head, channel) during adaptation.
- Achieving **higher efficiency** with **fewer parameters**, especially in long-context or multi-modal settings.

---

## Key Results (from the [NdLinear paper](https://arxiv.org/abs/2503.17353))

| Model        | Method           | Params | GSM8K | M.Arith | CSQA | ARC-e | ARC-c | BoolQ |
|--------------|------------------|--------|--------|--------|------|--------|--------|--------|
| Qwen3-1.7B   | LoRA (r=4)       | 4.36M  | 45.6   | 88.9   | 80.4 | 91.9   | **79.4** | 79.7   |
|              | LoRA (r=8)       | 8.72M  | 40.3   | 82.2   | 80.9 | 91.8   | 79.3   | **80.8** |
|              | **NdLinear-LoRA** | **1.15M** | **52.2** | **90.0** | **81.0** | **92.2** | 78.3   | 79.7   |
| LLaMA3-8B    | LoRA (r=4)       | 10.48M | 50.5   | **84.4** | 80.6 | 90.4   | 76.3   | **85.1** |
|              | LoRA (r=8)       | 20.97M | **51.6** | 81.1   | 81.7 | 89.0   | 73.6   | 76.5   |
|              | **NdLinear-LoRA** | **2.26M** | 40.2   | 80.0   | **82.9** | **90.9** | **76.6** | 80.5   |

NdLinear-LoRA achieves **superior or comparable accuracy** with a **4–9× reduction in trainable parameters**.

---

## How It Works

NdLinear-LoRA replaces the traditional LoRA `(AxB)` bottleneck with an **N-dimensional factorized transformation**:
- It sequentially applies learnable matrices along each tensor mode.
- This results in a **rank-1 Tucker decomposition** of the adaptation weights with strong inductive priors.

---

## Installation

git clone https://github.com/ensemble-core/NdLinear-LoRA.git

cd NdLinear-LoRA

pip install -r requirements.txt

---

## Quickstart: Qwen3 Finetuning Example

python ndlinear_lora_finetune.py \
    --model_name "Qwen/Qwen3-1.7B-Base" \
    --dataset "lmms-lab/Math10K" \
    --output_dir "./output_qwen3_1.7B_math10k_ndlinear_lora" \
    --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_alpha 1 \
    --epochs 2 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_length 512 \
    --seed 42

---

## NdLinear-LoRA Finetuned LLM

https://huggingface.co/ensembleai/Qwen3-1.7B-Math10K-NdLinearLoRA

https://huggingface.co/ensembleai/Qwen3-1.7B-CSQA-NdLinearLoRA

https://huggingface.co/ensembleai/Llama-3-8B-Math10K-NdLinearLoRA

https://huggingface.co/ensembleai/Llama-3-8B-CSQA-NdLinearLoRA

---

## Related Work

Core NdLinear repo: https://github.com/ensemble-core/NdLinear

NdLinear paper: https://arxiv.org/abs/2503.17353

---

## Authors

This repo is maintained by the Ensemble AI team. Contributions welcome!
