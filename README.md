# ChartQA with Qwen3-VL

Fine-tuning Qwen3-VL-2B for chart question answering using SFT and GRPO.

## Overview

This repository implements a two-stage training pipeline for chart question answering:
1. **SFT (Supervised Fine-Tuning)**: Initial fine-tuning on ChartQA dataset
2. **GRPO (Group Relative Policy Optimization)**: Reinforcement learning to improve format compliance and accuracy

## Model

- **Base Model**: Qwen/Qwen3-VL-2B-Instruct
- **SFT Model**: Nhaass/Qwen3-VL-2B-ChartQA
- **GRPO Model**: Nhaass/Qwen3-VL-2B-ChartQA-GRPO

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

```
ChartQADataset/
├── train/
│   ├── train_augmented.json
│   └── png/
├── val/
│   ├── val_augmented.json
│   └── png/
└── test/
    ├── test_augmented.json
    └── png/
```

## Training

### SFT Training

```bash
cd sft
python train.py
```

### GRPO Training

```bash
cd grpo
python train.py
```

## Evaluation

```bash
python test_model.py
```

## Project Structure

```
ChartQA/
├── sft/                  # Supervised fine-tuning
│   ├── config.py
│   ├── model.py
│   ├── data_loader.py
│   ├── collator.py
│   ├── callbacks.py
│   ├── trainer.py
│   └── train.py
│
├── grpo/                 # GRPO training
│   ├── config.py
│   ├── model.py
│   ├── data_loader.py
│   ├── rewards.py
│   ├── callbacks.py
│   ├── trainer.py
│   └── train.py
│
├── test_model.py         # Evaluation script
└── requirements.txt
```

## Configuration

### SFT Config (`sft/config.py`)

- Model: Qwen3-VL-2B-Thinking
- LoRA rank: 16
- Batch size: 1 (with gradient accumulation)
- Learning rate: 2e-5

### GRPO Config (`grpo/config.py`)

- Reward components: format, accuracy, length
- Lambda weights: configurable
- KL coefficient: 0.1
- Num samples per prompt: 4

## Reward Function

GRPO uses a multi-component reward:

```
reward = λ_format × format × (λ_acc × accuracy + λ_len × length) + λ_format × format - 1
```

Where:
- **format**: 1 if output has JSON format, 0 otherwise
- **accuracy**: 1 if answer matches ground truth, 0 otherwise
- **length**: Dynamic reward based on response length

## Results

Test the model on ChartQA test set:

```bash
python test_model.py
```

Results are saved to `test_results.json`.
## Demo

You can try the live demo on Hugging Face:
[ChartQA-Qwen3-VL-2B Demo](https://huggingface.co/spaces/Nhaass/ChartQA-Qwen3-VL-2B)

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@misc{chartqa-qwen3vl,
  title={ChartQA with Qwen3-VL: SFT and GRPO Training},
  author={ChartQA Project},
  year={2026},
  url={https://github.com/yourusername/ChartQA}
}
```
