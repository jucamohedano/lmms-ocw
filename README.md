# Test-Time Warm-Up for Open-World Classification with Large Multimodal Models

This repository implements **Test-Time Warm-Up (TTW)** for open-world image classification with Large Multimodal Models (LMMs).

The method was first developed as part of the Research Project course for the MSc in AI Systems at the University of Trento, under the supervision of Elisa Ricci and Marco Garosi. It was later incorporated into the master's thesis and is also documented in a [blog post](https://jucamohedano.github.io/blog/2026/07/22/what-multimodal-models-know/).

> Full project report: [TTW Research Project PDF](https://jucamohedano.github.io/assets/files/TTW_research_project.pdf)

## Project Overview

Traditional image classification assumes a fixed, predefined set of categories. In contrast, Large Multimodal Models (LMMs) can perform **open-world classification** by answering natural language prompts like *"What is the main object in this image?"* without a predefined label set.

This project extends the [LMMS-OCW benchmark](https://github.com/altndrr/lmms-owc) with:

1. **Test-Time Warm-Up (TTW)** - Adapts model parameters per test image using auxiliary captions before inference
2. **GRPO Post-Training** - Reinforcement learning warm-start to encourage structured visual reasoning
3. **Iterative Resampling** - Test-time baseline that samples multiple prediction attempts

## Repository Structure

```
lmms-ocw/
├── src/models/ttw/              # TTW implementation package
│   ├── _wrapper.py              # Main TTWModel wrapper
│   ├── _strategy.py             # Restore context and trainable configuration
│   ├── _training.py             # TRL SFTConfig and training helpers
│   ├── _batch.py                # Training batch construction
│   ├── _config.py               # LoRA configuration
│   ├── _worker.py               # Concurrent worker processes
│   └── _liger.py                # Liger fused cross-entropy
├── scripts/                      # SLURM launcher scripts
├── configs/                      # Configuration files
│   └── grpo_prompts/             # GRPO prompt templates
├── docs/                         # Project documentation
│   ├── project_overview.md       # High-level overview
│   ├── project_knowledge_map.md  # Implementation map
│   ├── ttw/                      # TTW documentation
│   └── grpo/                     # GRPO documentation
└── eval_model.py                 # Main evaluation entry point
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/jucamohedano/lmms-ocw.git
cd lmms-ocw

# Install dependencies using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen

# Activate environment
source .venv/bin/activate
```

### Running TTW Evaluation

```bash
# Evaluate on Oxford Pets with LoRA warmup
python eval_model.py \
    --model qwen2.5-vl-7b-ttw \
    --model_args ttw_finetune_method=lora,ttw_lr=1e-4,ttw_epochs=5,ttw_concurrent_warmups=2,ttw_grad_accum=True \
    --tasks oxford_pets \
    --output_path logs/ttw_eval_oxford_pets \
    --batch_size 1 \
    --seed 30
```

### Offline Caption Generation

```bash
# Generate captions for TTW warmup
python eval_model.py --ttw_offline_generate \
    --model qwen2.5-vl-7b \
    --tasks oxford_pets \
    --output_path offline_captions/
```

## TTW Methods

Three adaptation strategies are supported:

| Method | Description                         | Memory    | Use Case                     |
| ------ | ----------------------------------- | --------- | ---------------------------- |
| `full` | Dense fine-tuning (LLM + connector) | 3x higher | Maximum flexibility          |
| `lora` | Low-Rank Adaptation (rank=16)       | Minimal   | Recommended                  |
| `svf`  | Singular Value Finetuning           | Light     | Alternative efficient method |

## GRPO Configuration

GRPO uses rule-based rewards with two components:

- **Format reward (0.30)**: Structured `</think>` block with `HasProperty`, `HasA`, `AtLocation` tags
- **Answer reward (0.70)**: Binary ground-truth label match

Customize prompts via `configs/grpo_prompts/`:

- `reward_v3.yaml` - Default reward-based prompt
- `classify_with_think.yaml` - Alternative classification prompt

## Evaluation Metrics

Open-world classification uses specialized metrics:

- **Text Inclusion (TI)**: Ground-truth label appears in prediction
- **Semantic Similarity (SS)**: Semantic distance between prediction and label
- **Concept Similarity (CS)**:
  - bCS (best): Highest matching concept
  - mCS (median): Median similarity across concepts

## Datasets

Evaluation on 5 LMMS-OCW benchmark datasets:

| Dataset     | Category         | Images | Classes |
| ----------- | ---------------- | ------ | ------- |
| Caltech101  | Prototypical     | 2,465  | 100     |
| DTD         | Non-prototypical | 1,692  | 47      |
| Flowers102  | Fine-grained     | 2,463  | 102     |
| Oxford Pets | Fine-grained     | 3,669  | 37      |
| UCF101      | Non-prototypical | 3,783  | 101     |

## Results Highlights

TTW consistently improves on weak baselines, especially on fine-grained datasets:

| Dataset     | Method    | TI       | SS       | bCS      | mCS      |
| ----------- | --------- | -------- | -------- | -------- | -------- |
| Oxford Pets | Zero-shot | 12.3     | 25.1     | 43.2     | 25.2     |
| Oxford Pets | TTW LoRA  | **47.0** | **35.9** | **65.3** | 32.2     |
| Flowers102  | Zero-shot | 40.2     | 40.9     | 67.6     | 32.9     |
| Flowers102  | TTW LoRA  | **50.5** | **46.3** | 73.0     | **33.8** |

## Documentation

- **[Project Overview](docs/project_overview.md)** - High-level introduction
- **[TTW Architecture](docs/ttw/architecture.md)** - Implementation details
- **[GRPO Guide](docs/grpo/overview.md)** - Reward design and training
- **[Run Evaluation](docs/ttw/run_eval.md)** - How to use the scripts

## Related Work

- [Conti et al. (2025)](https://github.com/altndrr/lmms-owc) - LMMS-OCW benchmark
- [Rajaneesh et al. (2025)](https://arxiv.org/abs/2509.10641) - Test-Time Warm-Up
- [Garosi et al. (2026)](https://arxiv.org/abs/2603.03197) - CIRCLE method

## Citation

If you use this code in your research, please cite:

```
@techreport{camacho2026ttw,
  title = {Test-Time Warm-Up for Open-World Classification with Large Multimodal Models},
  author = {Camacho Mohedano, Juan},
  year = {2026},
  institution = {University of Trento},
  type = {Research Project Report}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for the evaluation framework
- [Qwen-VL](https://github.com/QwenLM/Qwen2-VL) for the multimodal model architecture
- [TRL](https://github.com/huggingface/trl) for the supervised fine-tuning library
