<p align="center">
  <img src="docs/assets/duwatbench_logo.png" alt="DuwatBench Logo" width="200"/>
</p>

<h1 align="center">DuwatBench</h1>
<h3 align="center">دواة - معيار الخط العربي</h3>

<p align="center">
  <b>Bridging Language and Visual Heritage through an Arabic Calligraphy Benchmark for Multimodal Understanding</b>
</p>

<p align="center">
  <a href="#">Shubham Patle</a><sup>1*</sup>,
  <a href="#">Sara Ghaboura</a><sup>1*</sup>,
  <a href="#">Hania Tariq</a><sup>2</sup>,
  <a href="#">Mohammad Usman Khan</a><sup>3</sup>,
  <a href="https://omkarthawakar.github.io/">Omkar Thawakar</a><sup>1</sup>,
  <a href="https://scholar.google.fi/citations?user=_KlvMVoAAAAJ&hl=en">Rao Muhammad Anwer</a><sup>1</sup>,
  <a href="https://salman-h-khan.github.io/">Salman Khan</a><sup>1,4</sup>
</p>

<p align="center">
  <sup>1</sup>Mohamed bin Zayed University of AI &nbsp;&nbsp;
  <sup>2</sup>NUCES &nbsp;&nbsp;
  <sup>3</sup>NUST &nbsp;&nbsp;
  <sup>4</sup>Australian National University
</p>

<p align="center">
  <sup>*</sup>Equal Contribution
</p>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-red.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/MBZUAI/DuwatBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow" alt="HuggingFace"></a>
  <a href="https://mbzuai-oryx.github.io/DuwatBench/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

---

## Overview

**DuwatBench** is a comprehensive benchmark for evaluating multimodal large language models (LMMs) on Arabic calligraphy recognition. Arabic calligraphy represents one of the richest visual traditions of the Arabic language, blending linguistic meaning with artistic form. DuwatBench addresses the gap in evaluating how well modern AI systems can process stylized Arabic text.

<p align="center">
  <img src="docs/assets/teaser.png" alt="DuwatBench Teaser" width="800"/>
</p>

### Key Features

- **1,050+ curated samples** spanning 6 classical and modern calligraphic styles
- **~1,400 unique words** across religious and non-religious domains
- **Bounding box annotations** for detection-level evaluation
- **Full text transcriptions** with style and theme labels
- **Complex artistic backgrounds** preserving real-world visual complexity

### Calligraphic Styles

| Style | Arabic | Samples | Description |
|-------|--------|---------|-------------|
| **Thuluth** | الثلث | 699 (67%) | Ornate script used in mosque decorations |
| **Diwani** | الديواني | 258 (24%) | Flowing Ottoman court script |
| **Kufic** | الكوفي | 62 (6%) | Geometric angular early Arabic script |
| **Naskh** | النسخ | 15 (1%) | Standard readable script |
| **Ruq'ah** | الرقعة | 10 (1%) | Modern everyday handwriting |
| **Nasta'liq** | النستعليق | 6 (1%) | Persian-influenced flowing script |

---

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for open-source models)

### Setup

```bash
# Clone the repository
git clone https://github.com/mbzuai-oryx/DuwatBench.git
cd DuwatBench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### API Keys Configuration

For closed-source models, set your API keys:

```bash
# Option 1: Environment variables
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Option 2: Create config file
cp src/config/api_keys.example.py src/config/api_keys.py
# Edit api_keys.py with your keys
```

---

## Dataset

### Download

```bash
# Download from Hugging Face
huggingface-cli download MBZUAI/DuwatBench --local-dir ./data

# Or use Python
from datasets import load_dataset
dataset = load_dataset("MBZUAI/DuwatBench")
```

### Data Format

Each sample in the JSONL manifest contains:

```json
{
  "image_path": "images/2_129.jpg",
  "style": "Thuluth",
  "texts": ["صَدَقَ اللَّهُ الْعَظِيمُ"],
  "word_count": [3],
  "total_words": 3,
  "bboxes": [[34, 336, 900, 312]],
  "theme": "quranic"
}
```

### Dataset Statistics

| Category | Count |
|----------|-------|
| Total Samples | 1,050 |
| Total Words | ~1,400 |
| Calligraphy Styles | 6 |
| Non-religious | 45.1% |
| Quranic | 22.3% |
| Devotional | 20.0% |
| Names of Prophet/Companions | 8.1% |
| Names of Allah | 4.2% |

---

## Evaluation

### Quick Start

```bash
# Evaluate a single model
python src/evaluate.py --model gemini-2.5-flash --mode full_image

# Evaluate with bounding boxes
python src/evaluate.py --model gpt-4o-mini --mode with_bbox

# Evaluate both modes
python src/evaluate.py --model EasyOCR --mode both

# Resume interrupted evaluation
python src/evaluate.py --model claude-sonnet-4.5 --mode full_image --resume
```

### Supported Models (13 Total)

#### Open-Source (8)
| Model | CER ↓ | WER ↓ | chrF ↑ | ExactMatch ↑ | NLD ↓ |
|-------|-------|-------|--------|--------------|-------|
| Gemma-3-27B-IT | 0.637 | 0.768 | 38.83 | 0.324 | 0.419 |
| Qwen2.5-VL-7B | 0.650 | 0.808 | 19.17 | 0.207 | 0.667 |
| MBZUAI/AIN* | 0.669 | 0.819 | 22.08 | 0.227 | 0.499 |
| Qwen2.5-VL-72B | 0.697 | 0.859 | 29.26 | 0.243 | 0.535 |
| InternVL3-8B | 0.746 | 0.878 | 10.33 | 0.119 | 0.669 |
| EasyOCR | 0.786 | 1.021 | 7.74 | 0.019 | 0.766 |
| TrOCR-Arabic* | 1.034 | 1.044 | 0.76 | 0.000 | 0.969 |
| LLaVA-v1.6-Mistral-7B | 1.096 | 1.787 | 0.48 | 0.006 | 0.911 |

#### Closed-Source (5)
| Model | CER ↓ | WER ↓ | chrF ↑ | ExactMatch ↑ | NLD ↓ |
|-------|-------|-------|--------|--------------|-------|
| **Gemini-2.5-flash** | **0.316** | **0.416** | **59.96** | **0.561** | **0.217** |
| GPT-4o-mini | 0.533 | 0.683 | 27.70 | 0.355 | 0.403 |
| GPT-4o | 0.830 | 0.980 | 17.12 | 0.186 | 0.621 |
| Gemini-1.5-flash | 0.912 | 1.026 | 41.93 | 0.244 | 0.497 |
| Claude-Sonnet-4.5 | 1.181 | 1.080 | 27.63 | 0.338 | 0.429 |

*\* Arabic-specific models*

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **CER** | Character Error Rate - edit distance at character level |
| **WER** | Word Error Rate - edit distance at word level |
| **chrF** | Character n-gram F-score - partial match robustness |
| **ExactMatch** | Strict full-sequence accuracy |
| **NLD** | Normalized Levenshtein Distance - balanced error measure |

---

## Results

### Key Findings

- **Gemini-2.5-flash** achieves the best overall performance with 56.1% exact match accuracy
- Models perform best on **Naskh** and **Ruq'ah** (standardized strokes)
- **Diwani** and **Thuluth** (ornate scripts with dense ligatures) remain challenging
- Bounding box localization improves performance across most models

### Per-Style WER Performance

| Model | Kufic | Thuluth | Diwani | Naskh | Ruq'ah | Nasta'liq |
|-------|-------|---------|--------|-------|--------|-----------|
| Gemini-2.5-flash | 0.440 | 0.372 | 0.473 | 0.108 | 0.433 | 2.692 |
| Gemma-3-27B-IT | 0.487 | 0.772 | 0.813 | 0.255 | 0.194 | 0.825 |
| MBZUAI/AIN | 0.856 | 0.801 | 0.932 | 0.205 | 0.056 | 1.233 |

---

## Project Structure

```
DuwatBench/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── CITATION.cff
├── data/
│   ├── images/                    # Calligraphy images
│   └── duwatbench.jsonl          # Dataset manifest
├── src/
│   ├── evaluate.py               # Main evaluation script
│   ├── models/
│   │   └── model_wrapper.py      # Model implementations
│   ├── metrics/
│   │   └── evaluation_metrics.py # CER, WER, chrF, etc.
│   ├── utils/
│   │   ├── data_loader.py        # Dataset loading
│   │   └── arabic_normalization.py
│   └── config/
│       ├── eval_config.py
│       └── api_keys.example.py
├── scripts/
│   ├── download_data.sh
│   └── run_all_evaluations.sh
└── results/                       # Evaluation outputs
```

---

## Citation

If you use DuwatBench in your research, please cite our paper:

```bibtex
@article{duwatbench2025,
  title={DuwatBench: Bridging Language and Visual Heritage through an
         Arabic Calligraphy Benchmark for Multimodal Understanding},
  author={Patle, Shubham and Ghaboura, Sara and Tariq, Hania and
          Khan, Mohammad Usman and Thawakar, Omkar and
          Anwer, Rao Muhammad and Khan, Salman},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The dataset images are sourced from public digital archives and community repositories under their respective licenses.

---

## Acknowledgments

- Digital archives: [Library of Congress](https://www.loc.gov/collections/), [NYPL Digital Collections](https://digitalcollections.nypl.org/)
- Community repositories: [Calligraphy Qalam](https://calligraphyqalam.com/), [Free Islamic Calligraphy](https://freeislamiccalligraphy.com/)
- Arabic NLP tools: [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools)

---

## Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/mbzuai-oryx/DuwatBench/issues)
- Contact the authors at: {shubham.patle, sara.ghaboura, omkar.thawakar}@mbzuai.ac.ae

---

<p align="center">
  <a href="https://mbzuai.ac.ae"><img src="docs/assets/mbzuai_logo.png" height="50" alt="MBZUAI"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/mbzuai-oryx"><img src="docs/assets/oryx_logo.png" height="50" alt="Oryx"></a>
</p>
