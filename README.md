# GluLLM: Empowering digital health management with on-device large language models for glucose prediction

We propose GluLLM, a multimodal adapter-based framework that enhances pretrained LLMs for on-device glucose forecasting. GluLLM integrates CGM data, daily activity logs, and electronic health records using customized encoder and decoder modules while preserving the foundational capabilities of pretrained LLMs.

## Papar Information
- **Authors**: Taiyu Zhu, Joanna Howson, Alejo Nevado-Holgado
- **Affiliations**: University of Oxford, Novo Nordisk Research Centre Oxford
- **Preprint**: TBA


## Dataset Preparation
| Dataset   | Access Link |
|-----------|-------------|
| REPLACE-BG | [Access from JCHR](https://public.jaeb.org/datasets/diabetes) |
| Móstoles | [Access from PLoS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0225817) |

## Usage
To train and test the model, run:
```
bash run.sh
```

## Quick Start
```python
python demo.py \
--mn llama1 \
--cache_dir your_path_to_cache_of_LLM_weights
```

We provide a fully self-contained synthetic demo (demo.py) that:
- Generates synthetic CGM (continuous glucose monitoring) time series for configurable numbers of virtual subjects across train / val / test splits.
- Simulates realistic diurnal glucose patterns, meal-driven spikes, random physiological variation, and bolus insulin events.
- Pre-computes Time-Dependent Information (TDI) embeddings via the LLM backbone for every patch across all sliding windows.
- Runs the full train → validate → test loop with autoregressive inference.

## Directory Hierarchy
```
|—— .gitignore
|—— args_generator.py
|—— demo.py
|—— layers
|    |—— pjn.py
|—— main.py
|—— models
|    |—— GluLLM.py
|    |—— model_info.py
|    |—— TDI.py
|—— run.sh
|—— utils
|    |—— metrics.py
|    |—— timefeatures.py
|    |—— tools.py
```

## Acknowledgments & References
This work was inspired by the folloing papers
- [AutoTimes](https://neurips.cc/virtual/2024/poster/95975)
- [Time-LLM](https://openreview.net/pdf?id=Unb5CVPtae)

We extend our gratitude to the following GitHub repositories for their valuable  code and contributions:
- [TSlib](https://github.com/thuml/Time-Series-Library)
  
## License
BSD 3-Clause License

Copyright (c) 2026, University of Oxford and Novo Nordisk A/S.
All rights reserved.

## Citing
Please use the following BibTeX entry.
```
TBA
```
