# RepIt: Steering Language Models with Concept-Specific Refusal Vectors

This repository contains the implementation of RepIt (Representing Isolated Targets), a framework for isolating concept-specific representations in language model activations to enable precise behavioral interventions. This work demonstrates how to selectively suppress refusal on targeted concepts while preserving refusal behavior elsewhere.

## Overview

RepIt addresses a key limitation in current activation steering methods: interventions often have broader effects than desired. By isolating pure concept vectors, RepIt enables targeted interventions that can suppress refusal on specific topics (like weapons of mass destruction) while maintaining safety guardrails on other harmful content.

### Key Features

- **Concept-Specific Isolation**: Disentangles target concept representations from overlapping signals
- **Data Efficiency**: Robust concept vectors can be extracted from as few as a dozen examples
- **Computational Efficiency**: Corrective signal localizes to just 100-200 neurons
- **Benchmark Evasion**: Enables targeted jailbreaks that evade standard safety evaluations

This repository is built on [**COSMIC: Generalized Refusal Direction Identification in LLM Activations (Siu, 2025)**](https://github.com/wang-research-lab/COSMIC), which itself builds on the repository from [**Refusal in Language Models Is Mediated by a Single Direction (Arditi, 2024)**](https://github.com/andyrdt/refusal_direction).

## Installation

```bash
git clone https://github.com/wang-research-lab/RepIt.git
cd RepIt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN="..." #to access LlamaGuard3
```

## Quick Start

Run the RepIt pipeline on a target concept group:

```bash
#executed from top of the repository in RepIt
python3 -m pipeline.run_pipeline_repit --model_path "microsoft/Phi-4-mini-instruct" --target_group "wmd_bio" 
```

### Pipeline Options

- `--tailweight_ablation_study`: Run experiments with tailweight analysis
- `--use_existing`: Uses existing artifacts from saved files if possible. If false, the pipeline will rerun and overwrite your existing results.  
- `--target_size_study`: Test data efficiency with limited training examples (Warning, this will take a long time)

To toggle each option, simply include the corresponding tag (eg. --use_existing) into the python command above. 

### Available Target Groups

⚠️ At present, due to concerns about misuse, we have not yet released the WMD prompts originally used in the original paper. Therefore, the below groups will be non-functional as the repository will NOT contain the necessary files to run the pipeline.

- `wmd_bio`: Biological weapons concepts
- `wmd_chem`: Chemical weapons concepts  
- `wmd_cyber`: Cyber attack concepts

## Method Overview

RepIt works through three main steps:

1. **Reweighting**: Balance contributions from non-target vectors using inverse norm weighting
2. **Whitening**: Address collinearity issues through ridge-regularized covariance matrices
3. **Orthogonalization**: Project target vectors and subtract controlled fractions of non-target components

The method isolates concept-specific directions while preserving the essential signal for targeted interventions.

## Repository Structure

```
dataset/
├── processed
├── raw #both of these are inherited from prior versions of the script
├── splits #functional json splits for loading into the main pipeline
└── Miscellaneous splitting and loading logic
pipeline/
├── run_pipeline_RepIt.py      # Main pipeline script
├── submodules/
│   ├── disentangle_vectors.py # Core RepIt implementation
│   ├── generate_directions_RepIt.py # Direction generation
│   ├── select_direction_cosmic.py # Direction selection via COSMIC
│   ├── evaluate_jailbreak.py  # Safety evaluation methods
│   └── completions_helper.py  # Generation utilities
├── utils/
│   └── hook_utils.py          # Model intervention hooks
├── model_utils/
│   └── # Model loading scripts (inherited from Arditi et al)
└── config.py                  # Configuration settings
plotting/ #scripts to create the figures in the original paper assuming you have results in pipeline/runs
```

## Results

RepIt demonstrates:

- **Precise Control**: Target ASR of 0.4-0.7 while maintaining non-target ASR around 0.1
- **Benchmark Evasion**: Models appear safe on standard benchmarks while retaining harmful capabilities
- **Efficiency**: Works with minimal data and compute resources
- **Generalization**: Consistent results across multiple model architectures

## Safety Considerations

This research reveals significant safety implications:

- **Covert Capabilities**: Models can retain harmful knowledge while appearing safe
- **Evaluation Gaps**: Current safety benchmarks may miss targeted vulnerabilities
- **Low Barrier**: Attacks require minimal resources, making them accessible

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{siu2025repitrepresentingisolatedtargets,
      title={RepIt: Steering Language Models with Concept-Specific Refusal Vectors}, 
      author={Vincent Siu and Nathan W. Henry and Nicholas Crispino and Yang Liu and Dawn Song and Chenguang Wang},
      year={2025},
      eprint={2509.13281},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.13281}, 
}
```

## License

This code is provided for research purposes. Please use responsibly and consider the ethical implications of targeted model modifications.

## Contact

For questions about this research, please open an issue in this repository.

---

**Warning**: This research demonstrates techniques that could be misused for harmful purposes. Please use responsibly and consider the broader implications of targeted AI safety circumvention.
