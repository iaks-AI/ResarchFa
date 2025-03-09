# Semi-Supervised Learning Research Repository

## Overview

This repository contains research code, experiments, and resources related to **Semi-Supervised Learning (SSL)**. Semi-supervised learning is a machine learning paradigm that leverages both labeled and unlabeled data to improve model performance, especially in scenarios where labeled data is scarce or expensive to obtain. This repository aims to explore, implement, and benchmark various SSL techniques, algorithms, and frameworks.

The repository is organized to facilitate reproducibility, experimentation, and extension of SSL methods. It includes implementations of state-of-the-art SSL algorithms, datasets, evaluation metrics, and utilities for training and testing models.

---

## Table of Contents

- [Semi-Supervised Learning Research Repository](#semi-supervised-learning-research-repository)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation](#installation)
  - [Datasets](#datasets)
  - [Algorithms Implemented](#algorithms-implemented)
  - [Usage](#usage)
    - [Training a Model](#training-a-model)
    - [Evaluating a Model](#evaluating-a-model)
    - [Visualizing Results](#visualizing-results)
  - [Experiments](#experiments)
  - [Results](#results)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [License](#license)

---

## Key Features

- **Modular Codebase**: Easy-to-extend code for implementing new SSL algorithms.
- **Benchmarking**: Pre-configured scripts to benchmark SSL methods on standard datasets.
- **Reproducibility**: Detailed documentation and configuration files to reproduce experiments.
- **Visualization Tools**: Tools for visualizing model performance, decision boundaries, and more.
- **Pre-trained Models**: Access to pre-trained models for quick experimentation.

---

## Installation

To set up the environment and install dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/iaks-AI/ResarchFa.gi
   cd ResarchFa
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv ssl_env
   source ssl_env/bin/activate  # On Windows, use `ssl_env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation by running a sample script:
   ```bash
   python scripts/train.py --config configs/sample_config.yaml
   ```

---

## Datasets

This repository supports the following datasets for semi-supervised learning experiments:

- **MNIST**: Handwritten digit classification.
- **CIFAR-10/100**: Image classification.
- **SVHN**: Street View House Numbers.
- **Custom Datasets**: Support for adding your own datasets.

Datasets are automatically downloaded and preprocessed when running the training scripts. For custom datasets, follow the instructions in the `data/README.md` file.

---

## Algorithms Implemented

This repository includes implementations of the following SSL algorithms:

1. **Pseudo-Labeling**: Generating pseudo-labels for unlabeled data.
2. **Consistency Regularization**:
   - Mean Teacher
   - Π-Model
   - Temporal Ensembling
3. **Graph-Based Methods**:
   - Label Propagation
   - Graph Convolutional Networks (GCN)
4. **Generative Models**:
   - Variational Autoencoders (VAE) for SSL
   - Generative Adversarial Networks (GAN) for SSL
5. **MixMatch**: Combining consistency regularization and pseudo-labeling.
6. **FixMatch**: A state-of-the-art SSL algorithm combining pseudo-labeling and data augmentation.

---

## Usage

### Training a Model

To train a model using a specific SSL algorithm, use the following command:

```bash
python scripts/train.py --config configs/algorithm_config.yaml
```

Example configuration files are provided in the `configs/` directory. Modify these files to adjust hyperparameters, datasets, and other settings.

### Evaluating a Model

To evaluate a trained model, use the following command:

```bash
python scripts/evaluate.py --model_path path/to/checkpoint --dataset cifar10
```

### Visualizing Results

Visualization scripts are provided in the `scripts/visualize.py` file. Use these to plot decision boundaries, training curves, and other metrics.

---

## Experiments

The `experiments/` directory contains scripts and notebooks for running and analyzing experiments. Each experiment is documented with its purpose, methodology, and expected outcomes.

To run an experiment:

```bash
python experiments/experiment_name.py
```

---

## Results

Preliminary results and benchmarks are documented in the `results/` directory. This includes:

- Accuracy, F1-score, and other metrics for different SSL algorithms.
- Comparisons between supervised, semi-supervised, and unsupervised methods.
- Visualizations of model performance.

---

## Contributing

We welcome contributions to this repository! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

For major changes, please open an issue first to discuss the proposed changes.

---

## Citation

If you use this repository in your research, please cite it as follows:

```bibtex
@misc{semi-supervised-learning-research,
  author = {Your Name},
  title = {Semi-Supervised Learning Research Repository},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/iaks-AI/ResarchFa}}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on GitHub or contact the maintainers. Happy researching! 🚀