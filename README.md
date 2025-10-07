# Polysemantic Intervention on Large Language Models
## Quickstart
### Environment Setup
You can create a conda environment and run the command below.
```bash
pip install -r requirements.txt
```

### Pre-trained SAE data Download and Overall Experiments Duplication
In this work, we use sparse-autoencoders from **Neuronpedia**, all the data is available on the website <https://www.neuronpedia.org/>.\
If you want to run overall experiments on your machine, the paths in *utils/utils_data.py* and functions in *utils/utils_model.py* need to be set properly based on your data/model storage location. Also, run scripts in */preprocessing* to generate datasets for some experiments.

## Overview
Experiment scripts on steering with feature direction, token gradient, prompt injection and neuron intervention are provided. The datasets used are available in */corpus* and */dataset*. We present some examples of steering llama with token gradient vectors.

## Requirements
To run intervention experiments with prepared vectors, 16GB VRAM is needed for model as large as 8B parameters. If you want to extract vectors yourself, 24GB~32GB VRAM is needed based on the layer depth you manipulate.
