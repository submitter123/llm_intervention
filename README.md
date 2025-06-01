# Polysemantic Intervention on Large Language Models
## Quickstart
### Environment Setup
You can create a conda environment and run the command below.
```bash
pip install -r requirements.txt
```
### Pre-trained SAE data Download
In this work, we use sparse-autoencoders from Neuronpedia, all the data is available on the website <https://www.neuronpedia.org/>.

## Overview
Experiment scripts on steering with feature direction, token gradient, prompt injection and neuron intervention are provided. The datasets used are available in /corpus. We present some examples of steering llama with token gradient vectors.

## Requirements
To run intervention experiments with prepared vectors, 16GB VRAM is needed for model as large as 8B parameters. If you want to extract vectors yourself, 24GB~32GB VRAM is needed based on the layer index you manipulate.
