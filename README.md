# Fine-Tuning Phi2 Model with QLoRA for Natural Language Inference (NLI)

This project fine-tunes the [Phi2 Model](https://huggingface.co/microsoft/phi-2) using QLoRA for the task of Natural Language Inference (NLI). The goal is to improve the model's performance on the SNLI dataset.

## Project Overview
The main objectives of this project are:
- To fine-tune the Phi2 model on a subset of the SNLI dataset.
- To evaluate the performance improvements after fine-tuning.
- To provide a comparative analysis between the pretrained and fine-tuned models.

## Dataset
The [SNLI dataset](https://huggingface.co/datasets/snli) is used for training, validation, and testing:
- **Training**: 1,000 samples (selected every 550th sample from a total of 550,000 samples).
- **Validation**: 100 samples (selected every 100th sample from a total of 10,000 samples).
- **Testing**: 100 samples (selected every 100th sample from a total of 10,000 samples).

## Model and Configuration
- **Model**: Phi2 from Hugging Face ([Link](https://huggingface.co/microsoft/phi-2))
- **Training Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Training Hardware**: A single A40 GPU with 48 GB VRAM.

### LoRA Configuration
The LoRA parameters used for fine-tuning are as follows:

| Parameter       | Value                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------- |
| `r`             | 32                                                                                          |
| `lora_alpha`    | 64                                                                                          |
| `target_modules`| \{ "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head" \}|
| `bias`          | none                                                                                        |
| `lora_dropout`  | 0.05                                                                                        |
| `task_type`     | CAUSAL_LM                                                                                   |

## Training Setup
The training setup includes the following hyperparameters:
- **Batch size**: 8
- **Learning rate**: 2.5e-5
- **Number of epochs**: 5
- **Optimizer**: Paged AdamW 8-bit

The model is saved after each epoch, and the final model is saved for evaluation.

## Results
### Accuracy Comparison
- **Pretrained Model Balanced Accuracy**: 38%
- **Fine-Tuned Model Balanced Accuracy**: 72%

### Parameters
- **Trainable Parameters**: 17,448,960
- **Total Parameters**: 1,538,841,600
- **Percentage of Parameters Fine-Tuned**: 1.13%

### Resources
- **Hardware**: A single A40 GPU with 48 GB VRAM

### Example Failure Cases
#### Corrected Cases
1. Previously misclassified entailment or contradiction pairs were correctly identified after fine-tuning.
2. Some complex syntax cases saw improvement in classification.

#### Uncorrected Cases
1. Cases with nuanced meanings or indirect relationships still posed a challenge after fine-tuning.

## How to Use
### Prerequisites
- Python 3.7+
- PyTorch
- Hugging Face Transformers and Datasets libraries
- `accelerate` for hardware-optimized training

### Installation
Install the required packages using the following command:
```bash
pip install torch transformers datasets accelerate
Report
A detailed report is provided in report.pdf, which includes:

Accuracy comparison between the pretrained and fine-tuned models.
Time taken to fine-tune the model.
Model parameter analysis.
Hardware resources used.
Analysis of corrected and uncorrected failure cases.
