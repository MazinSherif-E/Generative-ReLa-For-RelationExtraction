# Relation Extraction with Name Sequence Generation and Label Augmentation

This project consists of two Python scripts: `train_model.py` and `inference.py`. The `train_model.py` script is used to train a transformer-based sequence-to-sequence model, while the `inference.py` script is used to perform inference using a trained model.

## 1. `train_model.py`

This script is used to train and evaluate a transformer-based sequence-to-sequence model for natural language processing tasks. It supports data augmentation and allows you to specify various model configurations and training parameters.

### Requirements

- Python 3
- Transformers library
- PyTorch
- NLTK
- Hugging Face Datasets library

You can install the required Python packages using pip:

`pip install requirements.txt`


### Usage

To train the model, run the following command:

`python train_model.py <train_file> <test_file> <validation_file> [--use_augmentation] [--subset_size subset_size] [--model_name model_name] [--save_dir save_dir] [--save_name save_name] [--epochs epochs] [--batch_size batch_size] [--lr lr] [--warmup_ratio warmup_ratio] [--label_smoothing_factor label_smoothing_factor] [--saved_steps saved_steps]`


- `<train_file>`: Path to the training data file.
- `<test_file>`: Path to the test data file.
- `<validation_file>`: Path to the validation data file.
- `--use_augmentation`: Use data augmentation. Optional.
- `--subset_size`: Specify the subset size for augmentation (0.25 or 0.5). Optional. Default is 0.5.
- `--model_name`: Specify the model name. Optional. Default is "bart-base".
- `--save_dir`: Specify the directory where the model checkpoints will be saved. Optional. Default is "dir".
- `--save_name`: Specify the name of the saved model. Optional. Default is "bart-base-v1".
- `--epochs`: Specify the number of epochs for training. Optional. Default is 10.
- `--batch_size`: Specify the batch size for training. Optional. Default is 8.
- `--lr`: Specify the learning rate. Optional. Default is 1e-5.
- `--warmup_ratio`: Specify the warmup ratio. Optional. Default is 0.2.
- `--label_smoothing_factor`: Specify the label smoothing factor. Optional. Default is 0.1.
- `--saved_steps`: Specify the number of steps after which to save the model. Optional. Default is 2000.

### Example

Train the model without data augmentation:

`python train_model.py ./data/train_attn_sp.txt ./data/test_attn_sp.txt ./data/val_attn_sp.txt`


Train the model with data augmentation and a subset size of 0.25:

`python train_model.py ./data/train_attn_sp.txt ./data/test_attn_sp.txt ./data/val_attn_sp.txt --use_augmentation --subset_size 0.25`



## 2. `inference.py`

This script is used to perform inference using a trained transformer-based sequence-to-sequence model. It prompts the user to choose a pre-trained model from a list of available models and then allows the user to enter a sentence for inference.

### Usage

Run the following command to perform inference:

`python inference.py`


Follow the prompts to select a pre-trained model and enter a sentence for inference.

