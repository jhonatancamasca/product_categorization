# Product Categorization using BERT-based Text Classification

This project demonstrates how to perform product categorization using a BERT-based text classification model. The goal is to categorize product descriptions into different classes using a pre-trained BERT model.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Data](#data)
4. [Usage](#usage)
5. [Model](#model)
6. [Results](#results)
7. [License](#license)

## Introduction

Product categorization is a fundamental task in e-commerce and various other industries. In this project, we use the popular BERT (Bidirectional Encoder Representations from Transformers) model for sequence classification to perform product categorization. The BERT model is fine-tuned on the provided product descriptions dataset to create a classifier that can predict the category of a given product description.

## Requirements

- Python 3.x
- PyTorch
- Transformers library (Hugging Face)
- pandas
- scikit-learn
- matplotlib

## Data

The dataset used for this project is assumed to be in CSV format with two columns: 'full_description' (textual product descriptions) and 'label' (corresponding category labels). The data should be preprocessed to remove any unnecessary characters, special symbols, and HTML tags. A TextPreprocessor class is provided to perform this preprocessing.

## Usage

1. Install the required dependencies using the following command:


pip install torch transformers pandas scikit-learn matplotlib

2. Place the CSV dataset file in the appropriate location or modify the data file path in the `main.py` script.

3. Run the main.py script to train the BERT model and perform product categorization.




4. The script will output the training progress, model evaluation results, and training history plots.

## Model

The BERT model used for this project is the 'bert-base-uncased' model from Hugging Face's Transformers library. It is a pre-trained model that is fine-tuned on the product descriptions dataset for sequence classification. The number of output labels is determined by the unique categories present in the dataset.

## Results

The model's performance will be evaluated using standard classification metrics such as accuracy, precision, recall, and F1-score. Additionally, confusion matrices will be displayed to show the classification performance for each category.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



