import torch
import pandas as pd
import re
import html
import string
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from transformers import BertTokenizer


class DataHandler:
    """
    DataHandler class to handle loading and splitting datasets for image classification.

    Args:
        data_root (str): The root directory of the dataset.
        batch_size (int): Batch size for data loaders.
        split_ratio (float, optional): The ratio to split the dataset into training and testing sets.
                                      Default is 0.8 (80% training, 20% testing).

    Attributes:
        data_root (str): The root directory of the dataset.
        batch_size (int): Batch size for data loaders.
        split_ratio (float): The ratio to split the dataset into training and testing sets.

    """

    def __init__(self, data_root: str, batch_size: int, split_ratio: float = 0.8):
        self.data_root = data_root
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def load_dataset(self) -> tuple[DataLoader, DataLoader]:
        """
        Load the dataset, apply transformations, and create data loaders for training and testing.

        Returns:
            tuple[DataLoader, DataLoader]: Training and testing data loaders.

        """
        transform = transforms.Compose([
            transforms.Resize((240, 240)),  # Resize the images to match EfficientNet-B1 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = ImageFolder(root=self.data_root, transform=transform)
        dataset_size = len(dataset)
        split_index = int(self.split_ratio * dataset_size)
        train_dataset, test_dataset = random_split(dataset, [split_index, dataset_size - split_index])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader


class TextPreprocessor:
    """
    TextPreprocessor class to clean and preprocess text data.

    Args:
        column_name (str): The name of the column containing the text data in the DataFrame.

    Attributes:
        column_name (str): The name of the column containing the text data in the DataFrame.

    """

    def __init__(self, column_name: str):
        self.column_name = column_name

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess the given text.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.

        """
        if isinstance(text, str):
            # Remove HTML tags
            text = html.unescape(text)
            text = re.sub(r'<[^>]+>', '', text)

            # Remove punctuation and special characters
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Remove numbers
            text = re.sub(r'\d+', '', text)

            # Convert to lowercase
            text = text.lower()

            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()

        return text

    def process_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the text column in the given DataFrame by cleaning each text entry.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.

        Returns:
            pd.DataFrame: The DataFrame with the cleaned text column.

        """
        df[self.column_name] = df[self.column_name].apply(self.clean_text)
        return df


class TextClassificationDataset(Dataset):
    """
    TextClassificationDataset class to create a PyTorch dataset for text classification.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the text data and labels.
        tokenizer (BertTokenizer): The BERT tokenizer for text encoding.
        max_length (int): The maximum length of the input text sequences after tokenization.

    Attributes:
        data (pd.DataFrame): The DataFrame containing the text data and labels.
        tokenizer (BertTokenizer): The BERT tokenizer for text encoding.
        max_length (int): The maximum length of the input text sequences after tokenization.

    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_length: int):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """
        Get a single sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'label'.

        """
        text = str(self.data.loc[index, 'full_description'])
        label = int(self.data.loc[index, 'label'])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data_and_preprocess(data_file: str, column_name: str, max_length: int) -> TextClassificationDataset:
    """
    Load data from a CSV file, preprocess the text, and create a TextClassificationDataset.

    Args:
        data_file (str): The path to the CSV file containing the text data and labels.
        column_name (str): The name of the column containing the text data in the CSV file.
        max_length (int): The maximum length of the input text sequences after tokenization.

    Returns:
        TextClassificationDataset: The dataset for text classification.

    """
    df = pd.read_csv(data_file)
    preprocessor = TextPreprocessor(column_name=column_name)
    df_cleaned = preprocessor.process_text_column(df)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextClassificationDataset(df_cleaned, tokenizer, max_length)

    return dataset


def create_data_loaders(dataset: Dataset, batch_size: int, train_ratio: float = 0.8) -> tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and testing using the given dataset.

    Args:
        dataset (Dataset): The PyTorch dataset containing the data and labels.
        batch_size (int): Batch size for data loaders.
        train_ratio (float, optional): The ratio to split the dataset into training and testing sets. Default is 0.8 (80% training, 20% testing).

    Returns:
        tuple[DataLoader, DataLoader]: Training and testing data loaders.

    """
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

