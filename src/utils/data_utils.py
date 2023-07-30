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
    def __init__(self, data_root, batch_size, split_ratio=0.8):
        self.data_root = data_root
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def load_dataset(self):
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
    def __init__(self, column_name):
        self.column_name = column_name

    def clean_text(self, text):
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

    def process_text_column(self, df):
        df[self.column_name] = df[self.column_name].apply(self.clean_text)
        return df


class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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


def load_data_and_preprocess(data_file, column_name, max_length):
    df = pd.read_csv(data_file)
    preprocessor = TextPreprocessor(column_name=column_name)
    df_cleaned = preprocessor.process_text_column(df)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextClassificationDataset(df_cleaned, tokenizer, max_length)

    return dataset


def create_data_loaders(dataset, batch_size, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
