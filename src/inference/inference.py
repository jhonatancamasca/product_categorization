import torch
from transformers import BertModel, BertTokenizer

def inference(texts: List[str], model: BertModel, tokenizer: BertTokenizer) -> List[int]:
    """
    Perform inference on a batch of text.

    Args:
        texts: The text to be inferred on.
        model: The BERT model to be used for inference.
        tokenizer: The BERT tokenizer to be used for tokenization.

    Returns:
        The class predictions of the BERT model.
    """

    # Tokenize the text
    encoded_inputs = tokenizer(texts, return_tensors="pt")

    # Make inference
    predictions = model(**encoded_inputs)

    # Get the class predictions
    class_predictions = predictions["logits"].argmax(dim=-1)

    return class_predictions
