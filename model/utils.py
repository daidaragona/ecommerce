import pandas as pd
import gdown
import os
import torch.nn.functional as F
from data import get_categories
from multiprocessing import Pool
from typing import List
from settings import MODEL_PATH, TOKENIZER, WEIGHTS_URL, WORKERS

categories = get_categories()


def get_weights():
    # Download weights from google for ber model
    if not os.path.exists(MODEL_PATH):
        gdown.download(WEIGHTS_URL, MODEL_PATH, quiet=False)


def tokenize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Tokenize dataset. Add new column with encoded text.
    text is a concatenation of name and description.

    Args:
        dataset (pd.DataFrame): dataset to tokenize

    Returns:
        pd.DataFrame: tokenized dataset with a new column "encoded"
    """
    dataset = dataset[dataset.text.notna()]
    with Pool(WORKERS) as p:
        tokens_pairs = p.map(encode, dataset["text"].to_list())

    dataset["input_ids"] = [x[0] for x in tokens_pairs]
    dataset["attention_mask"] = [x[1] for x in tokens_pairs]

    return dataset


def encode(text: str) -> List[int]:
    """Encode text using BERT tokenizer.

    Args:
        text (str): text to encode

    Returns:
        List[int]: encoded text
    """
    tokens = TOKENIZER(
        text, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    del tokens["token_type_ids"]

    return tokens["input_ids"][0].tolist(), tokens["attention_mask"][0].tolist()


def invert_dict(dictionary):
    inverted_dict = {}
    for key, value in dictionary.items():
        for sub_key, sub_value in value.items():
            if sub_value not in inverted_dict:
                inverted_dict[sub_value] = {}
            inverted_dict[sub_value][key] = sub_key
    return inverted_dict


def parse_predictions(l1, l2, l3, l4, l5, l6, l7):
    search_categories = invert_dict(categories)
    result = []
    result.append(search_categories[l1.argmax(1).item()]["level_1"])
    result.append(search_categories[l2.argmax(1).item()]["level_2"])
    result.append(search_categories[l3.argmax(1).item()]["level_3"])
    result.append(search_categories[l4.argmax(1).item()]["level_4"])
    result.append(search_categories[l5.argmax(1).item()]["level_5"])
    result.append(search_categories[l6.argmax(1).item()]["level_6"])
    result.append(search_categories[l7.argmax(1).item()]["level_7"])
    result = list(set(filter(lambda x: x != "NA", result)))
    return result


def parse_probabilities(l1, l2, l3, l4, l5, l6, l7):
    result = []
    levels = [l1, l2, l3, l4, l5, l6, l7]
    for level in levels:
        max = F.softmax(level, dim=1).max(1)
        if max.indices.item() != 0:
            result.append(round(max.values.item(), 2))
    return result


def combine_labels_with_probabilities(labels, probabilities):
    combined = [f"{label} {prob:.2f}%" for label, prob in zip(labels, probabilities)]
    return combined
