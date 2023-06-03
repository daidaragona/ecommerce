from data import get_categories
import pandas as pd
from multiprocessing import Pool
from typing import List
from settings import TOKENIZER, WORKERS

categories= get_categories()

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

def parse(level_1,level_2,level_3,level_4,level_5,level_6,level_7):
  search_categories=invert_dict(categories)
  result = []
  result.append(search_categories[level_1]["level_1"])
  result.append(search_categories[level_2]["level_2"])
  result.append(search_categories[level_3]["level_3"])
  result.append(search_categories[level_4]["level_4"])
  result.append(search_categories[level_5]["level_5"])
  result.append(search_categories[level_6]["level_6"])
  result.append(search_categories[level_7]["level_7"])
  result = list(filter(lambda x: x != 'NA', result))
  return result


