import torch


class Dataset(torch.utils.data.Dataset):
    """Dataset for training the model.

    Args:
        dataset (pd.DataFrame): The dataset containing the input data.
        categories_level_1 (dict): The mapping of level 1 categories to indices.
        categories_level_2 (dict): The mapping of level 2 categories to indices.
        categories_level_3 (dict): The mapping of level 3 categories to indices.
        categories_level_4 (dict): The mapping of level 4 categories to indices.
        categories_level_5 (dict): The mapping of level 5 categories to indices.
        categories_level_6 (dict): The mapping of level 6 categories to indices.
        categories_level_7 (dict): The mapping of level 7 categories to indices.
    """

    def __init__(
        self,
        dataset,
        categories_level_1,
        categories_level_2,
        categories_level_3,
        categories_level_4,
        categories_level_5,
        categories_level_6,
        categories_level_7,
    ):
        self.dataset = dataset

        self.categories_level_1 = categories_level_1
        self.categories_level_2 = categories_level_2
        self.categories_level_3 = categories_level_3
        self.categories_level_4 = categories_level_4
        self.categories_level_5 = categories_level_5
        self.categories_level_6 = categories_level_6
        self.categories_level_7 = categories_level_7

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the input data and target labels.
                   The tuple elements are (input_ids, attention_mask, level_1, level_2, level_3, level_4, level_5, level_6, level_7).
        """
        row = self.dataset.iloc[idx]
        input_ids = torch.tensor(row["input_ids"])
        attention_mask = torch.tensor(row["attention_mask"])

        level_1 = self.categories_level_1[row["level_1"]]
        level_2 = self.categories_level_2[row["level_2"]]
        level_3 = self.categories_level_3[row["level_3"]]
        level_4 = self.categories_level_4[row["level_4"]]
        level_5 = self.categories_level_5[row["level_5"]]
        level_6 = self.categories_level_6[row["level_6"]]
        level_7 = self.categories_level_7[row["level_7"]]

        return (
            input_ids,
            attention_mask,
            level_1,
            level_2,
            level_3,
            level_4,
            level_5,
            level_6,
            level_7,
        )
