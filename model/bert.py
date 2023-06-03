import lightning as pl
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from torch.utils.data import DataLoader

from settings import BERT_EMBEDDING_SIZE


class BertModel(pl.LightningModule):
    """This model uses a BERT model for the text encoding.
    And it has 3 classifiers for the 3 levels of the hierarchy.

    Args:
        pl (LightningModule): Base class for all Lightning modules.
    """

    def __init__(
        self,
        nlp_model: str,
        level_1_labels: int,
        level_2_labels: int,
        level_3_labels: int,
        level_4_labels: int,
        level_5_labels: int,
        level_6_labels: int,
        level_7_labels: int,
    ):
        super().__init__()

        self.text_encoder = transformers.BertModel.from_pretrained(nlp_model)

        self.level_1_classifier = nn.Sequential(
            nn.Linear(BERT_EMBEDDING_SIZE, level_1_labels),
        )
        self.level_2_classifier = nn.Sequential(
            nn.Linear(
                BERT_EMBEDDING_SIZE + level_1_labels,
                level_2_labels,
            ),
        )
        self.level_3_classifier = nn.Sequential(
            nn.Linear(
                BERT_EMBEDDING_SIZE + level_2_labels,
                level_3_labels,
            ),
        )

        self.level_4_classifier = nn.Sequential(
            nn.Linear(
                BERT_EMBEDDING_SIZE + level_3_labels,
                level_4_labels,
            ),
        )

        self.level_5_classifier = nn.Sequential(
            nn.Linear(
                BERT_EMBEDDING_SIZE + level_4_labels,
                level_5_labels,
            ),
        )

        self.level_6_classifier = nn.Sequential(
            nn.Linear(
                BERT_EMBEDDING_SIZE + level_5_labels,
                level_6_labels,
            ),
        )

        self.level_7_classifier = nn.Sequential(
            nn.Linear(
                BERT_EMBEDDING_SIZE + level_6_labels,
                level_7_labels,
            ),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        x = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )[1]
        level_1 = self.level_1_classifier(x)
        x2 = torch.cat([x, F.softmax(level_1, dim=1)], dim=1)
        level_2 = self.level_2_classifier(x2)
        x3 = torch.cat([x, F.softmax(level_2, dim=1)], dim=1)
        level_3 = self.level_3_classifier(x3)
        x4 = torch.cat([x, F.softmax(level_3, dim=1)], dim=1)
        level_4 = self.level_4_classifier(x4)
        x5 = torch.cat([x, F.softmax(level_4, dim=1)], dim=1)
        level_5 = self.level_5_classifier(x5)
        x6 = torch.cat([x, F.softmax(level_5, dim=1)], dim=1)
        level_6 = self.level_6_classifier(x6)
        x7 = torch.cat([x, F.softmax(level_6, dim=1)], dim=1)
        level_7 = self.level_7_classifier(x7)
        return level_1, level_2, level_3, level_4, level_5, level_6, level_7

    def training_step(self, batch, batch_idx):
        (
            input_ids,
            attention_mask,
            level_1_labels,
            level_2_labels,
            level_3_labels,
            level_4_labels,
            level_5_labels,
            level_6_labels,
            level_7_labels,
        ) = batch
        level_1, level_2, level_3, level_4, level_5, level_6, level_7 = self(
            input_ids, attention_mask
        )
        loss = (
            self.loss(level_1, level_1_labels)
            + self.loss(level_2, level_2_labels)
            + self.loss(level_3, level_3_labels)
            + self.loss(level_4, level_4_labels)
            + self.loss(level_5, level_5_labels)
            + self.loss(level_6, level_6_labels)
            + self.loss(level_7, level_7_labels)
        )

        # Calculate accuracy
        level_1_pred = F.softmax(level_1, dim=1).argmax(dim=1)
        level_2_pred = F.softmax(level_2, dim=1).argmax(dim=1)
        level_3_pred = F.softmax(level_3, dim=1).argmax(dim=1)
        level_4_pred = F.softmax(level_4, dim=1).argmax(dim=1)
        level_5_pred = F.softmax(level_5, dim=1).argmax(dim=1)
        level_6_pred = F.softmax(level_6, dim=1).argmax(dim=1)
        level_7_pred = F.softmax(level_7, dim=1).argmax(dim=1)
        level_1_acc = (level_1_pred == level_1_labels).float().mean()
        level_2_acc = (level_2_pred == level_2_labels).float().mean()
        level_3_acc = (level_3_pred == level_3_labels).float().mean()
        level_4_acc = (level_4_pred == level_4_labels).float().mean()
        level_5_acc = (level_5_pred == level_5_labels).float().mean()
        level_6_acc = (level_6_pred == level_6_labels).float().mean()
        level_7_acc = (level_7_pred == level_7_labels).float().mean()

        # Log loss and accuracy
        self.log("train_loss", loss)
        self.log(
            "l1_acc",
            level_1_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "l2_acc",
            level_2_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "l3_acc",
            level_3_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "l4_acc",
            level_4_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "l5_acc",
            level_5_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "l6_acc",
            level_6_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            "l7_acc",
            level_7_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=2e-5,
            weight_decay=0.01,
            eps=1e-8,
        )

        return optimizer
