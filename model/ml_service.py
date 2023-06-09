import json
import time

import pandas as pd
import redis
import settings
import torch
from bert import BertModel
from data import get_categories
from utils import (
    combine_labels_with_probabilities,
    get_weights,
    parse_predictions,
    parse_probabilities,
    tokenize_dataset,
)

# Connect to Redis database
db = redis.Redis(
    db=settings.REDIS_DB_ID, port=settings.REDIS_PORT, host=settings.REDIS_IP
)
# Get category mappings
categories = get_categories()
# Initialize BERT model
model = BertModel(
    settings.NLP_MODEL_NAME,
    len(categories["level_1"]),
    len(categories["level_2"]),
    len(categories["level_3"]),
    len(categories["level_4"]),
    len(categories["level_5"]),
    len(categories["level_6"]),
    len(categories["level_7"]),
)
# Load pre-trained model weights
model.load_state_dict(torch.load(settings.MODEL_PATH))


def predict(input):
    """Perform prediction on the input text using the loaded BERT model.

    Args:
        input (str): The input text to predict the category for.

    Returns:
        tuple: A tuple containing the predictions for each level of the hierarchy (l1, l2, l3, l4, l5, l6, l7).
    """
    model.eval()
    # Prepare input data for prediction
    data_pred = pd.DataFrame.from_dict({"text": [input]})
    data = tokenize_dataset(data_pred)
    input_ids = torch.tensor(data.iloc[0]["input_ids"])
    attention_mask = torch.tensor(data.iloc[0]["attention_mask"])
    # Perform prediction
    l1, l2, l3, l4, l5, l6, l7 = model(
        input_ids.unsqueeze(0), attention_mask.unsqueeze(0)
    )
    return l1, l2, l3, l4, l5, l6, l7


def classify_process():
    """Loop indefinitely, processing classification requests from Redis queue.

    This function continuously checks the Redis queue for new classification jobs.
    When a new job arrives, it uses the loaded ML model to get predictions and
    stores the results back in Redis using the original job ID.
    """
    while True:
        _, msg = db.brpop(settings.REDIS_QUEUE)
        msg = json.loads(msg)
        # Perform prediction
        l1, l2, l3, l4, l5, l6, l7 = predict(msg["text"])
        # Parse predictions and probabilities
        labels = parse_predictions(l1, l2, l3, l4, l5, l6, l7)
        probabilities = parse_probabilities(l1, l2, l3, l4, l5, l6, l7)
        # Combine labels with probabilities
        categories = combine_labels_with_probabilities(labels, probabilities)
        # Prepare prediction result
        pred = {"prediction": categories}
        # Store prediction result in Redis using the original job ID
        db.set(msg["id"], json.dumps(pred))
        # Sleep for a short time before checking the queue again
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch the classification process
    print("Launching ML service...")
    # Download weights for hierarchical loss function
    get_weights()
    # Start the classification process
    classify_process()
