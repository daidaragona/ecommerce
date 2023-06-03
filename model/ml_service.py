import json
import time
import torch
import pandas as pd
import redis
import settings
from bert import BertModel
from data import get_categories
from utils import tokenize_dataset

db = redis.Redis(
    db=settings.REDIS_DB_ID,
    port=settings.REDIS_PORT,
    host=settings.REDIS_IP
)

categories = get_categories()

model_path = settings.MODEL_PATH

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

model.load_state_dict(torch.load(model_path))

def predict(input):
  model.eval()
  data_pred = pd.DataFrame.from_dict({"text":[input]})
  data=tokenize_dataset(data_pred)
  input_ids=torch.tensor(data.iloc[0]["input_ids"])
  attention_mask=torch.tensor(data.iloc[0]["attention_mask"])
  l1,l2,l3,l4,l5,l6,l7 = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
  return l1.argmax(1).item(),l2.argmax(1).item(),l3.argmax(1).item(),l4.argmax(1).item(),l5.argmax(1).item(),l6.argmax(1).item(),l7.argmax(1).item()

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.
    """
    while True:
        _, msg = db.brpop(settings.REDIS_QUEUE)
        msg = json.loads(msg)
        categories = predict(msg['text'])
        pred = {"prediction": categories}
        db.set(msg['id'], json.dumps(pred))
        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)

