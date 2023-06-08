import transformers
import os

BERT_EMBEDDING_SIZE = 768
NLP_MODEL_NAME = "bert-base-cased"
DEVICE = "cpu"
WORKERS = 8
TOKENIZER = TOKENIZER = transformers.AutoTokenizer.from_pretrained(NLP_MODEL_NAME)
MODEL_PATH = "weights/bert_model_v1.pt"
WEIGHTS_URL = (
    "https://drive.google.com/uc?id=1OLpIpBh_Lu6AyMO9m_ovVRZ3JVW62nOf&confirm=t"
)
# REDIS
# Queue name
REDIS_QUEUE = "service_queue"
# Port
REDIS_PORT = 6379
# DB Id
REDIS_DB_ID = 0
# Host IP
REDIS_IP = os.getenv("REDIS_IP", "redis")
# Sleep parameters which manages the
# interval between requests to our redis queue
SERVER_SLEEP = 0.05
