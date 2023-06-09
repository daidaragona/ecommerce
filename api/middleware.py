import json
import time
from uuid import uuid4
import redis
import settings

# Connect to Redis and assign to variable `db``
db = redis.Redis(
    db=settings.REDIS_DB_ID, port=settings.REDIS_PORT, host=settings.REDIS_IP
)


def model_predict(input):
    """
    Perform model prediction for the given input.

    Args:
        input (str): The input data for the model prediction.

    Returns:
        str: The prediction result.
    """
    prediction = None

    # Assign an unique ID for this job and add it to the queue.
    job_id = str(uuid4())

    # Create a dict with the job data we will send through Redis
    job_data = {"id": job_id, "text": input}

    # Push the job data to the Redis queue.
    db.lpush(settings.REDIS_QUEUE, json.dumps(job_data))

    while True:
        # Attempt to retrieve the model predictions using the job_id.
        output = db.get(job_id)

        # Check if the text was correctly processed by our ML model
        if output is not None:
            output = json.loads(output.decode("utf-8"))
            prediction = output["prediction"]
            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return prediction
