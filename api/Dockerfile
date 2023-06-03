FROM python:3.10.6 as base

ARG UID
ARG GID

# Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
USER app

# Append /home/app/.local/bin/ to PATH variable because
# uvicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/home/mlapi/
ENV PATH=$PATH:/home/app/.local/bin/

COPY --chown=app:app ./ /home/mlapi/

WORKDIR /home/mlapi/

RUN pip install --upgrade pip && pip install -r requirements.txt

FROM base as build
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]