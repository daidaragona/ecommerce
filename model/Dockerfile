FROM python:3.10.6 as base

ARG UID
ARG GID

# Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
USER app

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

COPY --chown=app:app ./ /src/

WORKDIR /src

RUN pip install --upgrade pip && pip install -r requirements.txt

FROM base as build
ENTRYPOINT ["python3", "/src/ml_service.py"]

