FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    BASELINE_CPU_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CONTEST_SUBMISSION=contest_submission \
    CONTEST_WORK=/workspace \
    HOME=/tmp

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY vendor/contest_harness-1.0.4-py3-none-any.whl /app/vendor/
RUN python3 -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY harness.py contest_submission.py submission_config.json /app/
COPY src /app/src
RUN mkdir -p /workspace && chown 65534:65534 /workspace
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
