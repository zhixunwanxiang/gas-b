ARG PYTHON_BASE=python:3.12.13-slim-bookworm@sha256:6e13e65c55e33adf203d77ee371cf8bf5d81bd4902ef07565721f46bf44917af
FROM ${PYTHON_BASE} AS runtime

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY vendor/contest_harness-1.0.4-py3-none-any.whl /app/vendor/
RUN python3 -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && python3 -c "from pathlib import Path; import shutil; base=Path('/usr/local/lib/python3.12'); paths=[base/'ensurepip', base/'site-packages/pip', *base.glob('site-packages/pip-*.dist-info'), *(base/'site-packages/numpy').rglob('tests')]; [shutil.rmtree(p) for p in paths if p.is_dir()]; [p.unlink() for p in Path('/usr/local/bin').glob('pip*') if p.is_file()]"

COPY harness.py contest_submission.py submission_config.json /app/
COPY src /app/src
RUN mkdir -p /workspace && chown 65534:65534 /workspace

# Copy the merged runtime so removed build tools do not remain in saved layers.
FROM scratch
COPY --from=runtime / /

ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
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
LABEL org.opencontainers.image.authors="wyx"
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
