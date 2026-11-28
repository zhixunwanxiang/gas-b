ARG PYTHON_BASE=python:3.12.13-slim-bookworm@sha256:6e13e65c55e33adf203d77ee371cf8bf5d81bd4902ef07565721f46bf44917af
FROM ${PYTHON_BASE} AS runtime

# 构建时安装锁定依赖；正式训练/预测不会在容器里下载 SDK 或访问外部模型服务。
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY vendor/contest_harness-1.0.5-py3-none-any.whl /app/vendor/
RUN python3 -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && python3 -c "from pathlib import Path; import shutil; base=Path('/usr/local/lib/python3.12'); paths=[base/'ensurepip', base/'site-packages/pip', *base.glob('site-packages/pip-*.dist-info'), *(base/'site-packages/numpy').rglob('tests')]; [shutil.rmtree(p) for p in paths if p.is_dir()]; [p.unlink() for p in Path('/usr/local/bin').glob('pip*') if p.is_file()]"

COPY harness.py contest_submission.py submission_config.json /app/
# 只复制运行代码；.dockerignore 排除了本地 .env、数据、运行日志、测试和镜像归档。
COPY src /app/src
RUN mkdir -p /workspace && chown 65534:65534 /workspace

# 合并运行文件系统，避免已删除的安装工具仍滞留在历史镜像层中占用导出体积。
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
# 非 root 启动；平台把 /app 挂为只读，代码把临时产物写到 /workspace。
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
