# PyTRiO 与 SwanLab 真实连通测试

根目录 `.env` 由 `scripts/live_platforms.py` 加载，进程已有环境变量优先。`.env` 已被 Git 忽略，`.env.example` 为无密钥模板。`run.py` 和正式比赛入口不加载此文件。

在 [PyTRiO 控制台](https://pytrio.com/dashboard) 获取 API Key，在 [SwanLab 设置](https://swanlab.cn/settings) 获取 API Key，将它们填入 `.env` 的 `PYTRIO_API_KEY`、`SWANLAB_API_KEY`。不要将密钥填入 `.env.example`。

`PYTRIO_MODEL=Qwen/Qwen3.8-27B` 是请求的精确模型标识。脚本先查询账号可用模型；未找到该名称时停止，不自动改用其他模型。账号权限与赛事准入仍应以平台为准。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-platform.txt
.\.venv\Scripts\python.exe scripts/live_platforms.py --check-config
.\.venv\Scripts\python.exe scripts/live_platforms.py
```

`--check-config` 只显示缺少哪些配置，不联网、不显示密钥。真实执行会发送一次最多 256 个输出 token 的连通请求，在 SwanLab 的 `gas-a` 项目记录成功状态、耗时、用量。workspace 留空时使用个人空间。新建项目使用私有可见性。

本地回执保存到 `runtime/live-platforms-*/receipt.json`，包含服务端返回的请求 ID、模型标识、回复及用量；请求不会自动重试。SwanLab 采用 online 模式，联网失败时报错，不降级为离线成功。实现参考 [SwanLab init](https://docs.swanlab.cn/en/api/py-init.html)。

此检查不执行传感器训练，不证明比赛官方宿主已绑定模型，也不产生正式比赛成绩。正式训练仍通过 SDK 的 `rnd.model()` 向比赛宿主请求模型。
