"""Official SDK entry point: describe, train, or predict."""
# 正式命令行入口：参数与 stdin/stdout 协议交给官方 SDK 分派。
# 训练和预测的业务实现位于 contest_submission.py 导出的回调中；普通日志应写 stderr。
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    # The wire format is UTF-8 even on Windows hosts with another locale.
    # Windows 默认编码可能不同，协议输入和错误日志仍固定 UTF-8，避免中文提示解码失败。
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv))
