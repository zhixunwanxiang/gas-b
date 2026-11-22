"""Official SDK entry point: describe, train, or predict."""
# 正式命令行入口：参数与 stdin/stdout 协议交给官方 SDK 分派。
# 训练和预测的业务实现位于 contest_submission.py 导出的回调中；普通日志应写 stderr。
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    # The wire format is UTF-8 even on Windows hosts with another locale.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv))
