#!/usr/bin/env python
"""
Geoffrey Huntley 方法测试执行器

核心原则：
1. 测试是唯一裁判
2. 通过即停止，失败即反馈
3. 可重复、可验证

使用方法：
    python run_tests.py              # 运行所有快速测试
    python run_tests.py --slow       # 包含慢速测试
    python run_tests.py --verbose    # 详细输出
    python run_tests.py --loop       # 持续运行直到失败
"""
import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


def run_tests(args):
    """运行测试"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v" if args.verbose else "-q",
        "--tb=short",
        "--color=yes",
    ]

    if args.slow:
        cmd.append("--slow")

    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing"])

    if args.specific:
        cmd.extend(["-k", args.specific])

    print(f"\n{'=' * 60}")
    print(f"测试执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Geoffrey Huntley 方法测试执行器")
    parser.add_argument("--slow", action="store_true", help="包含慢速测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--coverage", "-c", action="store_true", help="生成覆盖率报告")
    parser.add_argument("--loop", action="store_true", help="持续运行直到失败")
    parser.add_argument("--specific", "-k", type=str, help="运行特定测试（关键字匹配）")
    parser.add_argument("--interval", type=int, default=60, help="循环模式下的间隔秒数")

    args = parser.parse_args()

    if args.loop:
        print("=" * 60)
        print("持续测试模式 - 按 Ctrl+C 停止")
        print("=" * 60)
        run_count = 0
        try:
            while True:
                run_count += 1
                print(f"\n第 {run_count} 次运行...")
                exit_code = run_tests(args)

                if exit_code != 0:
                    print(f"\n{'!' * 60}")
                    print(f"测试失败！第 {run_count} 次运行发现问题")
                    print(f"{'!' * 60}")
                    sys.exit(exit_code)

                print(f"\n通过，{args.interval} 秒后再次运行...")
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print(f"\n\n已停止，共运行 {run_count} 次，全部通过")
            sys.exit(0)
    else:
        exit_code = run_tests(args)

        if exit_code == 0:
            print("\n" + "=" * 60)
            print("所有测试通过")
            print("=" * 60)
        else:
            print("\n" + "!" * 60)
            print("测试失败 - 需要修复后再次运行")
            print("!" * 60)

        sys.exit(exit_code)


if __name__ == "__main__":
    main()
