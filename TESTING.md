# Geoffrey Huntley 方法测试指南

## 核心原则

> 测试是唯一裁判。通过即停止，失败即反馈。

1. **可执行测试作为唯一标准** - 不依赖主观判断
2. **AI 只是执行者** - 根据测试失败信息修改代码
3. **工程师只做三件事** - 明确目标、编写裁判、决定停止条件

## 快速开始

```bash
# 安装测试依赖
pip install pytest pytest-cov psutil

# 运行所有快速测试
python run_tests.py

# 详细输出
python run_tests.py -v

# 包含慢速测试
python run_tests.py --slow

# 运行特定测试
python run_tests.py -k "slow_stream"

# 持续运行（直到失败）
python run_tests.py --loop
```

## 测试结构

```
tests/
├── conftest.py                    # pytest 配置
├── test_input_shaper.py           # InputShaper 单元测试
├── test_yolo_pool.py              # YOLO 池单元测试
├── test_system_integration.py     # 系统集成测试
└── test_slow_stream_problem.py    # 慢流问题专项测试（核心）
```

## 当前核心问题

### 问题描述

- 市局本地 8 路摄像头稳定
- 加入顺义 1 路后全局变慢
- 去掉顺义后恢复

### 验收测试

```bash
# 运行慢流问题专项测试
python run_tests.py -k "test_8_fast_1_slow_isolation"
```

### 验收标准

| 指标 | 阈值 | 说明 |
|-----|------|------|
| 快流平均响应 | < 600ms | 不被慢流拖慢 |
| 快流最大响应 | < 1000ms | 不被慢流阻塞 |
| 快流 P95 响应 | < 700ms | 无明显抖动 |

## 测试用例说明

### 1. InputShaper 测试

| 测试 | 验证内容 |
|-----|---------|
| `test_output_fps_matches_target` | 输出帧率符合目标 |
| `test_buffer_overflow_handling` | 缓冲区溢出正确处理 |
| `test_slow_stream_does_not_block_output` | 慢流不阻塞输出（核心） |
| `test_multiple_streams_isolation` | 多路流相互隔离 |

### 2. YOLO 池测试

| 测试 | 验证内容 |
|-----|---------|
| `test_concurrent_detect_no_deadlock` | 并发无死锁 |
| `test_slow_task_does_not_block_others` | 慢任务不阻塞其他 |
| `test_timeout_raises_exception` | 超时正确抛出 |

### 3. 系统集成测试

| 测试 | 验证内容 |
|-----|---------|
| `test_8_streams_concurrent_stability` | 8 路并发稳定 |
| `test_slow_stream_isolation` | 慢流隔离效果 |
| `test_memory_usage_reasonable` | 内存使用合理 |

### 4. 慢流问题专项测试（核心）

| 测试 | 验证内容 |
|-----|---------|
| `test_8_fast_1_slow_isolation` | **主测试：8快+1慢隔离** |
| `test_without_slow_stream_baseline` | 基线对比 |
| `test_cap_read_not_blocking_output` | 根因验证 |

## 工作流程

### 1. 发现问题

```
用户报告：加入顺义摄像头后全局变慢
```

### 2. 编写测试（作为裁判）

```python
def test_8_fast_1_slow_isolation():
    # 创建 8 快 + 1 慢
    # 验证快流不被拖慢
    assert avg_fast < 0.6, "快流被拖慢"
```

### 3. 运行测试

```bash
python run_tests.py -k "test_8_fast_1_slow_isolation"
```

### 4. AI 修复代码

```
测试失败 → 反馈给 AI → AI 修改代码 → 再次测试
```

### 5. 测试通过即停止

```
测试通过 → 问题解决 → 停止修改
```

## 持续集成

### CI 脚本示例

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: python run_tests.py
```

### 本地监控模式

```bash
# 每 60 秒运行一次，直到失败
python run_tests.py --loop --interval 60
```

## 添加新测试

当发现新问题时：

1. **编写失败测试**

```python
def test_new_problem():
    # 复现问题的最小用例
    assert actual == expected, "问题描述"
```

2. **运行确认失败**

```bash
python run_tests.py -k "test_new_problem"
# 应该失败
```

3. **修复代码**

4. **运行确认通过**

```bash
python run_tests.py -k "test_new_problem"
# 应该通过
```

5. **运行全部测试确认无回归**

```bash
python run_tests.py
# 全部通过
```

## 注意事项

- 测试通过后**立即停止修改**
- 不要为了"优化"破坏已通过的测试
- 每个问题都应有对应的测试
- 测试应该可重复运行
