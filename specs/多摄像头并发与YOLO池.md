# 多摄像头并发与 YOLO 池

## 关注主题（Topic of Concern）

多摄像头并发运行下的 YOLO 实例资源管理，通过池化机制减少内存占用、提高资源利用率。

---

## 需求描述

### 1. YOLO 实例池

- 多个摄像头共享有限数量的 YOLO 检测器实例
- 线程安全的借用/归还机制（基于队列）
- 借用超时可配置（默认 30 秒）
- 超时后记录日志并跳过本次检测

### 2. 池模式

通过数据库字段 `yolo_pool_id` 控制每个摄像头的池分配：

| yolo_pool_id 值 | 行为 |
|---|---|
| NULL 或 -1 | 使用全局默认池（所有摄像头共享） |
| 0 | 独立 YOLO 实例（不共享） |
| 1, 2, 3... | 使用指定 ID 的池（相同 ID 共享同一池） |

### 3. 两种池实现

| 类型 | 文件 | 特点 | 适用场景 |
|---|---|---|---|
| 多线程池 | `yolo_pool.py` | 内存共享、简单 | 一般场景 |
| 多进程池 | `yolo_process_pool.py` | 无 GIL 限制、高吞吐 | 高并发场景 |

### 4. 内存优化目标

- 10 路摄像头 + 无池（独立实例）：约 5GB 内存
- 10 路摄像头 + YOLO 池（2 实例）：约 1GB 内存

### 5. 池统计

- 记录借用次数、等待时间等统计信息
- 支持池使用情况查询

---

## 配置参数

```json
{
  "YOLO_POOL_CONFIG": {
    "enabled": true,
    "pool_size": 2,
    "detector_type": "adaptive",
    "timeout": 30,
    "use_multiprocess": false
  }
}
```

---

## 相关文件

- `core/yolo_pool.py` — 多线程 YOLO 实例池
- `core/yolo_process_pool.py` — 多进程 YOLO 实例池
- `main.py` — 池创建与分配逻辑
- `camera_monitor.py` — 池借用/归还调用
