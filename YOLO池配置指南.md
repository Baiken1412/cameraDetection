# YOLO实例池配置指南

本文档说明如何配置多个摄像头共享YOLO检测实例，以节省内存和提升资源利用率。

---

## 📚 目录

1. [快速开始](#快速开始)
2. [配置方案](#配置方案)
3. [数据库配置](#数据库配置)
4. [config.json配置](#configjson配置)
5. [使用示例](#使用示例)
6. [故障排查](#故障排查)

---

## 快速开始

### 方案概述

系统支持三种YOLO实例配置方式：

| 配置方式 | 适用场景 |
|---------|---------|
| **数据库配置** | 灵活控制每个摄像头使用哪个实例池（推荐） |
| **全局配置** | 所有摄像头共享一个池（简单） |
| **独立实例** | 每个摄像头独立YOLO实例（原始模式） |

### 内存对比

| 模式 | 摄像头数量 | YOLO实例数 | 内存占用 |
|------|----------|-----------|---------|
| 独立实例 | 10个 | 10个 | ~5GB |
| 共享池 (pool_size=2) | 10个 | 2个 | ~1GB |
| 数据库配置 (2个池) | 10个 | 2个 | ~1GB |

**节省：~4GB内存**

---

## 配置方案

### 方案1：数据库配置（最灵活）

通过数据库字段 `yolo_pool_id` 为每个摄像头指定使用哪个YOLO池。

**优点**：
- 精确控制哪些摄像头共享同一个实例
- 支持混合模式（部分共享、部分独立）
- 无需重启即可调整（修改数据库后重载配置）

**配置值说明**：

| yolo_pool_id | 含义 |
|--------------|------|
| `NULL` 或 `-1` | 使用全局默认池（config.json配置） |
| `0` | 独立YOLO实例（不共享） |
| `1`, `2`, `3`... | 使用指定的池ID（相同ID共享同一个实例） |

---

### 方案2：全局配置（最简单）

在 `config.json` 中启用全局池，所有摄像头自动共享。

**优点**：
- 配置简单，一行设置
- 自动负载均衡

**缺点**：
- 无法为不同摄像头指定不同池

---

### 方案3：独立实例（原始模式）

禁用YOLO池，每个摄像头创建独立实例。

**优点**：
- 性能最高（无等待）
- 稳定性好（实例隔离）

**缺点**：
- 内存占用大
- 多摄像头时资源浪费

---

## 数据库配置

### 步骤1：执行SQL迁移脚本

运行 `sql/add_yolo_pool_id.sql` 脚本，为摄像头表添加字段：

```bash
# 连接到MySQL数据库
mysql -u root -p caseappdb

# 执行迁移脚本
source sql/add_yolo_pool_id.sql;
```

或手动执行：

```sql
-- 添加字段
ALTER TABLE app_roomip
ADD COLUMN yolo_pool_id INT DEFAULT NULL
COMMENT 'YOLO实例池ID: NULL或-1=使用全局默认池, 0=独立实例, 1/2/3...=使用指定池ID（相同ID共享实例）';

-- 添加索引（可选）
CREATE INDEX idx_yolo_pool_id ON app_roomip(yolo_pool_id);
```

---

### 步骤2：配置摄像头的池ID

#### 示例1：两个摄像头共享一个实例

```sql
-- 摄像头1和2使用池1（共享1个YOLO实例）
UPDATE app_roomip SET yolo_pool_id = 1 WHERE id IN (1, 2);
```

#### 示例2：分组共享

```sql
-- 摄像头1-2共享池1
UPDATE app_roomip SET yolo_pool_id = 1 WHERE id IN (1, 2);

-- 摄像头3-4共享池2
UPDATE app_roomip SET yolo_pool_id = 2 WHERE id IN (3, 4);

-- 摄像头5独立实例
UPDATE app_roomip SET yolo_pool_id = 0 WHERE id = 5;

-- 摄像头6使用全局默认池
UPDATE app_roomip SET yolo_pool_id = NULL WHERE id = 6;
```

#### 示例3：查看当前配置

```sql
-- 查看所有摄像头的池配置
SELECT
    id,
    fjmc AS '摄像头名称',
    CASE
        WHEN yolo_pool_id IS NULL THEN '全局默认池'
        WHEN yolo_pool_id = 0 THEN '独立实例'
        ELSE CONCAT('池', yolo_pool_id)
    END AS '池配置'
FROM app_roomip
ORDER BY yolo_pool_id, id;
```

#### 示例4：统计每个池的摄像头数量

```sql
SELECT
    CASE
        WHEN yolo_pool_id IS NULL THEN '全局默认池'
        WHEN yolo_pool_id = 0 THEN '独立实例'
        ELSE CONCAT('池', yolo_pool_id)
    END AS pool_name,
    COUNT(*) AS camera_count
FROM app_roomip
GROUP BY yolo_pool_id
ORDER BY yolo_pool_id;
```

---

### 步骤3：API接口支持（如果使用API获取摄像头）

如果你的摄像头配置来自API接口（`camera_api_url`），确保接口返回数据包含 `yolo_pool_id` 字段：

**API返回示例**：

```json
{
  "code": 0,
  "msg": "成功",
  "data": [
    {
      "id": 1,
      "fjmc": "大厅监控",
      "rtspssl": "rtsp://...",
      "yolo_pool_id": 1
    },
    {
      "id": 2,
      "fjmc": "走廊监控",
      "rtspssl": "rtsp://...",
      "yolo_pool_id": 1
    }
  ]
}
```

**如果API不返回此字段**：系统会使用默认全局池或独立实例（取决于config.json配置）。

---

## config.json配置

### 完整配置示例

```json
{
  "YOLO_POOL_CONFIG": {
    "enabled": true,
    "pool_size": 2,
    "detector_type": "adaptive",
    "timeout": 30
  },
  "ADAPTIVE_DETECTION_CONFIG": {
    "enabled": true,
    "model_dir": "models",
    "conf_threshold": 0.5,
    "iou_threshold": 0.4,
    "force_engine": null,
    "num_threads": null
  }
}
```

### 配置项说明

| 配置项 | 说明 | 默认值 |
|-------|------|--------|
| `enabled` | 是否启用YOLO池功能 | `true` |
| `pool_size` | 全局默认池的实例数量 | `2` |
| `detector_type` | 检测器类型：`adaptive`（自适应）或 `traditional`（传统） | `adaptive` |
| `timeout` | 等待空闲YOLO实例的超时时间（秒） | `30` |

**注意**：
- `pool_size` 只影响 `yolo_pool_id=NULL` 的摄像头
- 数据库配置的池（`yolo_pool_id > 0`）固定为 `pool_size=1`

---

## 使用示例

### 示例1：2个摄像头共享1个YOLO实例

**数据库配置**：

```sql
UPDATE app_roomip SET yolo_pool_id = 1 WHERE id IN (1, 2);
```

**效果**：
- 摄像头1和2使用同一个YOLO实例
- 当摄像头1检测时，摄像头2需要等待
- 节省内存：从 ~1GB 降到 ~500MB

**日志输出**：

```
[INFO] YOLO池配置分析: 需要创建池ID=[1], 需要默认池=False
[INFO] 已创建YOLO池 ID=1: YoloDetectorPool(size=1, type=adaptive, available=1, detections=0)
[INFO] 摄像头 大厅监控 (ID: 1) 使用YOLO配置: 池1
[INFO] 摄像头 走廊监控 (ID: 2) 使用YOLO配置: 池1
[INFO] ✓ YOLO池检测结果 - 摄像头: 大厅监控, 人数: 3
```

---

### 示例2：4个摄像头分成2组，每组共享1个实例

**数据库配置**：

```sql
-- 摄像头1-2使用池1
UPDATE app_roomip SET yolo_pool_id = 1 WHERE id IN (1, 2);

-- 摄像头3-4使用池2
UPDATE app_roomip SET yolo_pool_id = 2 WHERE id IN (3, 4);
```

**效果**：
- 总共创建2个YOLO实例
- 摄像头1和2共享实例A
- 摄像头3和4共享实例B
- 节省内存：从 ~2GB 降到 ~1GB

**日志输出**：

```
[INFO] YOLO池配置分析: 需要创建池ID=[1, 2], 需要默认池=False
[INFO] 已创建YOLO池 ID=1: YoloDetectorPool(size=1, ...)
[INFO] 已创建YOLO池 ID=2: YoloDetectorPool(size=1, ...)
[INFO] 摄像头 大厅监控 (ID: 1) 使用YOLO配置: 池1
[INFO] 摄像头 走廊监控 (ID: 2) 使用YOLO配置: 池1
[INFO] 摄像头 停车场监控 (ID: 3) 使用YOLO配置: 池2
[INFO] 摄像头 后门监控 (ID: 4) 使用YOLO配置: 池2
```

---

### 示例3：混合模式（部分共享，部分独立）

**数据库配置**：

```sql
-- 摄像头1-2共享池1
UPDATE app_roomip SET yolo_pool_id = 1 WHERE id IN (1, 2);

-- 摄像头3独立实例（高优先级摄像头）
UPDATE app_roomip SET yolo_pool_id = 0 WHERE id = 3;

-- 摄像头4-5使用全局默认池
UPDATE app_roomip SET yolo_pool_id = NULL WHERE id IN (4, 5);
```

**config.json**：

```json
{
  "YOLO_POOL_CONFIG": {
    "enabled": true,
    "pool_size": 1
  }
}
```

**效果**：
- 摄像头1-2共享池1（1个实例）
- 摄像头3独立实例（1个实例）
- 摄像头4-5共享默认池（1个实例）
- 总共3个YOLO实例

---

## 故障排查

### 问题1：摄像头检测等待时间过长

**症状**：日志出现 `YOLO实例 0 等待时间: 5.2秒`

**原因**：池太小，多个摄像头竞争同一个实例

**解决方案**：

1. **增加池中实例数**（需要修改代码，当前固定为1）
2. **拆分摄像头到多个池**

```sql
-- 将10个摄像头分到3个池
UPDATE app_roomip SET yolo_pool_id = 1 WHERE id IN (1, 2, 3);
UPDATE app_roomip SET yolo_pool_id = 2 WHERE id IN (4, 5, 6);
UPDATE app_roomip SET yolo_pool_id = 3 WHERE id IN (7, 8, 9, 10);
```

---

### 问题2：数据库字段不存在

**症状**：启动时报错 `Unknown column 'yolo_pool_id'`

**解决方案**：

1. 执行SQL迁移脚本：

```bash
mysql -u root -p caseappdb < sql/add_yolo_pool_id.sql
```

2. 检查字段是否存在：

```sql
DESCRIBE app_roomip;
```

---

### 问题3：API返回的摄像头没有yolo_pool_id字段

**症状**：摄像头都使用默认池或独立实例

**解决方案**：

1. **修改API接口**，返回 `yolo_pool_id` 字段
2. **使用数据库存储摄像头配置**，而不是API接口
3. **接受当前行为**，API摄像头使用默认池

---

### 问题4：查看YOLO池使用统计

**查询当前系统中的池配置**：

系统会在启动时输出：

```
[INFO] YOLO池配置分析: 需要创建池ID=[1, 2], 需要默认池=True
[INFO] 已创建默认YOLO池: YoloDetectorPool(size=2, type=adaptive, available=2, detections=0)
[INFO] 已创建YOLO池 ID=1: YoloDetectorPool(size=1, ...)
[INFO] 已创建YOLO池 ID=2: YoloDetectorPool(size=1, ...)
```

**查看池的实时统计**（需要添加API接口）：

```python
# 在代码中添加
pool_stats = monitor_system.yolo_pools[1].get_stats()
print(pool_stats)
# 输出：
# {
#     'pool_size': 1,
#     'detector_type': 'adaptive',
#     'available_instances': 0,  # 当前空闲实例数
#     'total_detections': 523,   # 总检测次数
#     'avg_wait_time': 0.15,     # 平均等待时间
#     'max_wait_time': 2.3,      # 最大等待时间
#     'instance_usage': [523],   # 实例使用次数
#     'uptime_seconds': 3600
# }
```

---

## 性能调优建议

### CPU核心数与池大小

| CPU核心数 | 摄像头数量 | 建议池数量 | 每池实例数 |
|----------|----------|----------|-----------|
| 4核 | 2-4 | 1个池 | 1实例 |
| 8核 | 5-10 | 2-3个池 | 1实例 |
| 16核 | 10+ | 3-5个池 | 1实例 |

### 检测频率与共享比例

| 检测频率 | 共享比例建议 |
|---------|------------|
| 持续检测（1秒/次） | 2-3个摄像头共享1个实例 |
| 中等频率（5秒/次） | 5-8个摄像头共享1个实例 |
| 低频率（10秒+/次） | 所有摄像头共享1个实例 |

### 监控等待时间

**正常范围**：
- 平均等待时间 < 0.5秒：性能良好
- 平均等待时间 0.5-2秒：可接受
- 平均等待时间 > 2秒：建议增加池或拆分摄像头

**查看日志**：

```
[WARNING] YOLO实例 0 等待时间: 2.1秒  # 如果频繁出现，需要优化
```

---

## 常见配置模板

### 模板1：小型系统（2-4个摄像头）

```sql
-- 所有摄像头共享1个实例
UPDATE app_roomip SET yolo_pool_id = 1;
```

### 模板2：中型系统（5-10个摄像头）

```sql
-- 分成2-3个池
UPDATE app_roomip SET yolo_pool_id = 1 WHERE id IN (1, 2, 3);
UPDATE app_roomip SET yolo_pool_id = 2 WHERE id IN (4, 5, 6);
UPDATE app_roomip SET yolo_pool_id = 3 WHERE id IN (7, 8, 9, 10);
```

### 模板3：大型系统（10+个摄像头）

```sql
-- 按区域分组
UPDATE app_roomip SET yolo_pool_id = 1 WHERE gnslx = '一楼大厅';  -- 一楼摄像头共享池1
UPDATE app_roomip SET yolo_pool_id = 2 WHERE gnslx = '二楼走廊';  -- 二楼摄像头共享池2
UPDATE app_roomip SET yolo_pool_id = 3 WHERE gnslx = '停车场';    -- 停车场摄像头共享池3
UPDATE app_roomip SET yolo_pool_id = 0 WHERE gnslx = '前台';      -- 前台独立实例（高优先级）
```

---

## 总结

**推荐配置流程**：

1. ✅ 执行SQL迁移脚本添加字段
2. ✅ 根据业务需求配置每个摄像头的 `yolo_pool_id`
3. ✅ 在 `config.json` 中启用YOLO池功能
4. ✅ 启动系统，观察日志确认池创建成功
5. ✅ 监控等待时间，必要时调整池配置

**遇到问题？**

- 查看日志文件：`logs/rtsp_monitor.log`
- 检查数据库配置：`SELECT * FROM app_roomip`
- 参考故障排查章节

---

## 附录

### A. 完整SQL脚本

参见：`sql/add_yolo_pool_id.sql`

### B. 代码文件清单

- `core/yolo_pool.py` - YOLO实例池实现
- `main.py` - 多池管理逻辑
- `camera_monitor.py` - 摄像头使用池
- `database.py` - 读取yolo_pool_id字段

### C. 相关配置项

- `YOLO_POOL_CONFIG` - YOLO池全局配置
- `ADAPTIVE_DETECTION_CONFIG` - 自适应检测器配置
- `app_roomip.yolo_pool_id` - 摄像头池ID字段

---

**更新日期**：2026-01-13
**版本**：v1.0
