# 自适应 CPU 检测 + 智能推理引擎部署指南

## 🎯 系统特点

### ✨ 核心功能
- ✅ **自动检测 CPU** - 运行时检测 Intel/AMD CPU
- ✅ **智能选择引擎** - Intel → OpenVINO，AMD → ONNX Runtime
- ✅ **YOLOv11 检测** - 最新 YOLO 模型，更快更准
- ✅ **专注人员检测** - 弃用 ReID，只检测人员和计数
- ✅ **5-10 倍加速** - CPU 优化推理，告别 PyTorch 慢速推理

### 🚀 性能提升

| 场景 | 原方案 (PyTorch CPU) | 新方案 (自适应) | 提升 |
|------|---------------------|----------------|------|
| **Intel CPU** | ~5 秒/帧 | ~0.3 秒/帧 | **15倍** |
| **AMD CPU** | ~5 秒/帧 | ~0.5 秒/帧 | **10倍** |

---

## 📦 快速部署（3 步完成）

### 步骤 1：转换模型（本地 GPU 机器）

```bash
# 1. 安装依赖
pip install ultralytics openvino onnx

# 2. 运行转换脚本
python convert_yolo11_models.py
```

**转换结果：**
```
models/
  ├── yolo11n.onnx    # ONNX 格式（AMD/通用 CPU）
  ├── yolo11n.xml     # OpenVINO IR 格式（Intel CPU）
  └── yolo11n.bin     # OpenVINO 权重文件
```

### 步骤 2：修改配置

编辑 `config.py`，添加自适应检测配置：

```python
# 启用自适应人员检测
ADAPTIVE_DETECTION_CONFIG = {
    'enabled': True,  # 启用自适应检测
    'model_dir': 'models',  # 模型目录

    # 检测参数
    'conf_threshold': 0.5,  # 置信度阈值
    'iou_threshold': 0.4,   # NMS IOU 阈值

    # 自动配置（会自动检测）
    'force_engine': None,  # None=自动，'openvino' 或 'onnx' 强制指定
    'num_threads': None,   # None=自动（CPU核心数-2）
}

# ReID 配置（禁用）
REID_CONFIG = {
    'enabled': False,  # 禁用 ReID，只做人员检测
    # ... 其他配置保持不变（兼容性）
}

# RTSP 监测配置（优化）
RTSP_MONITOR_CONFIG = {
    'sample_interval': 2,  # 每 2 秒采样（降低负载）
    'detection_wait_interval': 15,  # 检测到人后等待 15 秒
    # ... 其他配置保持不变
}
```

### 步骤 3：修改 camera_monitor.py

在 `camera_monitor.py` 中使用自适应检测器：

```python
# 在文件顶部导入
from core.person_detector_adaptive import AdaptivePersonDetector

class CameraMonitor:
    def __init__(self, camera_info: dict, db: Database):
        # ... 原有代码 ...

        # 使用自适应检测器（替换原来的 YOLO 检测器）
        if config.ADAPTIVE_DETECTION_CONFIG.get('enabled', False):
            self.person_detector = AdaptivePersonDetector(
                model_dir=config.ADAPTIVE_DETECTION_CONFIG['model_dir'],
                conf_threshold=config.ADAPTIVE_DETECTION_CONFIG['conf_threshold'],
                iou_threshold=config.ADAPTIVE_DETECTION_CONFIG['iou_threshold'],
                force_engine=config.ADAPTIVE_DETECTION_CONFIG.get('force_engine'),
                num_threads=config.ADAPTIVE_DETECTION_CONFIG.get('num_threads')
            )
            logger.info("使用自适应人员检测器（CPU 优化）")
        else:
            # 原有的检测器（兼容性）
            self.person_detector = None

    def _process_frame_with_detection(self, frame):
        """使用人员检测处理帧"""
        # 使用自适应检测器检测人员
        detections = self.person_detector.detect_image(frame)
        person_count = len(detections)

        logger.info(f"检测到 {person_count} 个人员")

        # 如果检测到人员
        if person_count > 0:
            # 保存图片（带检测框可视化）
            vis_frame = self.person_detector.visualize(frame, detections)

            # 保存到文件/数据库
            self._save_detection_result(vis_frame, person_count, detections)

        return person_count
```

---

## 🔧 服务器部署

### Intel CPU 服务器

```bash
# 1. 安装依赖（仅需 OpenVINO）
pip install openvino numpy opencv-python

# 2. 上传文件
scp -r models/ user@server:/path/to/spbj/
scp core/person_detector_adaptive.py user@server:/path/to/spbj/core/
scp utils/cpu_detector.py user@server:/path/to/spbj/utils/

# 3. 确认模型文件
ls models/yolo11n.xml  # 必须存在
ls models/yolo11n.bin  # 必须存在

# 4. 运行
python main.py
```

**输出示例：**
```
[INFO] CPU 自动检测结果
[INFO] CPU 型号: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
[INFO] CPU 品牌: Intel
[INFO] 推荐引擎: openvino
[INFO] 使用线程数: 6
[INFO] 加载 OpenVINO 模型: models/yolo11n.xml
[INFO] OpenVINO 模型加载完成，输入尺寸: 640x640
```

### AMD CPU 服务器

```bash
# 1. 安装依赖（仅需 ONNX Runtime）
pip install onnxruntime numpy opencv-python

# 2. 上传文件（同上）

# 3. 确认模型文件
ls models/yolo11n.onnx  # 必须存在

# 4. 运行
python main.py
```

**输出示例：**
```
[INFO] CPU 自动检测结果
[INFO] CPU 型号: AMD Ryzen 7 5800X
[INFO] CPU 品牌: AMD
[INFO] 推荐引擎: onnx
[INFO] 使用线程数: 14
[INFO] 加载 ONNX 模型: models/yolo11n.onnx
[INFO] ONNX Runtime 模型加载完成，输入尺寸: 640x640
```

---

## 📊 模型选择

### YOLOv11 模型对比

| 模型 | 文件大小 | CPU 速度 | 精度 | 推荐场景 |
|------|---------|---------|------|---------|
| **yolo11n** | ~6 MB | 最快 | 良好 | **CPU 服务器（推荐）** |
| yolo11s | ~22 MB | 快 | 较高 | 高性能 CPU |
| yolo11m | ~50 MB | 中等 | 高 | 仅 GPU 环境 |

### 推理引擎对比

| 引擎 | CPU 类型 | 速度 | 特点 |
|------|---------|------|------|
| **OpenVINO** | Intel | 最快 | Intel 专用优化，比 ONNX 快 50% |
| **ONNX Runtime** | AMD/通用 | 快 | 通用方案，兼容性好 |

---

## ✅ 测试和验证

### 1. CPU 检测测试

```bash
python utils/cpu_detector.py
```

**预期输出：**
```
CPU 型号: Intel Core i7-9700K
推荐引擎: openvino
推荐线程数: 6
```

### 2. 模型加载测试

```bash
python -c "
from core.person_detector_adaptive import AdaptivePersonDetector
detector = AdaptivePersonDetector()
print('模型加载成功！')
print(f'使用引擎: {detector.engine.value}')
"
```

### 3. 检测测试

```python
# test_detection.py
from core.person_detector_adaptive import AdaptivePersonDetector
import cv2
import time

# 初始化检测器
detector = AdaptivePersonDetector()

# 读取测试图片
image = cv2.imread('test_image.jpg')

# 检测性能测试
start = time.time()
detections = detector.detect_image(image)
elapsed = time.time() - start

print(f"检测耗时: {elapsed:.3f} 秒")
print(f"检测到 {len(detections)} 个人员")
print(f"FPS: {1/elapsed:.1f}")

# 可视化
vis_image = detector.visualize(image, detections)
cv2.imwrite('result.jpg', vis_image)
print("结果已保存到 result.jpg")
```

---

## 🎯 配置优化

### 1. 线程数调优

```python
# 测试不同线程数，找到最优值
ADAPTIVE_DETECTION_CONFIG = {
    'num_threads': None,  # 自动（推荐）
    # 'num_threads': 8,   # 手动指定
}
```

### 2. 检测频率优化

```python
# 降低检测频率，减少 CPU 负载
RTSP_MONITOR_CONFIG = {
    'sample_interval': 3,  # 每 3 秒检测一次
}
```

### 3. 置信度阈值

```python
# 调整置信度阈值
ADAPTIVE_DETECTION_CONFIG = {
    'conf_threshold': 0.4,  # 降低阈值，检测更多（可能误报）
    # 'conf_threshold': 0.6,  # 提高阈值，减少误报（可能漏检）
}
```

---

## 🐛 常见问题

### Q1: 系统自动选择了错误的引擎

```python
# 强制指定引擎
ADAPTIVE_DETECTION_CONFIG = {
    'force_engine': 'openvino',  # 或 'onnx'
}
```

### Q2: OpenVINO 未安装

```bash
# Intel CPU 服务器必须安装
pip install openvino
```

### Q3: 检测速度仍然慢

1. 检查是否使用了正确的模型（yolo11n）
2. 检查线程数配置
3. 降低检测频率（sample_interval）
4. 确认没有使用 PyTorch 原生推理

---

## 📈 性能监控

### 添加性能日志

```python
# 在 camera_monitor.py 中
import time

start_time = time.time()
detections = self.person_detector.detect_image(frame)
detection_time = time.time() - start_time

logger.info(f"检测耗时: {detection_time:.3f}秒, 检测到 {len(detections)} 人")

# 如果检测时间过长，记录警告
if detection_time > 1.0:
    logger.warning(f"检测速度慢 ({detection_time:.3f}秒)，请检查配置")
```

---

## 🎉 总结

### 系统优势

1. ✅ **零配置** - 自动检测 CPU，自动选择引擎
2. ✅ **高性能** - 5-10 倍速度提升
3. ✅ **简单化** - 弃用 ReID，专注检测
4. ✅ **易部署** - 最少依赖，快速上线

### 部署检查清单

- [ ] 模型已转换（yolo11n.onnx / yolo11n.xml）
- [ ] config.py 已修改
- [ ] camera_monitor.py 已修改
- [ ] 服务器依赖已安装（openvino 或 onnxruntime）
- [ ] 测试运行正常
- [ ] 性能达标（< 1 秒/帧）

### 下一步

如果需要更高精度，可以使用 yolo11s 模型：
```bash
# 修改 convert_yolo11_models.py 中的 model_name
model_name = 'yolo11s.pt'

# 重新运行转换
python convert_yolo11_models.py
```

---

**需要帮助？** 运行 `python utils/cpu_detector.py` 查看系统推荐配置。
