# CPU 服务器优化部署指南

## 📋 目标
将识别速度提升 **5-10 倍**，通过使用 ONNX Runtime 替代 PyTorch 原生推理。

---

## 🚀 快速部署（3步完成）

### 步骤 1：安装依赖

```bash
# 安装 ONNX Runtime（CPU 版本）
pip install onnxruntime

# 安装 ONNX（用于模型转换）
pip install onnx
```

### 步骤 2：转换模型

```bash
# 在本地开发环境（有 GPU 的机器）运行
cd D:\work\spbj\spbj
python convert_models_to_onnx.py
```

**转换结果：**
- `models/yolov8n.onnx` - YOLO 检测模型（推荐 nano 版本）
- `models/osnet_x0_25.onnx` - ReID 模型（最轻量）
- `models/osnet_x0_5.onnx` - ReID 模型（平衡）

### 步骤 3：修改配置

编辑 `config.py`:

```python
# CPU 优化配置
REID_CONFIG = {
    'enabled': True,  # 启用 ReID
    'device': 'cpu',  # 使用 CPU
    'use_onnx': True,  # *** 新增：使用 ONNX 推理 ***

    # 模型配置（使用最轻量模型）
    'yolo_model_name': 'yolov8n.onnx',  # nano 版本（最快）
    'reid_model_name': 'osnet_x0_25.onnx',  # 最轻量（最快）

    # 批处理大小（CPU 推荐小批量）
    'detection_batch_size': 2,  # YOLO 批处理
    'feature_batch_size': 4,     # ReID 批处理

    # 线程数（根据服务器 CPU 核心数调整）
    'num_threads': 16,  # 20 核 CPU 推荐 16-18 线程

    # 检测参数
    'detection_conf_threshold': 0.5,
    'detection_iou_threshold': 0.5,
}

# RTSP 监测配置（降低检测频率）
RTSP_MONITOR_CONFIG = {
    'sample_interval': 2,  # 每 2 秒采样一次（降低计算量）
    'detection_wait_interval': 15,  # 检测到人后等待 15 秒
    # ... 其他配置保持不变
}
```

---

## 📊 性能对比

| 配置 | YOLO 模型 | ReID 模型 | 单帧耗时 | 速度提升 |
|------|-----------|-----------|---------|---------|
| **原始（PyTorch CPU）** | yolov8m.pt | osnet_x1_0 | ~5-10秒 | 基准 |
| **ONNX 优化** | yolov8n.onnx | osnet_x0_25.onnx | ~0.5-1秒 | **5-10倍** |

---

## 🎯 模型选择建议

### YOLO 模型

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|---------|
| **yolov8n.onnx** | 最快 | 较低 | CPU服务器（推荐） |
| yolov8s.onnx | 快 | 中等 | 平衡性能 |
| yolov8m.onnx | 中等 | 高 | GPU环境 |

### ReID 模型

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|---------|
| **osnet_x0_25.onnx** | 最快 | 较低 | CPU服务器（推荐） |
| osnet_x0_5.onnx | 快 | 中等 | 平衡性能 |
| osnet_x0_75.onnx | 中等 | 高 | 精度优先 |
| osnet_x1_0.onnx | 慢 | 最高 | GPU环境 |

---

## 🔧 高级优化

### 1. 线程数优化

根据服务器 CPU 核心数调整：

```python
import os
num_cores = os.cpu_count()

# 推荐配置
REID_CONFIG['num_threads'] = max(1, num_cores - 2)
```

### 2. 批处理大小优化

CPU 推荐小批量：

```python
# 测试不同批处理大小，找到最优值
REID_CONFIG['detection_batch_size'] = 2  # 尝试 1, 2, 4
REID_CONFIG['feature_batch_size'] = 4    # 尝试 2, 4, 8
```

### 3. 降低检测频率

```python
RTSP_MONITOR_CONFIG = {
    'sample_interval': 3,  # 每 3 秒检测一次
    'detection_wait_interval': 20,  # 检测到人后等待 20 秒
}
```

### 4. INT8 量化（可选，进一步加速）

```python
# 转换模型时启用量化
# 需要额外工具：pip install onnx-simplifier
# 速度可再提升 20-30%，精度略有损失
```

---

## 📦 部署到服务器

### 1. 打包文件

```bash
# 需要上传到服务器的文件
models/
  ├── yolov8n.onnx          # YOLO 模型
  ├── osnet_x0_25.onnx      # ReID 模型
config.py                   # 修改后的配置
core/
  ├── detector_onnx.py      # ONNX 检测器
  ├── feature_extractor_onnx.py  # ONNX 特征提取器
```

### 2. 服务器安装依赖

```bash
# 仅需安装 ONNX Runtime（不需要 PyTorch）
pip install onnxruntime numpy opencv-python pillow
```

### 3. 修改代码使用 ONNX

需要修改以下文件：

#### `camera_monitor.py`

```python
# 导入 ONNX 检测器
from core.detector_onnx import PersonDetectorONNX

class CameraMonitor:
    def __init__(self, camera_info: dict, db: Database):
        # ...
        # 使用 ONNX 检测器
        if config.REID_CONFIG.get('use_onnx', False):
            from core.detector_onnx import PersonDetectorONNX
            model_path = f"models/{config.REID_CONFIG['yolo_model_name']}"
            self.person_detector = PersonDetectorONNX(
                model_path=model_path,
                conf_threshold=config.REID_CONFIG['detection_conf_threshold'],
                iou_threshold=config.REID_CONFIG['detection_iou_threshold'],
                num_threads=config.REID_CONFIG.get('num_threads', 16)
            )
        else:
            # 原有的 PyTorch 检测器
            from core.detector import PersonDetector
            self.person_detector = PersonDetector(...)
```

#### `pipeline/reid_pipeline.py`

```python
# 在 __init__ 中选择检测器
if Config.REID_CONFIG.get('use_onnx', False):
    from core.detector_onnx import PersonDetectorONNX
    from core.feature_extractor_onnx import ReIDFeatureExtractorONNX

    yolo_model = f"models/{Config.REID_CONFIG['yolo_model_name']}"
    reid_model = f"models/{Config.REID_CONFIG['reid_model_name']}"

    self.detector = PersonDetectorONNX(
        model_path=yolo_model,
        conf_threshold=self.config.DETECTION_CONF_THRESHOLD,
        iou_threshold=self.config.DETECTION_IOU_THRESHOLD,
        num_threads=Config.REID_CONFIG.get('num_threads', 16)
    )

    self.extractor = ReIDFeatureExtractorONNX(
        model_path=reid_model,
        num_threads=Config.REID_CONFIG.get('num_threads', 16)
    )
else:
    # 原有的 PyTorch 推理
    self.detector = PersonDetector(...)
    self.extractor = ReIDFeatureExtractor(...)
```

---

## ✅ 验证部署

### 1. 测试模型加载

```python
# test_onnx_inference.py
from core.detector_onnx import PersonDetectorONNX
from core.feature_extractor_onnx import ReIDFeatureExtractorONNX
import time

# 测试 YOLO
print("测试 YOLO 检测...")
detector = PersonDetectorONNX('models/yolov8n.onnx')
start = time.time()
results = detector.detect('test_image.jpg')
print(f"检测耗时: {time.time() - start:.2f} 秒")
print(f"检测到 {len(results)} 个人员")

# 测试 ReID
print("\n测试 ReID 特征提取...")
extractor = ReIDFeatureExtractorONNX('models/osnet_x0_25.onnx')
start = time.time()
feature = extractor.extract_feature('person_crop.jpg')
print(f"特征提取耗时: {time.time() - start:.2f} 秒")
print(f"特征维度: {feature.shape}")
```

### 2. 性能基准测试

```bash
python test_onnx_inference.py
```

**预期结果（CPU 服务器）：**
- YOLO 检测：< 0.5 秒/张
- ReID 特征提取：< 0.1 秒/张

---

## 🐛 常见问题

### Q1: ONNX Runtime 找不到模型

```bash
# 检查模型路径是否正确
ls -l models/*.onnx
```

### Q2: 性能没有明显提升

```python
# 检查是否真的在使用 ONNX
import onnxruntime as ort
print(ort.get_available_providers())
# 应该看到 ['CPUExecutionProvider']
```

### Q3: 内存占用过高

```python
# 减小批处理大小
REID_CONFIG['detection_batch_size'] = 1
REID_CONFIG['feature_batch_size'] = 2
```

---

## 📈 监控和调优

### 添加性能监控

```python
import time

# 在检测代码中添加计时
start_time = time.time()
detections = detector.detect(image_path)
detection_time = time.time() - start_time

logger.info(f"YOLO检测耗时: {detection_time:.3f}秒")
```

### 调优建议

1. **线程数**：从 CPU 核心数-2 开始，逐步调整
2. **批处理大小**：从 2 开始，逐步增加测试
3. **检测间隔**：根据实际需求调整（2-5 秒）

---

## 🎉 总结

按照本指南操作后，你的 CPU 服务器识别速度应该能达到：

- ✅ YOLO 检测：从 3-5 秒 → **0.3-0.5 秒** （10倍提升）
- ✅ ReID 特征提取：从 1-2 秒 → **0.1-0.2 秒** （10倍提升）
- ✅ 总体识别速度：**5-10 倍提升**

**关键要点：**
1. 使用 ONNX Runtime 替代 PyTorch
2. 选择最轻量的模型（yolov8n + osnet_x0_25）
3. 优化批处理和线程配置
4. 降低检测频率
