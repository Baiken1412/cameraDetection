# 人员检测与 YOLO 推理

## 关注主题（Topic of Concern）

基于 YOLO 深度学习模型的人员检测，包括模型加载、推理引擎自适应选择、检测参数配置。

---

## 需求描述

### 1. 模型支持

- 支持 YOLOv8n / YOLOv8s / YOLOv8m
- 支持 YOLOv11n / YOLOv11s
- 模型文件格式：ONNX
- 模型目录可配置（默认 `models/`）

### 2. 推理引擎自适应

- 自动检测 CPU 类型（Intel / AMD / 其他）
- Intel CPU 使用 OpenVINO 引擎（AVX-512、INT8 量化优化）
- AMD / 其他 CPU 使用 ONNX Runtime 引擎（多线程优化）
- 支持强制指定引擎（`force_engine: 'openvino' / 'onnx' / null`）
- 推理线程数可配置（`null` 为自动）

### 3. 检测参数

- 置信度阈值：0.5（≥50% 才认为检测到人）
- NMS IOU 阈值：0.4（重叠框合并）
- 输入图像尺寸：640px
- 仅检测 person 类别

### 4. 检测输出

- 返回边界框坐标 `[x1, y1, x2, y2]`
- 返回置信度分数
- 返回类别标签（person）

### 5. 性能指标

- 单帧检测延迟：200-500ms
- 检测准确率：> 90%
- 误报率：< 5%

---

## 配置参数

```json
{
  "ADAPTIVE_DETECTION_CONFIG": {
    "enabled": true,
    "model_dir": "models",
    "conf_threshold": 0.5,
    "iou_threshold": 0.4,
    "force_engine": null,
    "num_threads": null,
    "imgsz": 640
  }
}
```

---

## 相关文件

- `core/person_detector_adaptive.py` — 自适应 YOLO 检测器
- `core/detector.py` — 传统 YOLO 检测器
- `core/detector_onnx.py` — ONNX Runtime 检测器
- `core/detector_openvino.py` — OpenVINO 检测器
- `utils/cpu_detector.py` — CPU 类型检测
