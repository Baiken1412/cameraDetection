# 跨镜头 ReID 追踪

## 关注主题（Topic of Concern）

基于特征提取的人员重识别（Re-Identification），实现跨摄像头的人员追踪与匹配。

---

## 需求描述

### 1. 特征提取

- 使用 OSNet_x1_0 模型提取人员特征向量
- 支持 ONNX 格式模型推理
- 支持多模型特征融合（ensemble）
- 支持 CUDA 设备加速

### 2. 相似度计算

- 支持余弦相似度（cosine）
- 支持欧氏距离（euclidean）
- 匹配阈值可配置（默认 0.7）
- 相似度阈值可配置（默认 0.7）

### 3. 人员聚类

- 基于图的人员聚类算法
- 跨摄像头人员分组
- 人员轨迹串联

### 4. 管理功能

- 锚点人员管理（anchor_manager）
- 手动确认/标注（manual_confirmation）
- 数据库存储确认结果（db_confirmation_manager）
- 待处理队列管理（pending_queue）

### 5. 数据存储

- 特征向量存储到数据库
- 裁剪后的人员图片存储到文件系统
- 匹配结果存储

### 6. Web 界面

- ReID 专用 Web 应用（`reid_web_app.py`）
- 人员分组展示
- 手动确认界面

---

## 配置参数

```json
{
  "REID_CONFIG": {
    "enabled": false,
    "match_threshold": 0.7,
    "similarity_threshold": 0.7,
    "detection_conf_threshold": 0.5,
    "detection_iou_threshold": 0.5,
    "reid_model_name": "osnet_x1_0",
    "reid_pretrained": true,
    "similarity_metric": "cosine",
    "device": "cuda",
    "input_image_dir": "data/input_images",
    "cropped_dir": "data/cropped_persons",
    "results_dir": "data/results",
    "features_dir": "data/features"
  }
}
```

---

## 相关文件

- `core/feature_extractor.py` — 特征提取
- `core/feature_extractor_onnx.py` — ONNX 特征提取
- `core/ensemble_extractor.py` — 多模型融合
- `core/similarity.py` — 相似度计算
- `core/clustering.py` — 人员聚类
- `core/anchor_manager.py` — 锚点人员管理
- `core/manual_confirmation.py` — 手动确认
- `core/db_confirmation_manager.py` — 数据库确认管理
- `core/pending_queue.py` — 待处理队列
- `reid_integration.py` — ReID 系统集成层
- `reid_config.py` / `reid_config_adapter.py` — ReID 配置
- `reid_database.py` — ReID 特征数据库
- `reid_web_app.py` — ReID Web 界面
- `pipeline/reid_pipeline.py` — ReID 完整流水线
