# Web 界面与 API

## 关注主题（Topic of Concern）

基于 Flask 的 RESTful API 和 Web 管理界面，用于检测数据查询、摄像头管理和手动确认操作。

---

## 需求描述

### 1. REST API 接口

| 接口 | 方法 | 功能 |
|---|---|---|
| /api/cameras | GET | 获取摄像头列表 |
| /api/detections | GET | 获取检测记录（支持分页） |
| /api/detections/{id} | GET | 获取单条检测记录 |
| /api/confirm | POST | 手动确认记录 |
| /api/statistics | GET | 获取统计数据 |

### 2. Web Dashboard

- 主仪表盘（`templates/index.html`）
- 检测记录列表（`templates/records.html`）
- 结果展示页（`templates/results.html`）
- 人员分组详情（`templates/group_detail.html`）
- 错误页面（`templates/error.html`）

### 3. 静态资源

- CSS 样式表（`static/css/`）
- JavaScript 脚本（`static/js/`）
- 字体资源（`static/fonts/`）

### 4. ReID Web 界面

- 独立的 ReID Web 应用（`reid_web_app.py`）
- 人员重识别结果展示
- 手动确认与标注
- Flask 配置：host=0.0.0.0, port=5000

---

## 启动方式

```bash
# 启动管理 API
python person_management_api.py

# 启动 ReID Web 界面
python reid_web_app.py
```

---

## 相关文件

- `person_management_api.py` — Flask REST API 服务
- `reid_web_app.py` — ReID Web 界面
- `templates/` — HTML 模板目录
- `static/` — 静态资源目录
