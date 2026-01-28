# IMPLEMENTATION_PLAN.md

本文件是执行状态的唯一来源。

- 所有当前任务、优先级和下一步操作都必须体现在此处。
- 不得假定本文件之外存在任何执行状态。
- 本文件可以在任何时候被重写、重新排序或重新生成。
- AGENTS.md 定义执行规则；本文件仅定义当前状态。
- 文件其余部分只能包含当前的执行计划。
- 不得包含历史日志或叙述性说明。

---

## 当前目标（Current Objective）

消除 specs 与现有代码之间的行为偏差（配置值未生效、功能缺失），并为缺失测试覆盖的核心模块补齐回压机制。

---

## 可执行任务（Executable Tasks · BUILD 阶段唯一合法来源）

> BUILD 阶段 **只能** 从本区选择任务
> 本区任务必须：
> - 已明确 Task Kind
> - 可在一次构建循环中完成
> - 具备明确的验证方式（tests / backpressure）

### 任务列表（按优先级排序）

1. ✅ `[CHANGE]` 将 rtsp_timeout 配置值应用到实际代码中
   - 已完成：OpenCV 和 PyAV 超时均改为读取 `config.get('rtsp_timeout', 30)`

2. ✅ `[CHANGE]` 将缓冲区清理参数从硬编码改为读取配置值
   - 已完成：清理方法调用处改为读取 `buffer_flush_max_frames` 和 `buffer_flush_after_yolo_frames`

3. ✅ `[CHANGE]` JPEG 保存质量从硬编码 85 改为可配置（默认 95）
   - 已完成：新增 `jpeg_quality` 配置参数，image_storage.py 三处保存均使用该值

4. ✅ `[CHANGE]` 扩展自适应检测器的模型发现规则以支持 YOLOv8
   - 已完成：OpenVINO 和 ONNX 搜索模式均扩展为同时支持 yolo11* 和 yolov8* 模式

5. ✅ `[NEW]` 为 image_storage 添加图片最大宽度限制（默认 1920px）
   - 已完成：新增 max_image_width 配置参数，超宽图片自动等比缩放

6. ✅ `[CHANGE]` 恢复 main.py 中被注释掉的许可证验证逻辑
   - 已完成：取消注释许可证检查代码，启动时验证 license.dat

7. ✅ `[NEW]` 实现 /api/cameras 接口（获取摄像头列表）
   - 已完成：在 person_management_api.py 添加 /api/cameras 端点

8. ✅ `[NEW]` 实现 /api/statistics 接口（获取统计数据）
   - 已完成：在 person_management_api.py 添加 /api/statistics 端点

9. ✅ `[TEST]` 为人员检测流水线（YOLO 推理）补充单元测试
   - 已完成：新增 test_person_detection.py，22 个测试用例全部通过

10. ✅ `[TEST]` 为数据库操作补充单元测试
    - 已完成：新增 test_database.py，19 个测试用例全部通过

11. ✅ `[TEST]` 为背景差分变化检测补充单元测试
    - 已完成：新增 test_image_detection.py，23 个测试用例全部通过

12. ✅ `[TEST]` 为 Web API 接口补充集成测试
    - 已完成：新增 test_web_api.py，23 个测试用例全部通过

13. ✅ `[TEST]` 为许可证验证系统补充单元测试
    - 已完成：新增 test_license.py，19 个测试用例全部通过

> 规则：
> - 每个任务必须且只能标注一个 Task Kind
> - 每次 BUILD 只能选择 **一个** 任务
> - 任务完成后必须更新本区状态

---

## 延后处理 / 不可执行任务（Deferred / Out of Scope）

> 本区任务 **明确不进入 BUILD 阶段执行**

### 技术债 / 风险记录

- `[DEBT]` 内存优化声明（10 路摄像头 YOLO 池 ≈ 1GB）已在文档中标注，但无性能基准测试代码验证
- `[DEBT]` 系统性能目标（10fps / 2GB / 30% CPU / 24h 稳定）缺乏自动化基准测试
- `[DEBT]` `person_management_api.py` 与 `reid_web_app.py` 存在部分路由功能重叠，未来可考虑合并
- `[DEBT]` ReID 功能默认关闭（`config.json` 中 `enabled: false`），跨镜头追踪的集成测试依赖外部模型文件和数据库环境
- `[DEBT]` 多进程 YOLO 池（`yolo_process_pool.py`）无独立测试覆盖，仅线程池有测试

### 未决问题（需要澄清后才能进入可执行区）

- PTS 时间戳策略默认为 `realtime` 而非 `pts_auto`——是否为有意设计决策？如需变更应调整 specs
- `image_storage.py` 文件命名格式为 `{camera_id}_{safe_name}_{timestamp}.jpg`，与 specs 中的 `{camera_id}_{timestamp}.jpg` 略有不同——是否视为偏差？

---

## 规划备注（仅用于 PLAN 阶段）

> 本区仅用于记录规划发现
> BUILD 阶段不得依据本区内容执行任何任务

- 9 个 specs 主题中，背景差分变化检测、跨镜头 ReID 追踪、多摄像头并发与 YOLO 池的**功能实现**已完整覆盖
- 主要偏差集中在：配置值未应用到代码（硬编码覆盖 config）、少量缺失端点、许可证验证被禁用
- 测试覆盖严重不足：9 个 specs 主题中仅 2 个有测试（视频流、YOLO 池），其余 7 个无任何回压机制
- 可执行任务共 13 项：5 项 CHANGE（行为对齐）+ 3 项 NEW（缺失能力）+ 5 项 TEST（回压补充）
