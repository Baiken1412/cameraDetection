"""
多进程 YOLO 检测池
解决 Python GIL 导致的线程阻塞问题

核心思路：
- 将 YOLO 推理放到独立子进程中运行
- 子进程有自己的 GIL，不会阻塞主进程的 VideoStreamReader
- 使用 multiprocessing.Queue 进行帧数据传输
"""
import logging
import multiprocessing
import queue
import uuid
import time
import threading
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def _worker_process(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    detector_config: dict,
    ready_event: multiprocessing.Event
):
    """
    YOLO Worker 进程主循环

    在独立进程中运行，有自己的 GIL，不会阻塞主进程

    Args:
        worker_id: Worker 编号
        task_queue: 任务队列（接收帧数据）
        result_queue: 结果队列（返回检测结果）
        detector_config: 检测器配置
        ready_event: 就绪事件（通知主进程 Worker 已初始化完成）
    """
    import os

    # 设置进程名称（方便调试）
    try:
        import setproctitle
        setproctitle.setproctitle(f"yolo_worker_{worker_id}")
    except ImportError:
        pass

    logger.info(f"YOLO Worker #{worker_id} 进程启动中 (PID: {os.getpid()})...")

    # 在子进程中加载 YOLO 模型（每个进程独立加载一次）
    detector = None
    try:
        detector_type = detector_config.get('detector_type', 'adaptive')

        if detector_type == 'adaptive':
            from core.person_detector_adaptive import AdaptivePersonDetector
            detector = AdaptivePersonDetector(
                model_dir=detector_config.get('model_dir', 'models'),
                conf_threshold=detector_config.get('conf_threshold', 0.5),
                iou_threshold=detector_config.get('iou_threshold', 0.4),
                force_engine=detector_config.get('force_engine'),
                num_threads=detector_config.get('num_threads'),
                imgsz=detector_config.get('imgsz', 320)  # 默认320加速推理
            )
        else:
            from core.detector import PersonDetector
            detector = PersonDetector()

        logger.info(f"YOLO Worker #{worker_id} 模型加载完成")

    except Exception as e:
        logger.error(f"YOLO Worker #{worker_id} 模型加载失败: {e}", exc_info=True)
        ready_event.set()  # 即使失败也要通知主进程
        return

    # 通知主进程：Worker 已就绪
    ready_event.set()
    logger.info(f"YOLO Worker #{worker_id} 已就绪，等待任务...")

    # 统计信息
    detection_count = 0
    total_inference_time = 0.0

    # 主循环：持续处理任务
    while True:
        try:
            # 从队列获取任务（阻塞等待）
            task = task_queue.get(timeout=1.0)

            # 毒丸信号：收到 None 表示需要退出
            if task is None:
                logger.info(f"YOLO Worker #{worker_id} 收到退出信号，正在关闭...")
                break

            request_id, frame = task

            # 执行 YOLO 推理
            start_time = time.time()
            try:
                detections = detector.detect_image(frame)
                inference_time = time.time() - start_time

                # 更新统计
                detection_count += 1
                total_inference_time += inference_time

                # 返回结果到共享的结果队列
                result_queue.put({
                    'request_id': request_id,
                    'success': True,
                    'detections': detections,
                    'worker_id': worker_id,
                    'inference_time': inference_time
                })

                if detection_count % 100 == 0:
                    avg_time = total_inference_time / detection_count
                    logger.info(
                        f"YOLO Worker #{worker_id} 统计: "
                        f"已处理 {detection_count} 帧, "
                        f"平均推理时间 {avg_time*1000:.1f}ms"
                    )

            except Exception as e:
                logger.error(f"YOLO Worker #{worker_id} 推理失败: {e}")
                result_queue.put({
                    'request_id': request_id,
                    'success': False,
                    'error': str(e),
                    'worker_id': worker_id
                })

        except queue.Empty:
            # 队列超时，继续等待
            continue
        except Exception as e:
            logger.error(f"YOLO Worker #{worker_id} 发生错误: {e}", exc_info=True)
            continue

    # 清理
    logger.info(
        f"YOLO Worker #{worker_id} 已关闭. "
        f"共处理 {detection_count} 帧, "
        f"总推理时间 {total_inference_time:.1f}s"
    )


class YoloProcessPool:
    """
    多进程 YOLO 检测池

    特点：
    - 每个 Worker 是独立进程，有自己的 GIL
    - 主进程的 VideoStreamReader 不会被 YOLO 阻塞
    - 接口与 YoloDetectorPool 兼容，可无缝切换
    """

    def __init__(
        self,
        pool_size: int = 2,
        detector_type: str = 'adaptive',
        detector_config: dict = None,
        timeout: float = 30.0
    ):
        """
        初始化多进程 YOLO 池

        Args:
            pool_size: Worker 进程数量
            detector_type: 检测器类型 ('adaptive' 或 'default')
            detector_config: 检测器配置
            timeout: 默认超时时间（秒）
        """
        self.pool_size = pool_size
        self.timeout = timeout
        self.running = True

        # 检测器配置
        self.detector_config = detector_config or {}
        self.detector_config['detector_type'] = detector_type

        # 创建任务队列（所有 Worker 共享）
        self.task_queue = multiprocessing.Queue()

        # 创建结果队列（所有 Worker 共享，结果通过 request_id 区分）
        self.result_queue = multiprocessing.Queue()

        # 等待中的请求: request_id -> threading.Event
        self._pending_requests = {}
        self._pending_results = {}
        self._pending_lock = threading.Lock()

        # 启动结果收集线程
        self._result_collector_running = True
        self._result_collector = threading.Thread(
            target=self._collect_results,
            name="YoloResultCollector",
            daemon=True
        )
        self._result_collector.start()

        # 统计信息
        self.stats = {
            'total_detections': 0,
            'total_wait_time': 0.0,
            'max_wait_time': 0.0,
            'created_at': datetime.now()
        }

        # 启动 Worker 进程
        self.workers = []
        self.ready_events = []

        logger.info(f"正在启动 {pool_size} 个 YOLO Worker 进程...")

        for i in range(pool_size):
            ready_event = multiprocessing.Event()
            self.ready_events.append(ready_event)

            p = multiprocessing.Process(
                target=_worker_process,
                args=(i, self.task_queue, self.result_queue, self.detector_config, ready_event),
                name=f"YoloWorker-{i}"
            )
            p.daemon = True  # 主进程退出时自动终止子进程
            p.start()
            self.workers.append(p)
            logger.info(f"YOLO Worker #{i} 进程已启动 (PID: {p.pid})")

        # 等待所有 Worker 初始化完成
        logger.info("等待所有 YOLO Worker 初始化完成...")
        for i, event in enumerate(self.ready_events):
            if not event.wait(timeout=60):
                logger.warning(f"YOLO Worker #{i} 初始化超时")

        logger.info(f"YOLO 多进程池初始化完成: {pool_size} 个 Worker")

    def _collect_results(self):
        """
        结果收集线程：持续从结果队列获取结果，并通知等待的请求
        """
        while self._result_collector_running:
            try:
                result = self.result_queue.get(timeout=0.1)

                request_id = result.get('request_id')
                if request_id:
                    with self._pending_lock:
                        if request_id in self._pending_requests:
                            # 存储结果并通知等待线程
                            self._pending_results[request_id] = result
                            self._pending_requests[request_id].set()

            except queue.Empty:
                continue
            except Exception as e:
                if self._result_collector_running:
                    logger.error(f"结果收集线程异常: {e}")

    def detect(self, frame, timeout: float = None) -> List[Dict]:
        """
        检测图像中的人员（同步接口，兼容 YoloDetectorPool）

        Args:
            frame: BGR 图像 (numpy array)
            timeout: 超时时间（秒）

        Returns:
            检测结果列表 [{'bbox': [...], 'conf': 0.9, 'class_id': 0}, ...]
        """
        if not self.running:
            raise RuntimeError("YOLO 进程池已关闭")

        if timeout is None:
            timeout = self.timeout

        # 生成唯一请求 ID
        request_id = str(uuid.uuid4())

        # 创建等待事件
        wait_event = threading.Event()

        with self._pending_lock:
            self._pending_requests[request_id] = wait_event

        # 记录等待开始时间
        wait_start = time.time()

        # 发送任务到队列（只传 request_id 和 frame）
        try:
            self.task_queue.put((request_id, frame), timeout=5.0)
        except queue.Full:
            logger.error("YOLO 任务队列已满，检测请求被丢弃")
            with self._pending_lock:
                self._pending_requests.pop(request_id, None)
            return []

        # 等待结果
        try:
            if wait_event.wait(timeout=timeout):
                wait_time = time.time() - wait_start

                # 获取结果
                with self._pending_lock:
                    result = self._pending_results.pop(request_id, None)
                    self._pending_requests.pop(request_id, None)

                if result is None:
                    logger.error("YOLO 检测结果丢失")
                    return []

                # 更新统计
                self.stats['total_detections'] += 1
                self.stats['total_wait_time'] += wait_time
                self.stats['max_wait_time'] = max(self.stats['max_wait_time'], wait_time)

                if wait_time > 1.0:
                    logger.warning(
                        f"YOLO 检测等待时间较长: {wait_time:.2f}s "
                        f"(Worker #{result.get('worker_id', '?')})"
                    )

                if result.get('success'):
                    return result.get('detections', [])
                else:
                    logger.error(f"YOLO 检测失败: {result.get('error')}")
                    return []
            else:
                # 超时
                logger.error(f"YOLO 检测超时 ({timeout}s)")
                with self._pending_lock:
                    self._pending_requests.pop(request_id, None)
                    self._pending_results.pop(request_id, None)
                return []

        except Exception as e:
            logger.error(f"YOLO 检测异常: {e}")
            with self._pending_lock:
                self._pending_requests.pop(request_id, None)
                self._pending_results.pop(request_id, None)
            return []

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats['total_detections'] > 0:
            stats['avg_wait_time'] = stats['total_wait_time'] / stats['total_detections']
        else:
            stats['avg_wait_time'] = 0.0

        # 添加 Worker 状态
        stats['workers'] = []
        for i, p in enumerate(self.workers):
            stats['workers'].append({
                'id': i,
                'pid': p.pid,
                'alive': p.is_alive()
            })

        return stats

    def restart_workers(self):
        """
        重启所有 Worker 进程，释放长时间运行积累的内存。
        定时调用（如每天凌晨），不影响主进程和摄像头监测线程的正常运行。
        """
        logger.info("========== 开始重启 YOLO Worker 进程（定时清理内存） ==========")

        # 1. 发送毒丸信号，通知所有 Worker 退出
        for _ in range(len(self.workers)):
            try:
                self.task_queue.put(None, timeout=1.0)
            except Exception:
                pass

        # 2. 等待 Worker 进程退出（最多10秒）
        for i, p in enumerate(self.workers):
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"YOLO Worker #{i} 未能正常退出，强制终止")
                p.terminate()
                p.join(timeout=2)

        # 3. 通知所有正在等待的检测请求，让它们立即失败返回（避免永久阻塞）
        with self._pending_lock:
            for request_id, event in list(self._pending_requests.items()):
                self._pending_results[request_id] = {
                    'request_id': request_id,
                    'success': False,
                    'error': 'YOLO进程池重启中，请稍后重试'
                }
                event.set()
            self._pending_requests.clear()
            self._pending_results.clear()

        # 4. 清空任务队列和结果队列中的残余数据
        for q in (self.task_queue, self.result_queue):
            try:
                while True:
                    q.get_nowait()
            except Exception:
                pass

        # 5. 重新启动 Worker 进程
        self.workers = []
        self.ready_events = []

        logger.info(f"正在重新启动 {self.pool_size} 个 YOLO Worker 进程...")
        for i in range(self.pool_size):
            ready_event = multiprocessing.Event()
            self.ready_events.append(ready_event)

            p = multiprocessing.Process(
                target=_worker_process,
                args=(i, self.task_queue, self.result_queue, self.detector_config, ready_event),
                name=f"YoloWorker-{i}"
            )
            p.daemon = True
            p.start()
            self.workers.append(p)
            logger.info(f"YOLO Worker #{i} 重启完成 (PID: {p.pid})")

        # 6. 等待所有 Worker 初始化完成（最多60秒）
        for i, event in enumerate(self.ready_events):
            if not event.wait(timeout=60):
                logger.warning(f"YOLO Worker #{i} 重启后初始化超时")

        # 7. 重置统计信息
        self.stats = {
            'total_detections': 0,
            'total_wait_time': 0.0,
            'max_wait_time': 0.0,
            'created_at': datetime.now()
        }

        logger.info("========== YOLO Worker 进程重启完成 ==========")

    def shutdown(self):
        """关闭进程池"""
        if not self.running:
            return

        self.running = False
        self._result_collector_running = False
        logger.info("正在关闭 YOLO 进程池...")

        # 发送毒丸信号
        for i in range(len(self.workers)):
            try:
                self.task_queue.put(None, timeout=1.0)
            except:
                pass

        # 等待进程结束
        for i, p in enumerate(self.workers):
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"YOLO Worker #{i} 未能正常退出，强制终止")
                p.terminate()

        # 等待结果收集线程结束
        if self._result_collector.is_alive():
            self._result_collector.join(timeout=2)

        # 清理队列
        try:
            self.task_queue.close()
            self.result_queue.close()
        except:
            pass

        logger.info("YOLO 进程池已关闭")

    def __del__(self):
        """析构时确保资源释放"""
        try:
            self.shutdown()
        except:
            pass


# Windows 多进程支持
if __name__ == '__main__':
    # Windows 上需要这个保护
    multiprocessing.freeze_support()
