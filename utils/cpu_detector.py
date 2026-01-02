"""
CPU 自动检测模块
运行时检测 CPU 类型，自动选择最优推理引擎
- Intel CPU → OpenVINO
- AMD/其他 CPU → ONNX Runtime
"""
import platform
import subprocess
import logging
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class InferenceEngine(Enum):
    """推理引擎类型"""
    OPENVINO = "openvino"  # Intel CPU 专用
    ONNX_RUNTIME = "onnx"  # AMD/其他 CPU


class CPUDetector:
    """CPU 自动检测器"""

    def __init__(self):
        self.cpu_brand = None
        self.cpu_model = None
        self.cpu_cores = None
        self.recommended_engine = None
        self.recommended_threads = None

    def detect(self) -> Tuple[str, InferenceEngine, int]:
        """
        自动检测 CPU 并推荐推理引擎

        Returns:
            (CPU型号, 推荐引擎, 推荐线程数)
        """
        # 检测 CPU 信息
        self._detect_cpu_info()

        # 推荐推理引擎
        self._recommend_engine()

        # 推荐线程数
        self._recommend_threads()

        logger.info("=" * 70)
        logger.info("CPU 自动检测结果")
        logger.info("=" * 70)
        logger.info(f"CPU 型号: {self.cpu_model}")
        logger.info(f"CPU 品牌: {self.cpu_brand}")
        logger.info(f"CPU 核心数: {self.cpu_cores}")
        logger.info(f"推荐引擎: {self.recommended_engine.value}")
        logger.info(f"推荐线程数: {self.recommended_threads}")
        logger.info("=" * 70)

        return self.cpu_model, self.recommended_engine, self.recommended_threads

    def _detect_cpu_info(self):
        """检测 CPU 详细信息"""
        import os
        system = platform.system()

        # 获取核心数
        self.cpu_cores = os.cpu_count() or 4

        # 根据操作系统检测 CPU 型号
        if system == "Windows":
            self._detect_windows_cpu()
        elif system == "Linux":
            self._detect_linux_cpu()
        else:
            logger.warning(f"不支持的操作系统: {system}")
            self.cpu_brand = "Unknown"
            self.cpu_model = "Unknown CPU"

    def _detect_windows_cpu(self):
        """Windows 系统 CPU 检测"""
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True,
                text=True,
                encoding='gbk',
                timeout=5
            )
            cpu_name = result.stdout.strip().split('\n')[1].strip()
            self.cpu_model = cpu_name

            # 判断品牌
            if 'Intel' in cpu_name:
                self.cpu_brand = 'Intel'
            elif 'AMD' in cpu_name:
                self.cpu_brand = 'AMD'
            else:
                self.cpu_brand = 'Unknown'

        except Exception as e:
            logger.warning(f"无法获取 Windows CPU 信息: {e}")
            self._detect_cpu_fallback()

    def _detect_linux_cpu(self):
        """Linux 系统 CPU 检测"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_name = line.split(':')[1].strip()
                        self.cpu_model = cpu_name

                        # 判断品牌
                        if 'Intel' in cpu_name:
                            self.cpu_brand = 'Intel'
                        elif 'AMD' in cpu_name:
                            self.cpu_brand = 'AMD'
                        else:
                            self.cpu_brand = 'Unknown'
                        break
        except Exception as e:
            logger.warning(f"无法获取 Linux CPU 信息: {e}")
            self._detect_cpu_fallback()

    def _detect_cpu_fallback(self):
        """后备检测方法"""
        try:
            # 使用 platform 模块
            processor = platform.processor()
            self.cpu_model = processor

            if 'Intel' in processor:
                self.cpu_brand = 'Intel'
            elif 'AMD' in processor:
                self.cpu_brand = 'AMD'
            else:
                self.cpu_brand = 'Unknown'
        except Exception as e:
            logger.error(f"CPU 检测失败: {e}")
            self.cpu_brand = 'Unknown'
            self.cpu_model = 'Unknown CPU'

    def _recommend_engine(self):
        """根据 CPU 品牌推荐推理引擎"""
        if self.cpu_brand == 'Intel':
            self.recommended_engine = InferenceEngine.OPENVINO
            logger.info("检测到 Intel CPU，推荐使用 OpenVINO（针对 Intel 优化，速度快 2-3 倍）")
        else:
            self.recommended_engine = InferenceEngine.ONNX_RUNTIME
            if self.cpu_brand == 'AMD':
                logger.info("检测到 AMD CPU，推荐使用 ONNX Runtime")
            else:
                logger.info(f"检测到 {self.cpu_brand} CPU，推荐使用 ONNX Runtime（通用方案）")

    def _recommend_threads(self):
        """推荐线程数"""
        # 推荐使用 CPU 核心数 - 2（留给系统）
        self.recommended_threads = max(1, self.cpu_cores - 2)

        # 如果核心数很多，不要用满所有核心
        if self.cpu_cores > 16:
            self.recommended_threads = min(self.recommended_threads, 14)

    @staticmethod
    def check_engine_availability(engine: InferenceEngine) -> bool:
        """
        检查推理引擎是否可用

        Args:
            engine: 推理引擎类型

        Returns:
            是否可用
        """
        if engine == InferenceEngine.OPENVINO:
            try:
                import openvino as ov
                logger.info(f"OpenVINO 可用，版本: {ov.__version__}")
                return True
            except ImportError:
                logger.warning("OpenVINO 未安装，请运行: pip install openvino")
                return False

        elif engine == InferenceEngine.ONNX_RUNTIME:
            try:
                import onnxruntime as ort
                logger.info(f"ONNX Runtime 可用，版本: {ort.__version__}")
                return True
            except ImportError:
                logger.warning("ONNX Runtime 未安装，请运行: pip install onnxruntime")
                return False

        return False


# 全局单例
_cpu_detector = None


def get_cpu_detector() -> CPUDetector:
    """获取 CPU 检测器单例"""
    global _cpu_detector
    if _cpu_detector is None:
        _cpu_detector = CPUDetector()
        _cpu_detector.detect()
    return _cpu_detector


def get_recommended_engine() -> InferenceEngine:
    """获取推荐的推理引擎"""
    detector = get_cpu_detector()
    return detector.recommended_engine


def get_recommended_threads() -> int:
    """获取推荐的线程数"""
    detector = get_cpu_detector()
    return detector.recommended_threads


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    detector = CPUDetector()
    cpu_model, engine, threads = detector.detect()

    print(f"\n推荐配置:")
    print(f"  推理引擎: {engine.value}")
    print(f"  线程数: {threads}")

    # 检查引擎可用性
    print(f"\n引擎可用性检查:")
    print(f"  OpenVINO: {CPUDetector.check_engine_availability(InferenceEngine.OPENVINO)}")
    print(f"  ONNX Runtime: {CPUDetector.check_engine_availability(InferenceEngine.ONNX_RUNTIME)}")
