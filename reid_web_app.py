"""
整合的 ReID Web 应用
包含所有 ReID 训练和管理页面，使用 caseapp 的配置
"""
import logging
import sys
import os
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from loguru import logger

# 导入 caseapp 的配置
import config as caseapp_config

# 注意：必须在添加 ReID 系统路径之前导入 caseapp 的 database 模块
# 否则 Python 会从 ReID 系统的 database 模块导入
import database as caseapp_database
from database import Database  # 从 caseapp 的 database 模块导入

# 导入 caseapp 的配置适配器
from reid_config_adapter import Config
from reid_database import ReIDDatabase
from reid_integration import ReIDIntegration, REID_AVAILABLE as REID_INTEGRATION_AVAILABLE, REID_IMPORT_ERROR

# ReID系统已整合到caseapp项目中，直接使用本地模块
# 导入 ReID 模块（pipeline和utils需要先复制到项目中）
try:
    from pipeline.reid_pipeline import ReIDPipeline
    from utils.file_handler import load_json
    # 注意：DatabaseManager需要从reid_database_modu导入（如果已复制）
    # 暂时保持原有逻辑，等待database模块整合完成
    try:
        # 尝试从reid_database_modu导入（整合后的路径）
        from reid_database_modu.database_manager import DatabaseManager as ReIDDatabaseManager
        logger.debug("成功从reid_database_modu导入DatabaseManager")
    except ImportError as db_import_error:
        logger.debug(f"从reid_database_modu导入DatabaseManager失败: {db_import_error}")
        # 如果reid_database_modu不存在，尝试从database导入（需要处理模块冲突）
        _caseapp_database_backup = sys.modules.get('database')
        if 'database' in sys.modules:
            del sys.modules['database']
        try:
            from database import DatabaseManager as ReIDDatabaseManager
            logger.debug("成功从database导入DatabaseManager（向后兼容）")
        except ImportError as db_import_error2:
            logger.warning(f"从database导入DatabaseManager也失败: {db_import_error2}")
            raise  # 重新抛出异常，让外层捕获
        finally:
            if _caseapp_database_backup:
                sys.modules['database'] = _caseapp_database_backup
    REID_AVAILABLE = True
    logger.info("✓ ReID模块导入成功（pipeline, utils, database_manager）")
except ImportError as e:
    logger.error(f"ReID模块导入失败: {e}", exc_info=True)
    REID_AVAILABLE = False
    ReIDPipeline = None
    ReIDDatabaseManager = None
except Exception as e:
    logger.error(f"ReID模块导入时发生未知错误: {e}", exc_info=True)
    REID_AVAILABLE = False
    ReIDPipeline = None
    ReIDDatabaseManager = None

# 确保所有必要的目录存在
Config.ensure_dirs()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 创建Flask应用
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['JSON_AS_ASCII'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 全局Pipeline实例
pipeline = None
processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': '',
    'error': None
}

# ReID数据库实例
reid_db = ReIDDatabase()

# caseapp 数据库实例（延迟初始化）
caseapp_db = None


def get_caseapp_db():
    """获取 caseapp 数据库实例（单例模式）"""
    global caseapp_db
    if caseapp_db is None:
        # 使用 caseapp 的 Database 类
        caseapp_db = caseapp_database.Database()
        caseapp_db.connect()
    return caseapp_db


def get_pipeline():
    """获取Pipeline实例（单例模式）"""
    global pipeline
    if pipeline is None and REID_AVAILABLE:
        try:
            logger.info(f"正在初始化ReID Pipeline，使用配置: INPUT_IMAGE_DIR={Config.INPUT_IMAGE_DIR}")
            pipeline = ReIDPipeline(config=Config)
            logger.info("ReID Pipeline初始化成功")
        except Exception as e:
            logger.error(f"ReID Pipeline初始化失败: {e}", exc_info=True)
            return None
    elif not REID_AVAILABLE:
        logger.error("ReID模块不可用，无法创建Pipeline")
        return None
    return pipeline


def load_results_from_database():
    """从数据库加载结果"""
    try:
        # 使用 caseapp 的 ReIDDatabase
        groups = reid_db.get_all_groups()
        
        # 获取所有人员
        all_persons = reid_db.get_all_persons()
        total_persons = len(all_persons)
        
        # 获取总图片数
        image_ids = set(p.image_id for p in all_persons if hasattr(p, 'image_id'))
        total_images = len(image_ids)
        
        # 构建结果
        results = {
            'timestamp': 'from database',
            'config': {
                'similarity_threshold': Config.SIMILARITY_THRESHOLD,
                'detection_conf_threshold': Config.DETECTION_CONF_THRESHOLD,
                'reid_model': Config.REID_MODEL_NAME
            },
            'total_images': total_images,
            'total_persons': total_persons,
            'total_groups': len(groups),
            'groups': groups,
            'loaded_from_database': True
        }
        
        logger.info(f"从数据库加载结果: {len(groups)}个分组, {total_persons}个人员")
        return results
    except Exception as e:
        logger.error(f"从数据库加载结果失败: {e}", exc_info=True)
        raise


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """主页 - 训练页面"""
    return render_template('index.html', config=Config)


@app.route('/results')
def results_page():
    """结果展示页面"""
    try:
        # 从数据库加载结果
        results = load_results_from_database()
        
        # 应用人工确认的合并操作
        pipeline_instance = get_pipeline()
        if pipeline_instance and hasattr(pipeline_instance, 'confirmation_manager'):
            try:
                results = pipeline_instance.confirmation_manager.apply_confirmations_to_results(results)
            except Exception as e:
                logger.warning(f"应用确认结果失败: {e}")
        
        # 动态添加命名信息
        for group in results.get('groups', []):
            if group.get('persons') and len(group['persons']) > 0:
                representative_pid = group['persons'][0]['person_id']
                group_name = reid_db.get_person_name(representative_pid)
                group['group_name'] = group_name
            else:
                group['group_name'] = ''
        
        return render_template('results.html', results=results, config=Config)
    except Exception as e:
        logger.error(f"加载结果页面失败: {e}", exc_info=True)
        return render_template('error.html', message=str(e))


@app.route('/group/<int:group_id>')
def group_detail(group_id):
    """分组详情页面"""
    try:
        results = load_results_from_database()
        
        # 应用人工确认
        pipeline_instance = get_pipeline()
        if pipeline_instance and hasattr(pipeline_instance, 'confirmation_manager'):
            try:
                results = pipeline_instance.confirmation_manager.apply_confirmations_to_results(results)
            except Exception as e:
                logger.warning(f"应用确认结果失败: {e}")
        
        # 动态添加命名信息
        for group in results.get('groups', []):
            if group.get('persons') and len(group['persons']) > 0:
                representative_pid = group['persons'][0]['person_id']
                group_name = reid_db.get_person_name(representative_pid)
                group['group_name'] = group_name
            else:
                group['group_name'] = ''
        
        # 查找指定的分组
        group = None
        for g in results['groups']:
            if g['group_id'] == group_id:
                group = g
                break
        
        if group is None:
            return render_template('error.html', message=f'分组 {group_id} 不存在')
        
        return render_template('group_detail.html', group=group, config=Config)
    except Exception as e:
        logger.error(f"加载分组详情失败: {e}", exc_info=True)
        return render_template('error.html', message=str(e))


# ==================== API 路由 ====================

@app.route('/api/process', methods=['POST', 'OPTIONS'])
def api_process():
    """启动ReID处理"""
    global processing_status
    
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    logger.info("收到处理请求")
    logger.info(f"请求方法: {request.method}, Content-Type: {request.content_type}")
    logger.info(f"请求数据: {request.get_data()}")
    
    if processing_status['is_processing']:
        logger.warning("处理正在进行中，拒绝新请求")
        return jsonify({
            'success': False,
            'message': '处理正在进行中，请稍后再试'
        }), 400
    
    if not REID_AVAILABLE:
        # 获取详细的错误信息
        error_message = 'ReID模块不可用，请检查ReID系统是否正确安装'
        error_details = []
        
        # 检查pipeline模块
        try:
            import pipeline.reid_pipeline
            error_details.append('✓ pipeline.reid_pipeline 模块存在')
        except ImportError as e:
            error_details.append(f'✗ pipeline.reid_pipeline 导入失败: {e}')
        
        # 检查utils模块
        try:
            import utils.file_handler
            error_details.append('✓ utils.file_handler 模块存在')
        except ImportError as e:
            error_details.append(f'✗ utils.file_handler 导入失败: {e}')
        
        # 检查reid_database_modu模块
        try:
            import reid_database_modu.database_manager
            error_details.append('✓ reid_database_modu.database_manager 模块存在')
        except ImportError as e:
            error_details.append(f'✗ reid_database_modu.database_manager 导入失败: {e}')
        
        logger.error(f"ReID模块不可用。详细信息:\n" + "\n".join(error_details))
        return jsonify({
            'success': False,
            'message': error_message,
            'details': error_details
        }), 500
    
    try:
        # 获取参数
        data = request.get_json() or {}
        threshold = float(data.get('threshold', Config.SIMILARITY_THRESHOLD))
        
        # 更新配置
        Config.SIMILARITY_THRESHOLD = threshold
        
        # 重置状态
        processing_status = {
            'is_processing': True,
            'progress': 0,
            'message': '正在初始化...',
            'error': None
        }
        
        # 在后台线程中执行处理
        def process_in_background():
            try:
                logger.info("开始ReID处理...")
                pipeline_instance = get_pipeline()
                if pipeline_instance:
                    results = pipeline_instance.run()
                    processing_status.update({
                        'is_processing': False,
                        'progress': 100,
                        'message': '处理完成',
                        'error': None
                    })
                    logger.info("ReID处理完成")
                else:
                    raise Exception("Pipeline初始化失败")
            except Exception as e:
                logger.error(f"处理失败: {e}", exc_info=True)
                processing_status.update({
                    'is_processing': False,
                    'progress': 0,
                    'message': '',
                    'error': str(e)
                })
        
        thread = threading.Thread(target=process_in_background, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'ReID处理已启动',
            'status': processing_status
        })
        
    except Exception as e:
        logger.error(f"启动处理失败: {e}", exc_info=True)
        processing_status = {
            'is_processing': False,
            'progress': 0,
            'message': '',
            'error': str(e)
        }
        return jsonify({
            'success': False,
            'message': f'启动处理失败: {str(e)}'
        }), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """获取处理状态"""
    return jsonify(processing_status)


@app.route('/api/results', methods=['GET'])
def api_results():
    """获取处理结果"""
    try:
        results = load_results_from_database()
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        logger.error(f"获取结果失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/recluster', methods=['POST'])
def api_recluster():
    """使用新阈值重新聚类"""
    try:
        data = request.get_json()
        new_threshold = float(data.get('threshold'))
        
        if not (0.0 <= new_threshold <= 1.0):
            return jsonify({
                'success': False,
                'message': '阈值必须在0-1之间'
            }), 400
        
        logger.info(f"重新聚类，新阈值: {new_threshold}")
        
        pipeline_instance = get_pipeline()
        if not pipeline_instance:
            return jsonify({
                'success': False,
                'message': 'Pipeline未初始化'
            }), 500
        
        # 尝试加载缓存数据
        if pipeline_instance.similarity_matrix is None:
            pipeline_instance.load_cached_data(Config.RESULTS_DIR)
        
        # 重新聚类
        results = pipeline_instance.recluster(new_threshold)
        
        return jsonify({
            'success': True,
            'message': '重新聚类完成',
            'results': {
                'total_groups': results['total_groups'],
                'groups': results['groups']
            }
        })
    except Exception as e:
        logger.error(f"重新聚类失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/images/cropped/<path:filename>')
def serve_cropped_image(filename):
    """提供裁剪图片"""
    cropped_dir = Path(Config.CROPPED_DIR)
    return send_from_directory(str(cropped_dir), filename)


@app.route('/images/original/<path:filename>')
def serve_original_image(filename):
    """提供原始图片"""
    input_dir = Path(Config.INPUT_IMAGE_DIR)
    return send_from_directory(str(input_dir), filename)


@app.route('/api/merge', methods=['POST'])
def api_merge_persons():
    """人工合并人员"""
    try:
        data = request.get_json()
        person_ids = data.get('person_ids', [])
        operator = data.get('operator', 'web_user')
        note = data.get('note', '')
        
        if not person_ids or len(person_ids) < 2:
            return jsonify({
                'success': False,
                'message': '至少需要2个person_id进行合并'
            }), 400
        
        merged_id = reid_db.merge_persons(person_ids, operator)
        
        return jsonify({
            'success': True,
            'message': '人员合并成功',
            'merged_person_id': merged_id
        })
    except Exception as e:
        logger.error(f"人员合并失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/unmerge', methods=['POST'])
def api_unmerge_persons():
    """取消人员合并"""
    try:
        data = request.get_json()
        merged_person_id = data.get('merged_person_id')
        operator = data.get('operator', 'web_user')
        note = data.get('note', '')
        
        if merged_person_id is None:
            return jsonify({
                'success': False,
                'message': '缺少merged_person_id参数'
            }), 400
        
        original_ids = reid_db.unmerge_persons(merged_person_id, operator)
        
        return jsonify({
            'success': True,
            'message': '取消合并成功',
            'original_person_ids': original_ids
        })
    except Exception as e:
        logger.error(f"取消合并失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/set_name', methods=['POST'])
def api_set_name():
    """设置人员/分组名称"""
    try:
        data = request.get_json()
        person_id = data.get('person_id')
        name = data.get('name', '').strip()
        
        if person_id is None:
            return jsonify({
                'success': False,
                'message': '缺少person_id参数'
            }), 400
        
        if not name:
            return jsonify({
                'success': False,
                'message': '名称不能为空'
            }), 400
        
        success = reid_db.set_person_name(person_id, name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'已命名为: {name}'
            })
        else:
            return jsonify({
                'success': False,
                'message': '设置名称失败'
            }), 500
    except Exception as e:
        logger.error(f"设置名称失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ==================== 数据展示页面 ====================

@app.route('/records')
def records_page():
    """监测记录展示页面"""
    try:
        db = get_caseapp_db()
        # 获取所有记录（限制1000条）
        records = db.get_all_records(limit=1000)
        
        # 统计信息
        total_count = len(records)
        with_name_count = sum(1 for r in records if r.get('ryxm'))
        without_name_count = total_count - with_name_count
        
        return render_template('records.html', 
                             records=records,
                             total_count=total_count,
                             with_name_count=with_name_count,
                             without_name_count=without_name_count)
    except Exception as e:
        logger.error(f"加载记录页面失败: {e}", exc_info=True)
        return render_template('error.html', 
                             error_message=f"加载记录失败: {str(e)}"), 500


@app.route('/api/records/reidentify', methods=['POST', 'OPTIONS'])
def api_reidentify():
    """重新识别所有有图片但人员姓名为空的记录"""
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    logger.info("收到重新识别请求")
    try:
        db = get_caseapp_db()
        # 获取所有需要重新识别的记录
        records = db.get_records_without_name()
        logger.info(f"查询到 {len(records)} 条需要重新识别的记录")
        
        if not records:
            return jsonify({
                'success': True,
                'message': '没有需要重新识别的记录',
                'processed': 0,
                'identified': 0
            })
        
        logger.info(f"开始重新识别 {len(records)} 条记录")
        
        # 初始化 ReID 集成
        reid = ReIDIntegration()
        if not reid.enabled:
            # 获取详细的失败原因
            error_details = []
            # ReID系统已整合到caseapp项目中，检查本地模块
            caseapp_dir = Path(__file__).parent
            
            if not caseapp_config.REID_CONFIG.get('enabled', True):
                error_details.append('配置中ReID功能已禁用（请检查config.py中的REID_CONFIG.enabled）')
            elif not REID_INTEGRATION_AVAILABLE:
                # 检查本地核心模块是否存在
                core_dir = caseapp_dir / 'core'
                detector_file = core_dir / 'detector.py'
                extractor_file = core_dir / 'feature_extractor.py'
                
                missing_files = []
                if not detector_file.exists():
                    missing_files.append('core/detector.py')
                if not extractor_file.exists():
                    missing_files.append('core/feature_extractor.py')
                
                if missing_files:
                    error_details.append(f'ReID系统核心模块缺失：{", ".join(missing_files)}')
                    error_details.append('请确保ReID系统已正确整合到caseapp项目中')
                    error_details.append(f'检查路径：{caseapp_dir}')
                else:
                    # 获取详细的导入错误信息
                    from reid_integration import REID_IMPORT_ERROR
                    error_details.append('ReID系统模块导入失败（可能是依赖包未安装）')
                    if REID_IMPORT_ERROR:
                        error_details.append(f'具体错误：{REID_IMPORT_ERROR}')
                    error_details.append('请运行以下命令安装依赖：')
                    error_details.append('  pip install ultralytics torch torchvision torchreid')
                    error_details.append(f'检查本地模块路径：{caseapp_dir / "core"}')
            elif reid.reid_db is None:
                error_details.append('ReID数据库模块未初始化')
            elif not reid.reid_db.enabled:
                error_details.append('ReID数据库未启用或连接失败（请检查ReID数据库配置）')
            elif reid.detector is None or reid.extractor is None:
                error_details.append('ReID检测器或特征提取器初始化失败（请查看日志获取详细信息）')
                error_details.append('可能的原因：')
                error_details.append('  1. 模型文件缺失（检查caseapp/models/目录）')
                error_details.append('  2. 依赖包版本不兼容')
                error_details.append('  3. GPU/CUDA配置问题')
            else:
                error_details.append('ReID系统初始化失败（请查看日志获取详细信息）')
            
            error_message = 'ReID系统未启用，无法进行识别。\n\n原因：\n' + '\n'.join(['  • ' + detail for detail in error_details])
            logger.warning(error_message)
            return jsonify({
                'success': False,
                'message': error_message
            }), 500
        
        # 获取图片保存路径
        image_save_path = Path(caseapp_config.RTSP_MONITOR_CONFIG.get('image_save_path', 'D:/ruoyi/uploadPath/caseapp'))
        image_url_prefix = caseapp_config.RTSP_MONITOR_CONFIG.get('image_url_prefix', 'https://localhost:8090/profile/caseapp/')
        
        processed_count = 0
        identified_count = 0
        
        # 在后台线程中处理
        def process_in_background():
            nonlocal processed_count, identified_count
            # 在新线程中创建数据库连接
            db = caseapp_database.Database()
            db.connect()
            
            try:
                for record in records:
                    try:
                        record_id = record['id']
                        pstp = record.get('pstp', '')
                        
                        if not pstp:
                            continue
                        
                        # 从URL路径转换为本地文件路径
                        # 例如: https://localhost:8090/profile/caseapp/20251212/22_xxx.jpg
                        # 转换为: D:/ruoyi/uploadPath/caseapp/20251212/22_xxx.jpg
                        if pstp.startswith('http://') or pstp.startswith('https://'):
                            # 提取相对路径
                            if image_url_prefix in pstp:
                                relative_path = pstp.replace(image_url_prefix, '')
                                image_path = image_save_path / relative_path
                            else:
                                # 尝试从URL中提取文件名
                                import urllib.parse
                                parsed = urllib.parse.urlparse(pstp)
                                filename = os.path.basename(parsed.path)
                                image_path = image_save_path / filename
                        else:
                            # 已经是本地路径
                            image_path = Path(pstp)
                            if not image_path.is_absolute():
                                image_path = image_save_path / image_path
                        
                        # 检查文件是否存在
                        if not image_path.exists():
                            logger.warning(f"图片文件不存在: {image_path} (记录ID: {record_id})")
                            continue
                        
                        # 调用 ReID 识别
                        person_name = reid.identify_person_from_image_path(str(image_path))
                        
                        if person_name:
                            # 更新数据库
                            success = db.update_person_name(record_id, person_name)
                            if success:
                                identified_count += 1
                                logger.info(f"✓ 识别成功 - 记录ID: {record_id}, 姓名: {person_name}")
                            else:
                                logger.warning(f"更新数据库失败 - 记录ID: {record_id}")
                        else:
                            logger.debug(f"未识别到人员 - 记录ID: {record_id}")
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"处理记录失败 (ID: {record.get('id')}): {e}", exc_info=True)
                        processed_count += 1
            finally:
                db.close()
        
        thread = threading.Thread(target=process_in_background, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'已启动重新识别任务，共 {len(records)} 条记录',
            'total': len(records),
            'status': 'processing'
        })
        
    except Exception as e:
        logger.error(f"重新识别失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'重新识别失败: {str(e)}'
        }), 500


@app.route('/api/records', methods=['GET'])
def api_get_records():
    """获取监测记录列表（API）"""
    try:
        db = get_caseapp_db()
        limit = request.args.get('limit', type=int, default=100)
        offset = request.args.get('offset', type=int, default=0)
        
        records = db.get_all_records(limit=limit, offset=offset)
        
        return jsonify({
            'success': True,
            'records': records,
            'count': len(records)
        })
    except Exception as e:
        logger.error(f"获取记录失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


if __name__ == '__main__':
    logger.info("启动ReID Web应用...")
    logger.info(f"服务地址: http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )

