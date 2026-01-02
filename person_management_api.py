"""
人员管理API接口
提供Web API用于人员标注和命名
"""
from flask import Flask, request, jsonify, render_template_string, render_template
from loguru import logger
from person_manager import PersonManager
import json
from pathlib import Path
import sys

# 添加caseapp目录到路径
caseapp_dir = Path(__file__).parent
sys.path.insert(0, str(caseapp_dir))

# 导入caseapp的数据库和ReID集成模块
import database as caseapp_database
from database import Database
from reid_integration import ReIDIntegration, REID_AVAILABLE as REID_INTEGRATION_AVAILABLE, REID_IMPORT_ERROR
import config as caseapp_config

# 创建Flask应用
app = Flask(__name__, 
            template_folder=str(caseapp_dir / 'templates'),
            static_folder=str(caseapp_dir / 'static'),
            static_url_path='/static')
app.config['JSON_AS_ASCII'] = False

# 人员管理器
person_manager = PersonManager()

# caseapp 数据库实例（延迟初始化）
caseapp_db = None

def get_caseapp_db():
    """获取 caseapp 数据库实例（单例模式）"""
    global caseapp_db
    if caseapp_db is None:
        caseapp_db = caseapp_database.Database()
        caseapp_db.connect()
    return caseapp_db


@app.route('/api/person/emp_list', methods=['GET'])
def api_get_emp_list():
    """
    获取人员列表（来自 app_person），用于ReID命名下拉框
    """
    try:
        db = get_caseapp_db()
        rows = db.get_all_person()
        emp_list = [
            {
                'id': r.get('id'),
                'empno': r.get('empno'),
                'empname': r.get('empname'),
                'empsex': r.get('empsex'),
                'emp_dept': r.get('emp_dept'),
            }
            for r in rows
        ]
        return jsonify({
            'success': True,
            'emp_list': emp_list,
            'total': len(emp_list),
        })
    except Exception as e:
        logger.error(f"获取人员列表失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': str(e),
        }), 500


@app.route('/api/person/groups', methods=['GET'])
def api_get_groups():
    """获取所有人员分组"""
    try:
        groups = person_manager.get_all_groups()
        return jsonify({
            'success': True,
            'groups': groups,
            'total': len(groups)
        })
    except Exception as e:
        logger.error(f"获取分组失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/person/set_name', methods=['POST'])
def api_set_name():
    """设置人员名称"""
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
                'message': '人员姓名不能为空'
            }), 400

        # 校验姓名必须存在于 app_person（只允许选择管理员）
        db = get_caseapp_db()
        if not db.is_valid_emp_name(name):
            return jsonify({
                'success': False,
                'message': f'姓名“{name}”不在管理员列表中，请从下拉列表选择'
            }), 400
        
        success = person_manager.set_person_name(person_id, name)
        
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
        logger.error(f"设置名称失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/person/merge', methods=['POST'])
def api_merge_persons():
    """合并多个人员"""
    try:
        data = request.get_json()
        person_ids = data.get('person_ids', [])
        operator = data.get('operator', 'user')
        
        if not person_ids or len(person_ids) < 2:
            return jsonify({
                'success': False,
                'message': '至少需要2个person_id才能合并'
            }), 400
        
        success = person_manager.merge_persons(person_ids, operator)
        
        if success:
            return jsonify({
                'success': True,
                'message': '人员合并成功'
            })
        else:
            return jsonify({
                'success': False,
                'message': '合并失败'
            }), 500
            
    except Exception as e:
        logger.error(f"合并人员失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/person/unmerge', methods=['POST'])
def api_unmerge_persons():
    """取消人员合并"""
    try:
        data = request.get_json()
        merged_person_id = data.get('merged_person_id')
        operator = data.get('operator', 'user')
        
        if merged_person_id is None:
            return jsonify({
                'success': False,
                'message': '缺少merged_person_id参数'
            }), 400
        
        success = person_manager.unmerge_persons(merged_person_id, operator)
        
        if success:
            return jsonify({
                'success': True,
                'message': '取消合并成功'
            })
        else:
            return jsonify({
                'success': False,
                'message': '取消合并失败'
            }), 500
            
    except Exception as e:
        logger.error(f"取消合并失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/person/search', methods=['GET'])
def api_search_persons():
    """搜索人员"""
    try:
        name = request.args.get('name', '').strip()
        
        if not name:
            return jsonify({
                'success': False,
                'message': '缺少name参数'
            }), 400
        
        results = person_manager.search_persons_by_name(name)
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
            
    except Exception as e:
        logger.error(f"搜索人员失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/person/info/<int:person_id>', methods=['GET'])
def api_get_person_info(person_id):
    """获取人员详细信息"""
    try:
        info = person_manager.get_person_info(person_id)
        
        if info:
            return jsonify({
                'success': True,
                'person': info
            })
        else:
            return jsonify({
                'success': False,
                'message': '人员不存在'
            }), 404
            
    except Exception as e:
        logger.error(f"获取人员信息失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/person/status', methods=['GET'])
def api_get_status():
    """获取ReID数据库状态"""
    return jsonify({
        'success': True,
        'enabled': person_manager.is_enabled()
    })


# 简单的Web界面
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>人员管理</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .group { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
        .person { display: inline-block; margin: 5px; padding: 10px; border: 1px solid #ccc; }
        .person img { max-width: 100px; max-height: 150px; }
        input[type="text"] { padding: 5px; margin: 5px; }
        button { padding: 5px 15px; margin: 5px; cursor: pointer; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>人员管理</h1>
        <div id="status"></div>
        <div id="groups"></div>
    </div>
    
    <script>
        let empList = [];

        function loadMjList() {
            return fetch('/api/person/emp_list')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        empList = data.emp_list || [];
                    } else {
                        alert('获取人员列表失败: ' + (data.message || '未知错误'));
                    }
                })
                .catch(err => {
                    console.error('获取人员列表失败', err);
                    alert('获取人员列表失败，请检查后端日志');
                });
        }

        function loadGroups() {
            fetch('/api/person/groups')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        displayGroups(data.groups);
                    }
                });
        }
        
        function displayGroups(groups) {
            const container = document.getElementById('groups');
            container.innerHTML = '';
            
            if (!empList || empList.length === 0) {
                const warn = document.createElement('div');
                warn.className = 'error';
                warn.innerText = '未获取到人员列表（app_person），无法进行命名，请先检查管理员数据。';
                container.appendChild(warn);
                return;
            }

            groups.forEach(group => {
                const div = document.createElement('div');
                div.className = 'group';

                // 生成下拉选项
                const optionsHtml = empList.map(emp => {
                    const selected = (group.group_name && group.group_name === mj.empname) ? 'selected' : '';
                    const label = mj.empno ? `${mj.empno} - ${mj.empname}` : mj.empname;
                    return `<option value="${mj.empname}" ${selected}>${label}</option>`;
                }).join('');

                div.innerHTML = `
                    <h3>分组 ${group.group_id} - ${group.group_name || '未命名'} (${group.person_count}人)</h3>
                    <div>
                        <select id="name_${group.group_id}">
                            <option value="">请选择人员</option>
                            ${optionsHtml}
                        </select>
                        <button onclick="setName(${group.persons[0]?.person_id || 0}, 'name_${group.group_id}')">设置名称</button>
                    </div>
                    <div>
                        ${group.persons.map(p => `
                            <div class="person">
                                <img src="${p.crop_path}" alt="Person ${p.person_id}">
                                <div>ID: ${p.person_id}</div>
                            </div>
                        `).join('')}
                    </div>
                `;
                container.appendChild(div);
            });
        }
        
        function setName(personId, inputId) {
            const select = document.getElementById(inputId);
            const name = select.value;
            if (!name) {
                alert('请选择人员姓名');
                return;
            }
            fetch('/api/person/set_name', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({person_id: personId, name: name})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('设置成功');
                    loadGroups();
                } else {
                    alert('设置失败: ' + data.message);
                }
            });
        }
        
        // 检查状态
        fetch('/api/person/status')
            .then(r => r.json())
            .then(data => {
                const statusDiv = document.getElementById('status');
                if (data.enabled) {
                    statusDiv.innerHTML = '<div class="success">ReID数据库已连接</div>';
                    // 先加载人员列表，再加载分组
                    loadMjList().then(() => {
                        loadGroups();
                    });
                } else {
                    statusDiv.innerHTML = '<div class="error">ReID数据库未连接</div>';
                }
            });
    </script>
</body>
</html>
"""


@app.route('/person/manage', methods=['GET'])
def person_manage_page():
    """人员管理页面"""
    return render_template_string(HTML_TEMPLATE)


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
                'message': error_message,
                'solutions': error_details
            }), 500
        
        # 获取图片保存路径
        image_save_path = Path(caseapp_config.RTSP_MONITOR_CONFIG.get('image_save_path', 'D:/ruoyi/uploadPath/caseapp'))
        image_url_prefix = caseapp_config.RTSP_MONITOR_CONFIG.get('image_url_prefix', 'https://localhost:8090/profile/caseapp/')
        
        processed_count = 0
        identified_count = 0
        
        # 在后台线程中处理
        import threading
        def process_in_background():
            nonlocal processed_count, identified_count
            # 在新线程中创建数据库连接
            db = caseapp_database.Database()
            db.connect()
            
            try:
                for record in records:
                    try:
                        processed_count += 1
                        record_id = record['id']
                        image_path = record.get('pstp', '')
                        
                        if not image_path:
                            logger.debug(f"记录 {record_id} 没有图片路径，跳过")
                            continue
                        
                        # 调试：记录原始路径
                        logger.info(f"记录 {record_id} 原始pstp: {image_path}")
                        
                        # 构建完整的图片路径
                        full_image_path = None
                        
                        if image_path.startswith('http://') or image_path.startswith('https://'):
                            # 如果是URL，从URL中提取相对路径
                            # URL格式: http://192.168.100.11:7081/profile/caseapp/20251216/28_摄像头名称_20251216125542.jpg
                            # 需要提取: 20251216/28_摄像头名称_20251216125542.jpg
                            try:
                                from urllib.parse import urlparse
                                parsed_url = urlparse(image_path)
                                # 提取路径部分，去掉前缀（如 /profile/caseapp/）
                                url_path = parsed_url.path
                                logger.info(f"记录 {record_id} 处理URL: {image_path}, 解析的路径: {url_path}")
                                
                                # 查找 caseapp/ 之后的部分
                                if '/caseapp/' in url_path:
                                    relative_path = url_path.split('/caseapp/', 1)[1]
                                    logger.info(f"记录 {record_id} 从URL提取相对路径: {relative_path}")
                                    # 相对路径应该包含日期目录，如: 20251216/28_摄像头名称_20251216125542.jpg
                                    # 确保使用正确的路径分隔符（Windows使用\，但URL使用/）
                                    # 将URL路径中的/转换为系统路径分隔符
                                    if relative_path:
                                        # 使用Path来处理路径，自动处理分隔符
                                        relative_path_parts = [p for p in relative_path.split('/') if p]  # 移除空字符串
                                        full_image_path = image_save_path
                                        for part in relative_path_parts:
                                            full_image_path = full_image_path / part
                                        logger.info(f"记录 {record_id} URL构建的完整路径: {full_image_path}")
                                    else:
                                        logger.warning(f"记录 {record_id} URL中caseapp/后没有内容: {url_path}")
                                        continue
                                else:
                                    # 如果没有找到 caseapp/，尝试直接使用路径的最后部分
                                    logger.warning(f"记录 {record_id} URL中未找到/caseapp/，尝试其他方式: {url_path}")
                                    relative_path = url_path.lstrip('/').split('/', 1)[-1] if '/' in url_path else url_path.lstrip('/')
                                    # 尝试从文件名中提取日期（格式：YYYYMMDD_...）
                                    if len(relative_path) >= 8 and relative_path[:8].isdigit():
                                        date_dir = relative_path[:8]
                                        filename = relative_path[9:] if len(relative_path) > 9 else relative_path
                                        full_image_path = image_save_path / date_dir / filename
                                        logger.info(f"记录 {record_id} 从URL路径提取日期目录: {date_dir}, 文件名: {filename}, 完整路径: {full_image_path}")
                                    else:
                                        # 如果无法提取日期，尝试在保存路径的所有日期目录中查找
                                        logger.warning(f"记录 {record_id} 无法从URL提取日期目录: {image_path}, 相对路径: {relative_path}")
                                        continue
                            except Exception as e:
                                logger.warning(f"记录 {record_id} 解析URL失败: {image_path}, 错误: {e}", exc_info=True)
                                continue
                        elif Path(image_path).is_absolute():
                            # 如果是绝对路径，检查是否包含日期目录
                            path_obj = Path(image_path)
                            # 检查路径中是否包含日期目录（8位数字的目录名）
                            parts = path_obj.parts
                            has_date_dir = False
                            for part in parts:
                                if part.isdigit() and len(part) == 8:
                                    has_date_dir = True
                                    break
                            
                            if has_date_dir:
                                # 路径已包含日期目录，直接使用
                                full_image_path = path_obj
                            else:
                                # 绝对路径但缺少日期目录，尝试从文件名中提取日期
                                filename = path_obj.name
                                # 文件名格式：28_摄像头名称_20251216125912.jpg
                                if '_' in filename:
                                    parts = filename.split('_')
                                    if len(parts) >= 3:
                                        # 最后一部分是时间戳，格式：YYYYMMDDHHMMSS.jpg
                                        timestamp_part = parts[-1].replace('.jpg', '').replace('.JPG', '')
                                        if len(timestamp_part) >= 8 and timestamp_part[:8].isdigit():
                                            date_dir = timestamp_part[:8]
                                            # 构建包含日期目录的路径
                                            # 原路径：D:\ruoyi\uploadPath\caseapp\28_摄像头名称_20251216125912.jpg
                                            # 新路径：D:\ruoyi\uploadPath\caseapp\20251216\28_摄像头名称_20251216125912.jpg
                                            full_image_path = path_obj.parent / date_dir / filename
                                            logger.info(f"记录 {record_id} 从绝对路径提取日期目录: {date_dir}, 原路径: {path_obj}, 新路径: {full_image_path}")
                                        else:
                                            # 无法提取日期，尝试在所有日期目录中搜索
                                            logger.debug(f"记录 {record_id} 无法从文件名提取日期，搜索所有日期目录")
                                            found = False
                                            if path_obj.parent.exists():
                                                for date_dir in path_obj.parent.iterdir():
                                                    if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
                                                        potential_path = date_dir / filename
                                                        if potential_path.exists():
                                                            full_image_path = potential_path
                                                            found = True
                                                            logger.debug(f"记录 {record_id} 在日期目录 {date_dir.name} 中找到文件")
                                                            break
                                            if not found:
                                                full_image_path = path_obj  # 如果找不到，使用原路径
                                    else:
                                        full_image_path = path_obj
                                else:
                                    full_image_path = path_obj
                        else:
                            # 如果是相对路径，转换为绝对路径
                            # 相对路径可能是：20251216/28_摄像头名称_20251216125542.jpg 或只有文件名
                            if '/' in image_path or '\\' in image_path:
                                # 包含目录分隔符，直接拼接
                                full_image_path = image_save_path / image_path
                            else:
                                # 只有文件名，需要从文件名中提取日期或搜索所有日期目录
                                # 文件名格式：28_摄像头名称_20251216125542.jpg
                                # 尝试提取日期：从时间戳中提取日期部分（YYYYMMDD）
                                filename = image_path
                                if '_' in filename:
                                    parts = filename.split('_')
                                    if len(parts) >= 3:
                                        # 最后一部分是时间戳，格式：YYYYMMDDHHMMSS.jpg
                                        timestamp_part = parts[-1].replace('.jpg', '')
                                        if len(timestamp_part) >= 8 and timestamp_part[:8].isdigit():
                                            date_dir = timestamp_part[:8]
                                            full_image_path = image_save_path / date_dir / filename
                                        else:
                                            # 无法提取日期，尝试在所有日期目录中搜索
                                            logger.warning(f"记录 {record_id} 无法从文件名提取日期: {filename}")
                                            # 搜索所有日期目录
                                            found = False
                                            if image_save_path.exists():
                                                for date_dir in image_save_path.iterdir():
                                                    if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
                                                        potential_path = date_dir / filename
                                                        if potential_path.exists():
                                                            full_image_path = potential_path
                                                            found = True
                                                            break
                                            if not found:
                                                continue
                                    else:
                                        # 文件名格式不符合预期，尝试在所有日期目录中搜索
                                        found = False
                                        if image_save_path.exists():
                                            for date_dir in image_save_path.iterdir():
                                                if date_dir.is_dir() and date_dir.name.isdigit() and len(date_dir.name) == 8:
                                                    potential_path = date_dir / filename
                                                    if potential_path.exists():
                                                        full_image_path = potential_path
                                                        found = True
                                                        break
                                        if not found:
                                            continue
                                else:
                                    # 文件名格式不符合预期
                                    logger.warning(f"记录 {record_id} 文件名格式不符合预期: {filename}")
                                    continue
                        
                        if full_image_path is None:
                            logger.warning(f"记录 {record_id} 路径构建失败，原始pstp: {image_path}")
                            continue
                        
                        if not full_image_path.exists():
                            logger.warning(f"记录 {record_id} 的图片文件不存在: {full_image_path}")
                            logger.info(f"记录 {record_id} 原始pstp: {image_path}, 构建的路径: {full_image_path}")
                            
                            # 尝试在所有日期目录中搜索（最后的尝试）
                            # 从路径或文件名中提取文件名
                            if Path(image_path).is_absolute():
                                filename = Path(image_path).name
                            else:
                                # 相对路径，提取文件名
                                filename = image_path.split('/')[-1].split('\\')[-1]
                            
                            logger.info(f"记录 {record_id} 尝试在所有日期目录中搜索文件: {filename}")
                            found_in_search = False
                            if image_save_path.exists():
                                for date_dir_obj in image_save_path.iterdir():
                                    if date_dir_obj.is_dir() and date_dir_obj.name.isdigit() and len(date_dir_obj.name) == 8:
                                        potential_path = date_dir_obj / filename
                                        if potential_path.exists():
                                            full_image_path = potential_path
                                            found_in_search = True
                                            logger.info(f"记录 {record_id} 在日期目录 {date_dir_obj.name} 中找到文件: {potential_path}")
                                            break
                            
                            if not found_in_search:
                                logger.warning(f"记录 {record_id} 在所有日期目录中未找到文件: {filename}")
                                continue
                        
                        # 使用ReID系统识别人员
                        person_name = reid.identify_person_from_image_path(str(full_image_path))
                        
                        if person_name:
                            # 更新数据库记录
                            db.update_person_name(record_id, person_name)
                            identified_count += 1
                            logger.info(f"记录 {record_id} 识别成功: {person_name}")
                        else:
                            logger.debug(f"记录 {record_id} 未能识别到人员")
                            
                    except Exception as e:
                        logger.error(f"处理记录 {record.get('id', 'unknown')} 失败: {e}", exc_info=True)
                        continue
                
                logger.info(f"重新识别完成: 处理 {processed_count} 条，识别 {identified_count} 条")
            finally:
                db.close()
        
        # 启动后台处理线程
        thread = threading.Thread(target=process_in_background, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'重新识别任务已启动，正在处理 {len(records)} 条记录...',
            'total': len(records),
            'processed': processed_count,
            'identified': identified_count
        })
        
    except Exception as e:
        logger.error(f"重新识别失败: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'重新识别失败: {str(e)}',
            'solutions': ["请查看后端日志 (logs/reid_process.log) 获取详细错误信息。"]
        }), 500


if __name__ == '__main__':
    logger.info("启动人员管理API服务...")
    app.run(host='0.0.0.0', port=5001, debug=True)

