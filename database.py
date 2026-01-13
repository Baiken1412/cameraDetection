"""
数据库操作模块 - 线程安全连接池版
"""
import pymysql
from dbutils.pooled_db import PooledDB
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from loguru import logger
import config_loader as config
import requests


class Database:
    """数据库操作类（集成连接池，线程安全）"""
    
    _pool = None  # 类变量存储连接池，保证全局唯一

    def __init__(self):
        self.config = config.DATABASE_CONFIG
        # 初始化连接池（如果尚未初始化）
        if Database._pool is None:
            self._init_pool()
    
    def _init_pool(self):
        """初始化数据库连接池"""
        try:
            logger.info("正在初始化数据库连接池...")
            Database._pool = PooledDB(
                creator=pymysql,          # 使用 pymysql 驱动
                maxconnections=20,        # 连接池允许的最大连接数
                mincached=5,              # 初始化时至少创建的空闲连接
                maxcached=10,             # 最大的空闲连接数
                blocking=True,            # 连接池满时是否阻塞等待
                ping=1,                   # 检查连接是否可用
                # 以下是 pymysql 的配置参数
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config['charset'],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True           # 开启自动提交
            )
            logger.info("数据库连接池初始化成功")
        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {e}")

    def _get_conn(self):
        """从池中获取一个连接"""
        if Database._pool is None:
            self._init_pool()
        try:
            return Database._pool.connection()
        except Exception as e:
            logger.error(f"获取数据库连接失败: {e}")
            return None

    def connect(self):
        """
        兼容旧接口。在连接池模式下，只要池初始化成功即视为连接成功。
        """
        if Database._pool is None:
            self._init_pool()
        return Database._pool is not None
    
    def close(self):
        """
        兼容旧接口。在连接池模式下，不需要手动关闭实例连接。
        真正的关闭连接池操作通常在程序退出时进行。
        """
        pass

    def close_pool(self):
        """关闭整个连接池（程序退出时调用）"""
        if Database._pool:
            Database._pool.close()
            logger.info("数据库连接池已关闭")
    
    def _check_connection(self) -> bool:
        """检查连接池状态"""
        return Database._pool is not None
    
    def _reconnect(self) -> bool:
        """兼容旧接口，尝试重新初始化池"""
        if Database._pool is None:
            self._init_pool()
        return Database._pool is not None

    def _fetch_cameras_from_api(self) -> List[Dict[str, Any]]:
        """
        调用配置的接口获取摄像头列表
        """
        api_url = config.RTSP_MONITOR_CONFIG.get('camera_api_url')
        verify_ssl = config.RTSP_MONITOR_CONFIG.get('camera_api_verify_ssl', False)
        timeout = config.RTSP_MONITOR_CONFIG.get('camera_api_timeout', 5)

        if not api_url:
            logger.error("摄像头配置接口地址未配置 (RTSP_MONITOR_CONFIG.camera_api_url)")
            return []

        try:
            logger.info(f"调用摄像头配置接口: {api_url}")
            resp = requests.post(api_url, json={}, timeout=timeout, verify=verify_ssl)
            resp.raise_for_status()

            data = resp.json()
            if not isinstance(data, dict):
                logger.error(f"摄像头配置接口返回格式异常（非JSON对象）: {data}")
                return []

            if data.get('code') != 0:
                logger.error(f"摄像头配置接口返回错误: code={data.get('code')}, msg={data.get('msg')}")
                return []

            cameras = data.get('data') or []
            if not isinstance(cameras, list):
                logger.error(f"摄像头配置接口 data 字段格式异常: {type(cameras)}")
                return []

            logger.info(f"接口返回 {len(cameras)} 个摄像头配置")
            return cameras
        except requests.exceptions.RequestException as e:
            logger.error(f"调用摄像头配置接口失败: {e}")
            return []
        except ValueError as e:
            logger.error(f"解析摄像头配置接口返回JSON失败: {e}")
            return []
        except Exception as e:
            logger.error(f"获取摄像头配置时发生未知错误: {e}", exc_info=True)
            return []

    def get_all_cameras(self) -> List[Dict[str, Any]]:
        """
        获取所有有效的摄像头配置（通过接口获取 + USB摄像头）
        """
        cameras = self._fetch_cameras_from_api()

        # 如果启用了USB摄像头，从数据库读取配置
        if config.RTSP_MONITOR_CONFIG.get('enable_usb_camera', False):
            conn = self._get_conn()
            if conn:
                try:
                    with conn.cursor() as cursor:
                        sql = "SELECT id, fjmc, gnslx, ip, dk, tdh, xh, zh, mm, rtspssl, yolo_pool_id FROM app_roomip WHERE rtspssl LIKE 'usb:%' LIMIT 1"
                        cursor.execute(sql)
                        usb_camera_db = cursor.fetchone()
                        
                        if usb_camera_db:
                            rtspssl_str = usb_camera_db['rtspssl']
                            if rtspssl_str.startswith('usb://'):
                                rtspssl_str = rtspssl_str.replace('usb://', 'usb:')
                                
                            usb_camera = {
                                'id': usb_camera_db['id'],
                                'fjmc': usb_camera_db['fjmc'] or 'USB摄像头',
                                'gnslx': usb_camera_db['gnslx'] or '测试区域',
                                'rtspssl': rtspssl_str,
                                'ip': usb_camera_db['ip'] or 'localhost',
                                'dk': usb_camera_db['dk'] or 0,
                                'tdh': usb_camera_db['tdh'] or 0,
                                'xh': usb_camera_db['xh'] or '',
                                'zh': usb_camera_db['zh'] or '',
                                'mm': usb_camera_db['mm'] or '',
                                'sblx': 'USB',
                                'yolo_pool_id': usb_camera_db.get('yolo_pool_id'),  # YOLO池ID
                            }
                            cameras.append(usb_camera)
                            logger.info(f"已从数据库加载USB摄像头: {usb_camera['fjmc']} (ID: {usb_camera['id']})")
                        else:
                            logger.warning("数据库中未找到USB摄像头配置（rtspssl LIKE 'usb:%'）")
                except Exception as e:
                    logger.error(f"查询USB摄像头配置失败: {e}")
                finally:
                    conn.close() # 归还连接

        return cameras
    
    def get_camera_by_id(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """
        根据ID查询摄像头配置（包括USB摄像头）
        """
        # 如果是USB摄像头ID
        if camera_id == -1 and config.RTSP_MONITOR_CONFIG.get('enable_usb_camera', False):
            return {
                'id': -1,
                'fjmc': config.RTSP_MONITOR_CONFIG.get('usb_camera_name', 'USB摄像头'),
                'gnslx': config.RTSP_MONITOR_CONFIG.get('usb_camera_area', 'USB监控区域'),
                'rtspssl': f"usb:{config.RTSP_MONITOR_CONFIG.get('usb_camera_id', 0)}",
                'ip': 'localhost',
                'dk': 0,
                'tdh': 0,
                'xh': '',
                'zh': '',
                'mm': '',
                'sblx': 'USB',
            }

        # 通过接口获取所有摄像头，再按ID过滤
        try:
            cameras = self._fetch_cameras_from_api()
            for cam in cameras:
                if cam.get('id') == camera_id:
                    return cam
            logger.warning(f"接口返回的摄像头列表中未找到ID={camera_id}的记录")
            return None
        except Exception as e:
            logger.error(f"通过接口查询摄像头配置失败 (ID: {camera_id}): {e}", exc_info=True)
            return None

    # ==================== 人员相关查询 ====================

    def get_all_person(self) -> List[Dict[str, Any]]:
        """查询所有有效的人员信息"""
        conn = self._get_conn()
        if not conn:
            return []
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT id, empno, empname, empsex, emp_dept, isdel, idno, phone
                    FROM app_person
                    WHERE isdel = '0' OR isdel IS NULL
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                logger.info(f"查询到 {len(rows)} 条人员信息 (app_person)")
                return rows
        except Exception as e:
            logger.error(f"查询人员信息失败: {e}")
            return []
        finally:
            conn.close()

    def is_valid_emp_name(self, name: str) -> bool:
        """验证人员姓名"""
        if not name:
            return False
        conn = self._get_conn()
        if not conn:
            return False
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT 1
                    FROM app_person
                    WHERE (isdel = '0' OR isdel IS NULL)
                      AND empname = %s
                    LIMIT 1
                """
                cursor.execute(sql, (name,))
                row = cursor.fetchone()
                return row is not None
        except Exception as e:
            logger.error(f"验证人员姓名失败: {e}")
            return False
        finally:
            conn.close()
    
    def save_detection_record(
        self,
        pssj: datetime,
        pstp: str,
        qyid: int,
        qymc: str,
        sxtmx: str,
        rysl: Optional[int] = None,
    ) -> Optional[int]:
        """
        保存监测记录 (线程安全)
        """
        conn = self._get_conn()
        if not conn:
            logger.error(f"保存监测记录失败 - 无法获取数据库连接")
            return None

        sql = """
            INSERT INTO app_track (pssj, jssj, pstp, qyid, qymc, sxtmx, rysl, jscs, bzzt, track_duration, is_long_track)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        try:
            with conn.cursor() as cursor:
                jssj = pssj + timedelta(seconds=5)
                jscs = 1  # 初始检测次数为1
                bzzt = '0'  # 默认为待标注状态
                track_duration = 5  # 初始轨迹时长为5秒
                long_track_threshold = config.RTSP_MONITOR_CONFIG.get('long_track_threshold', 60)
                is_long_track = 1 if track_duration >= long_track_threshold else 0
                cursor.execute(sql, (pssj, jssj, pstp, qyid, qymc, sxtmx, rysl, jscs, bzzt, track_duration, is_long_track))
                record_id = cursor.lastrowid
                
                logger.info(f"保存监测记录成功 - 摄像头: {sxtmx} (ID: {qyid}), 区域: {qymc}, 记录ID: {record_id}, 时间: {pssj}")

                # 触发Java系统生成复合事件（异步调用，不影响主流程）
                try:
                    self.trigger_event_sync(record_id)
                except Exception as e:
                    logger.warning(f"触发复合事件生成失败（不影响轨迹保存）: {e}")

                return record_id
        except Exception as e:
            logger.error(f"保存监测记录失败 - 摄像头: {sxtmx} (ID: {qyid}): {e}")
            return None
        finally:
            conn.close()
    
    def trigger_event_sync(self, record_id: int) -> bool:
        """
        触发Java系统为指定轨迹生成复合事件
        """
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            from config import RTSP_MONITOR_CONFIG

            # === 修改开始 ===
            # 原代码（已失效）：
            # base_url = RTSP_MONITOR_CONFIG['image_url_prefix'].split('/profile/')[0]
            
            # 新代码：尝试从 camera_api_url 获取服务器地址，或者使用默认值
            camera_api = RTSP_MONITOR_CONFIG.get('camera_api_url', '')
            if '://' in camera_api:
                # 例如从 'https://localhost:8090/caseapp/track/rtspStream' 提取 'https://localhost:8090'
                from urllib.parse import urlparse
                parsed = urlparse(camera_api)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
            else:
                # 如果获取不到，就使用默认的本地地址
                base_url = 'https://localhost:8090'
            # === 修改结束 ===

            api_url = f"{base_url}/caseapp/track/syncEventForTrack/{record_id}"

            response = requests.post(api_url, timeout=5, verify=False)

            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0:
                    logger.debug(f"已触发轨迹ID={record_id}的复合事件生成")
                    return True
                else:
                    logger.warning(f"触发复合事件生成失败 - 轨迹ID={record_id}: {result.get('msg')}")
                    return False
            return False
        except Exception as e:
            logger.warning(f"触发复合事件生成异常 - 轨迹ID={record_id}: {e}")
            return False

    def get_last_record_time(self, qyid: int) -> Optional[datetime]:
        """查询指定摄像头最近一次记录时间"""
        conn = self._get_conn()
        if not conn:
            return None
        try:
            with conn.cursor() as cursor:
                sql = "SELECT MAX(pssj) as last_time FROM app_track WHERE qyid = %s"
                cursor.execute(sql, (qyid,))
                result = cursor.fetchone()
                if result and result.get('last_time'):
                    return result['last_time']
                return None
        except Exception as e:
            logger.error(f"查询最近记录时间失败 (摄像头ID: {qyid}): {e}")
            return None
        finally:
            conn.close()
    
    def get_last_record(self, qyid: int) -> Optional[Dict[str, Any]]:
        """查询指定摄像头最近一次完整记录"""
        conn = self._get_conn()
        if not conn:
            return None
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT id, pssj, jssj, pstp, qyid, qymc 
                    FROM app_track 
                    WHERE qyid = %s 
                    ORDER BY pssj DESC 
                    LIMIT 1
                """
                cursor.execute(sql, (qyid,))
                result = cursor.fetchone()
                return result
        except Exception as e:
            logger.error(f"查询最近记录失败 (摄像头ID: {qyid}): {e}")
            return None
        finally:
            conn.close()
    
    def update_record_end_time(self, record_id: int, jssj: datetime) -> bool:
        """更新记录的结束时间"""
        conn = self._get_conn()
        if not conn:
            return False
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE app_track SET jssj = %s WHERE id = %s"
                cursor.execute(sql, (jssj, record_id))
                logger.info(f"更新记录结束时间成功 - 记录ID: {record_id}, 结束时间: {jssj}")
                return True
        except Exception as e:
            logger.error(f"更新记录结束时间失败 - 记录ID: {record_id}: {e}")
            return False
        finally:
            conn.close()

    def update_record_people_count_if_higher(self, record_id: int, new_count: int) -> bool:
        """如果新的人员数量大于当前记录中的 rysl，则更新 rysl"""
        if new_count is None or new_count <= 0:
            return False
        conn = self._get_conn()
        if not conn:
            return False
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT rysl FROM app_track WHERE id = %s", (record_id,))
                row = cursor.fetchone()
                current = row.get('rysl') if row else None

                if current is None or current < new_count:
                    cursor.execute(
                        "UPDATE app_track SET rysl = %s WHERE id = %s",
                        (new_count, record_id),
                    )
                    logger.info(f"更新记录人数 - 记录ID: {record_id}, 原人数: {current}, 新人数: {new_count}")
                    return True
                return False
        except Exception as e:
            logger.error(f"更新记录人数失败 - 记录ID: {record_id}: {e}")
            return False
        finally:
            conn.close()

    def update_record_image(self, record_id: int, image_url: str) -> bool:
        """更新记录的图片URL"""
        conn = self._get_conn()
        if not conn:
            return False
        try:
            with conn.cursor() as cursor:
                sql = "UPDATE app_track SET pstp = %s WHERE id = %s"
                cursor.execute(sql, (image_url, record_id))
                logger.info(f"更新记录图片成功 - 记录ID: {record_id}")
                return True
        except Exception as e:
            logger.error(f"更新记录图片失败 - 记录ID: {record_id}: {e}")
            return False
        finally:
            conn.close()

    def get_last_insert_id(self) -> Optional[int]:
        """获取最后插入的记录ID"""
        conn = self._get_conn()
        if not conn:
            return None
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT LAST_INSERT_ID() as id")
                result = cursor.fetchone()
                return result['id'] if result else None
        except Exception as e:
            logger.error(f"获取最后插入ID失败: {e}")
            return None
        finally:
            conn.close()
    
    def update_person_name(self, record_id: int, person_name: str) -> bool:
        """更新记录的人员姓名"""
        conn = self._get_conn()
        if not conn:
            return False
        try:
            label_status = '1'
            label_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with conn.cursor() as cursor:
                sql = """
                    UPDATE app_track 
                    SET ryxm = %s,
                        bzzt = %s,
                        bzsj = %s
                    WHERE id = %s
                """
                cursor.execute(sql, (person_name, label_status, label_time, record_id))
                logger.info(f"更新人员姓名成功 - 记录ID: {record_id}, 姓名: {person_name}")
                return True
        except Exception as e:
            logger.error(f"更新人员姓名失败 - 记录ID: {record_id}: {e}")
            return False
        finally:
            conn.close()
    
    def get_all_records(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """查询所有监测记录"""
        conn = self._get_conn()
        if not conn:
            return []
        try:
            with conn.cursor() as cursor:
                if limit:
                    sql = """
                        SELECT id, pssj, jssj, pstp, qyid, qymc, sxtmx, ryxm 
                        FROM app_track 
                        ORDER BY pssj DESC 
                        LIMIT %s OFFSET %s
                    """
                    cursor.execute(sql, (limit, offset))
                else:
                    sql = """
                        SELECT id, pssj, jssj, pstp, qyid, qymc, sxtmx, ryxm 
                        FROM app_track 
                        ORDER BY pssj DESC
                    """
                    cursor.execute(sql)
                records = cursor.fetchall()
                return records
        except Exception as e:
            logger.error(f"查询监测记录失败: {e}")
            return []
        finally:
            conn.close()
    
    def get_records_without_name(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """查询有图片但人员姓名为空的记录"""
        conn = self._get_conn()
        if not conn:
            return []
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT id, pssj, jssj, pstp, qyid, qymc, sxtmx, ryxm 
                    FROM app_track 
                    WHERE pstp IS NOT NULL AND pstp != '' 
                    AND (ryxm IS NULL OR ryxm = '')
                    ORDER BY pssj DESC
                """
                if limit:
                    sql += " LIMIT %s"
                    cursor.execute(sql, (limit,))
                else:
                    cursor.execute(sql)
                records = cursor.fetchall()
                return records
        except Exception as e:
            logger.error(f"查询未命名记录失败: {e}")
            return []
        finally:
            conn.close()
    
    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """根据ID查询单条记录"""
        conn = self._get_conn()
        if not conn:
            return None
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT id, pssj, jssj, pstp, qyid, qymc, sxtmx, ryxm
                    FROM app_track
                    WHERE id = %s
                """
                cursor.execute(sql, (record_id,))
                result = cursor.fetchone()
                return result
        except Exception as e:
            logger.error(f"查询记录失败 (ID: {record_id}): {e}")
            return None
        finally:
            conn.close()

    def update_trajectory_detection(self, record_id: int, jscs: int, jssj: datetime, rysl: int = None) -> bool:
        """
        更新轨迹记录的检测次数、结束时间、人员数量、轨迹时长和长轨迹标记
        """
        conn = self._get_conn()
        if not conn:
            return False
        try:
            with conn.cursor() as cursor:
                # 先获取开始时间以计算轨迹时长
                cursor.execute("SELECT pssj FROM app_track WHERE id = %s", (record_id,))
                row = cursor.fetchone()
                if not row:
                    logger.error(f"未找到轨迹记录 - 记录ID: {record_id}")
                    return False

                pssj = row['pssj']
                track_duration = int((jssj - pssj).total_seconds())
                long_track_threshold = config.RTSP_MONITOR_CONFIG.get('long_track_threshold', 60)
                is_long_track = 1 if track_duration >= long_track_threshold else 0

                if rysl is not None:
                    sql = "UPDATE app_track SET jscs = %s, jssj = %s, rysl = GREATEST(COALESCE(rysl, 0), %s), track_duration = %s, is_long_track = %s WHERE id = %s"
                    cursor.execute(sql, (jscs, jssj, rysl, track_duration, is_long_track, record_id))
                    logger.info(f"更新轨迹检测次数成功 - 记录ID: {record_id}, 检测次数: {jscs}, 结束时间: {jssj}, 人员数量: {rysl}, 轨迹时长: {track_duration}秒, 长轨迹: {is_long_track}")
                else:
                    sql = "UPDATE app_track SET jscs = %s, jssj = %s, track_duration = %s, is_long_track = %s WHERE id = %s"
                    cursor.execute(sql, (jscs, jssj, track_duration, is_long_track, record_id))
                    logger.info(f"更新轨迹检测次数成功 - 记录ID: {record_id}, 检测次数: {jscs}, 结束时间: {jssj}, 轨迹时长: {track_duration}秒, 长轨迹: {is_long_track}")
                return True
        except Exception as e:
            logger.error(f"更新轨迹检测次数失败 - 记录ID: {record_id}: {e}")
            return False
        finally:
            conn.close()

    def save_trajectory_screenshot(
        self,
        track_id: int,
        screenshot_url: str,
        screenshot_time: datetime,
        screenshot_order: int
    ) -> Optional[int]:
        """
        保存轨迹截图记录
        """
        conn = self._get_conn()
        if not conn:
            return None

        sql = """
            INSERT INTO app_track_screenshot (track_id, screenshot_url, screenshot_time, screenshot_order)
            VALUES (%s, %s, %s, %s)
        """

        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, (track_id, screenshot_url, screenshot_time, screenshot_order))
                screenshot_id = cursor.lastrowid
                logger.info(f"保存轨迹截图成功 - 轨迹ID: {track_id}, 截图ID: {screenshot_id}")
                return screenshot_id
        except Exception as e:
            logger.error(f"保存轨迹截图失败 - 轨迹ID: {track_id}: {e}")
            return None
        finally:
            conn.close()