"""
数据库操作单元测试
覆盖 specs/数据库与存储.md 的核心需求
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatabasePoolCreation:
    """数据库连接池创建/回收测试"""

    def test_pool_initialization_parameters(self):
        """测试连接池初始化参数"""
        with patch('database.PooledDB') as mock_pooled:
            with patch('database.config') as mock_config:
                mock_config.DATABASE_CONFIG = {
                    'host': 'localhost',
                    'port': 3306,
                    'user': 'root',
                    'password': 'test',
                    'database': 'testdb',
                    'charset': 'utf8mb4'
                }

                from database import Database
                Database._pool = None  # 重置单例

                db = Database()

                # 验证 PooledDB 被调用
                mock_pooled.assert_called_once()

                # 检查关键参数
                call_kwargs = mock_pooled.call_args[1]
                assert call_kwargs['maxconnections'] == 20
                assert call_kwargs['mincached'] == 5
                assert call_kwargs['maxcached'] == 10
                assert call_kwargs['blocking'] is True

    def test_pool_singleton_pattern(self):
        """连接池应为单例模式"""
        with patch('database.PooledDB') as mock_pooled:
            mock_pooled.return_value = Mock()

            with patch('database.config') as mock_config:
                mock_config.DATABASE_CONFIG = {
                    'host': 'localhost',
                    'port': 3306,
                    'user': 'root',
                    'password': 'test',
                    'database': 'testdb',
                    'charset': 'utf8mb4'
                }

                from database import Database
                Database._pool = None  # 重置单例

                db1 = Database()
                db2 = Database()

                # PooledDB 只应被调用一次
                assert mock_pooled.call_count == 1

    def test_get_connection_from_pool(self):
        """测试从池中获取连接"""
        mock_pool = Mock()
        mock_conn = Mock()
        mock_pool.connection.return_value = mock_conn

        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}

            from database import Database
            Database._pool = mock_pool

            db = Database()
            conn = db._get_conn()

            assert conn == mock_conn
            mock_pool.connection.assert_called_once()


class TestSaveDetectionRecord:
    """保存检测记录测试"""

    @pytest.fixture
    def mock_db(self):
        """创建模拟数据库实例"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}
            mock_config.RTSP_MONITOR_CONFIG = {
                'long_track_threshold': 60,
                'camera_api_url': 'http://localhost:8090/api'
            }

            from database import Database
            mock_pool = Mock()
            mock_conn = Mock()
            mock_cursor = Mock()

            mock_pool.connection.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            Database._pool = mock_pool
            db = Database()

            return db, mock_cursor

    def test_save_record_returns_id(self, mock_db):
        """保存记录应返回记录 ID"""
        db, mock_cursor = mock_db
        mock_cursor.lastrowid = 123

        with patch.object(db, 'trigger_event_sync', return_value=True):
            record_id = db.save_detection_record(
                pssj=datetime.now(),
                pstp='/path/to/image.jpg',
                qyid=1,
                qymc='测试区域',
                sxtmx='测试摄像头',
                rysl=5
            )

        assert record_id == 123

    def test_save_record_sql_parameters(self, mock_db):
        """验证保存记录的 SQL 参数"""
        db, mock_cursor = mock_db
        mock_cursor.lastrowid = 1

        test_time = datetime(2025, 1, 15, 10, 30, 0)

        with patch.object(db, 'trigger_event_sync', return_value=True):
            db.save_detection_record(
                pssj=test_time,
                pstp='/images/test.jpg',
                qyid=5,
                qymc='A区',
                sxtmx='入口摄像头',
                rysl=3
            )

        # 验证 execute 被调用
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args

        # SQL 语句
        sql = call_args[0][0]
        assert 'INSERT INTO app_track' in sql

        # 参数
        params = call_args[0][1]
        assert params[0] == test_time  # pssj
        assert params[2] == '/images/test.jpg'  # pstp
        assert params[3] == 5  # qyid
        assert params[4] == 'A区'  # qymc
        assert params[5] == '入口摄像头'  # sxtmx
        assert params[6] == 3  # rysl

    def test_save_record_handles_exception(self, mock_db):
        """保存记录出错时应返回 None"""
        db, mock_cursor = mock_db
        mock_cursor.execute.side_effect = Exception("Database error")

        result = db.save_detection_record(
            pssj=datetime.now(),
            pstp='/path/to/image.jpg',
            qyid=1,
            qymc='测试区域',
            sxtmx='测试摄像头',
            rysl=1
        )

        assert result is None


class TestRecordDeduplication:
    """记录去重规则测试（min_record_interval）"""

    def test_min_record_interval_config(self):
        """min_record_interval 配置应存在且默认为 10 秒"""
        import config_loader as config
        interval = config.RTSP_MONITOR_CONFIG.get('min_record_interval', 10)
        assert interval == 10

    def test_get_last_record_time(self):
        """测试获取最近记录时间"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}

            from database import Database

            mock_pool = Mock()
            mock_conn = Mock()
            mock_cursor = Mock()

            mock_pool.connection.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            last_time = datetime(2025, 1, 15, 10, 30, 0)
            # 注意：方法使用 'last_time' 作为键名（SQL 中使用 AS last_time）
            mock_cursor.fetchone.return_value = {'last_time': last_time}

            Database._pool = mock_pool
            db = Database()

            result = db.get_last_record_time(qyid=1)

            assert result == last_time
            mock_cursor.execute.assert_called_once()


class TestTrajectoryMerge:
    """轨迹合并测试（trajectory_merge_interval）"""

    def test_trajectory_merge_interval_config(self):
        """trajectory_merge_interval 配置应存在且默认为 60 秒"""
        import config_loader as config
        interval = config.RTSP_MONITOR_CONFIG.get('trajectory_merge_interval', 60)
        assert interval == 60

    def test_update_record_end_time(self):
        """测试更新记录结束时间（轨迹合并）"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}
            mock_config.RTSP_MONITOR_CONFIG = {'long_track_threshold': 60}

            from database import Database

            mock_pool = Mock()
            mock_conn = Mock()
            mock_cursor = Mock()

            mock_pool.connection.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_cursor.rowcount = 1

            Database._pool = mock_pool
            db = Database()

            end_time = datetime(2025, 1, 15, 10, 35, 0)
            result = db.update_record_end_time(record_id=123, jssj=end_time)

            assert result is True
            mock_cursor.execute.assert_called_once()

            # 验证 SQL 包含更新 jssj
            sql = mock_cursor.execute.call_args[0][0]
            assert 'UPDATE' in sql
            assert 'jssj' in sql

    def test_update_record_people_count(self):
        """测试更新记录人数（合并时取最大值）"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}

            from database import Database

            mock_pool = Mock()
            mock_conn = Mock()
            mock_cursor = Mock()

            mock_pool.connection.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            # 模拟当前记录人数为 3，新人数为 5（应更新）
            mock_cursor.fetchone.return_value = {'rysl': 3}
            mock_cursor.rowcount = 1

            Database._pool = mock_pool
            db = Database()

            result = db.update_record_people_count_if_higher(record_id=123, new_count=5)

            assert result is True
            # execute 被调用两次：一次 SELECT，一次 UPDATE
            assert mock_cursor.execute.call_count == 2

            # 验证 UPDATE SQL 包含 rysl
            update_call = mock_cursor.execute.call_args_list[1]
            sql = update_call[0][0]
            assert 'UPDATE' in sql
            assert 'rysl' in sql


class TestGetAllRecords:
    """获取所有记录测试"""

    def test_get_all_records_with_limit(self):
        """测试获取记录带限制"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}

            from database import Database

            mock_pool = Mock()
            mock_conn = Mock()
            mock_cursor = Mock()

            mock_pool.connection.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

            mock_records = [
                {'id': 1, 'pssj': datetime.now(), 'qymc': 'A区'},
                {'id': 2, 'pssj': datetime.now(), 'qymc': 'B区'},
            ]
            mock_cursor.fetchall.return_value = mock_records

            Database._pool = mock_pool
            db = Database()

            result = db.get_all_records(limit=100)

            assert len(result) == 2
            mock_cursor.execute.assert_called_once()

            sql = mock_cursor.execute.call_args[0][0]
            assert 'LIMIT' in sql

    def test_get_all_records_order_by_time(self):
        """获取记录应按时间降序排列"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}

            from database import Database

            mock_pool = Mock()
            mock_conn = Mock()
            mock_cursor = Mock()

            mock_pool.connection.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_cursor.fetchall.return_value = []

            Database._pool = mock_pool
            db = Database()

            db.get_all_records()

            sql = mock_cursor.execute.call_args[0][0]
            assert 'ORDER BY' in sql
            assert 'DESC' in sql


class TestCameraOperations:
    """摄像头操作测试"""

    def test_get_all_cameras_from_api(self):
        """测试从 API 获取摄像头列表"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}
            mock_config.RTSP_MONITOR_CONFIG = {
                'camera_api_url': 'http://localhost:8090/api/cameras',
                'camera_api_verify_ssl': False,
                'camera_api_timeout': 5,
                'enable_usb_camera': False
            }

            with patch('database.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    'code': 0,
                    'msg': '操作成功',
                    'data': [
                        {'id': 1, 'fjmc': '摄像头1', 'rtspssl': 'rtsp://192.168.1.1/stream'},
                        {'id': 2, 'fjmc': '摄像头2', 'rtspssl': 'rtsp://192.168.1.2/stream'},
                    ]
                }
                mock_post.return_value = mock_response

                from database import Database
                Database._pool = Mock()
                db = Database()

                cameras = db.get_all_cameras()

                assert len(cameras) == 2
                assert cameras[0]['fjmc'] == '摄像头1'
                mock_post.assert_called_once()

    def test_get_camera_by_id(self):
        """测试根据 ID 获取摄像头"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}
            mock_config.RTSP_MONITOR_CONFIG = {
                'camera_api_url': 'http://localhost:8090/api/cameras',
                'camera_api_verify_ssl': False,
                'camera_api_timeout': 5,
                'enable_usb_camera': False
            }

            with patch('database.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    'code': 0,
                    'data': [
                        {'id': 5, 'fjmc': '目标摄像头', 'rtspssl': 'rtsp://192.168.1.5/stream'},
                    ]
                }
                mock_post.return_value = mock_response

                from database import Database
                Database._pool = Mock()
                db = Database()

                camera = db.get_camera_by_id(5)

                assert camera is not None
                assert camera['id'] == 5
                assert camera['fjmc'] == '目标摄像头'


class TestConnectionResilience:
    """连接恢复测试"""

    def test_reconnect_on_pool_none(self):
        """连接池为空时应尝试重新初始化"""
        with patch('database.PooledDB') as mock_pooled:
            mock_pool = Mock()
            mock_pooled.return_value = mock_pool

            with patch('database.config') as mock_config:
                mock_config.DATABASE_CONFIG = {
                    'host': 'localhost',
                    'port': 3306,
                    'user': 'root',
                    'password': 'test',
                    'database': 'testdb',
                    'charset': 'utf8mb4'
                }

                from database import Database
                Database._pool = None

                db = Database()
                result = db._reconnect()

                assert result is True
                assert Database._pool is not None

    def test_check_connection_returns_pool_status(self):
        """检查连接应返回池状态"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}

            from database import Database

            # 池存在时
            Database._pool = Mock()
            db = Database()
            assert db._check_connection() is True

            # 池不存在时
            Database._pool = None
            assert db._check_connection() is False


class TestLongTrackDetection:
    """长时间滞留检测测试"""

    def test_long_track_threshold_config(self):
        """long_track_threshold 配置应存在"""
        import config_loader as config
        threshold = config.RTSP_MONITOR_CONFIG.get('long_track_threshold', 60)
        assert threshold == 60

    def test_is_long_track_flag(self):
        """保存记录时应计算 is_long_track 标志"""
        with patch('database.config') as mock_config:
            mock_config.DATABASE_CONFIG = {}
            mock_config.RTSP_MONITOR_CONFIG = {
                'long_track_threshold': 60,
                'camera_api_url': ''
            }

            from database import Database

            mock_pool = Mock()
            mock_conn = Mock()
            mock_cursor = Mock()

            mock_pool.connection.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
            mock_cursor.lastrowid = 1

            Database._pool = mock_pool
            db = Database()

            with patch.object(db, 'trigger_event_sync', return_value=True):
                db.save_detection_record(
                    pssj=datetime.now(),
                    pstp='/path/image.jpg',
                    qyid=1,
                    qymc='区域',
                    sxtmx='摄像头',
                    rysl=1
                )

            # 验证 SQL 参数包含 is_long_track
            call_args = mock_cursor.execute.call_args
            params = call_args[0][1]
            # is_long_track 是最后一个参数
            is_long_track = params[-1]
            assert is_long_track in [0, 1]
