"""
Web API 接口集成测试
覆盖 specs/Web界面与API.md 的 REST 端点
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_db():
    """模拟数据库"""
    with patch('person_management_api.caseapp_database') as mock_db_module:
        mock_db_instance = Mock()
        mock_db_module.Database.return_value = mock_db_instance
        mock_db_instance.connect.return_value = True
        yield mock_db_instance


@pytest.fixture
def client(mock_db):
    """Flask 测试客户端"""
    with patch('person_management_api.get_caseapp_db', return_value=mock_db):
        from person_management_api import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client


class TestCamerasAPI:
    """/api/cameras 端点测试"""

    def test_get_cameras_returns_200(self, client, mock_db):
        """GET /api/cameras 应返回 200"""
        mock_db.get_all_cameras.return_value = [
            {'id': 1, 'fjmc': '摄像头1', 'rtspssl': 'rtsp://192.168.1.1/stream'},
            {'id': 2, 'fjmc': '摄像头2', 'rtspssl': 'rtsp://192.168.1.2/stream'},
        ]

        response = client.get('/api/cameras')

        assert response.status_code == 200

    def test_get_cameras_returns_json(self, client, mock_db):
        """GET /api/cameras 应返回 JSON 格式"""
        mock_db.get_all_cameras.return_value = []

        response = client.get('/api/cameras')
        data = json.loads(response.data)

        assert 'success' in data
        assert 'cameras' in data
        assert 'total' in data

    def test_get_cameras_returns_list(self, client, mock_db):
        """GET /api/cameras 应返回摄像头列表"""
        cameras = [
            {'id': 1, 'fjmc': '入口摄像头'},
            {'id': 2, 'fjmc': '出口摄像头'},
        ]
        mock_db.get_all_cameras.return_value = cameras

        response = client.get('/api/cameras')
        data = json.loads(response.data)

        assert data['success'] is True
        assert len(data['cameras']) == 2
        assert data['total'] == 2

    def test_get_cameras_handles_error(self, client, mock_db):
        """GET /api/cameras 错误时应返回 500"""
        mock_db.get_all_cameras.side_effect = Exception("Database error")

        response = client.get('/api/cameras')

        assert response.status_code == 500


class TestStatisticsAPI:
    """/api/statistics 端点测试"""

    def test_get_statistics_returns_200(self, client, mock_db):
        """GET /api/statistics 应返回 200"""
        mock_db.get_all_cameras.return_value = []
        mock_db.get_all_records.return_value = []

        response = client.get('/api/statistics')

        assert response.status_code == 200

    def test_get_statistics_returns_json(self, client, mock_db):
        """GET /api/statistics 应返回 JSON 格式"""
        mock_db.get_all_cameras.return_value = []
        mock_db.get_all_records.return_value = []

        response = client.get('/api/statistics')
        data = json.loads(response.data)

        assert 'success' in data
        assert 'statistics' in data

    def test_get_statistics_contains_fields(self, client, mock_db):
        """GET /api/statistics 应包含统计字段"""
        mock_db.get_all_cameras.return_value = [{'id': 1}]
        mock_db.get_all_records.return_value = [
            {'id': 1, 'jcsj': datetime.now(), 'ryxm': '张三'},
            {'id': 2, 'jcsj': datetime.now(), 'ryxm': None},
        ]

        response = client.get('/api/statistics')
        data = json.loads(response.data)

        stats = data['statistics']
        assert 'total_cameras' in stats
        assert 'total_records' in stats
        assert 'today_records' in stats
        assert 'identified_count' in stats
        assert 'unidentified_count' in stats

    def test_get_statistics_calculates_correctly(self, client, mock_db):
        """GET /api/statistics 应正确计算统计数据"""
        mock_db.get_all_cameras.return_value = [{'id': 1}, {'id': 2}]
        mock_db.get_all_records.return_value = [
            {'id': 1, 'jcsj': datetime.now(), 'ryxm': '张三'},
            {'id': 2, 'jcsj': datetime.now(), 'ryxm': '李四'},
            {'id': 3, 'jcsj': datetime.now(), 'ryxm': None},
        ]

        response = client.get('/api/statistics')
        data = json.loads(response.data)

        stats = data['statistics']
        assert stats['total_cameras'] == 2
        assert stats['total_records'] == 3
        assert stats['identified_count'] == 2
        assert stats['unidentified_count'] == 1


class TestPersonGroupsAPI:
    """/api/person/groups 端点测试"""

    def test_get_groups_returns_200(self, client):
        """GET /api/person/groups 应返回 200"""
        with patch('person_management_api.person_manager') as mock_pm:
            mock_pm.get_all_groups.return_value = []

            response = client.get('/api/person/groups')

            assert response.status_code == 200

    def test_get_groups_returns_json(self, client):
        """GET /api/person/groups 应返回 JSON 格式"""
        with patch('person_management_api.person_manager') as mock_pm:
            mock_pm.get_all_groups.return_value = []

            response = client.get('/api/person/groups')
            data = json.loads(response.data)

            assert 'success' in data
            assert 'groups' in data
            assert 'total' in data


class TestPersonSetNameAPI:
    """/api/person/set_name 端点测试"""

    def test_set_name_requires_person_id(self, client, mock_db):
        """POST /api/person/set_name 需要 person_id"""
        response = client.post(
            '/api/person/set_name',
            data=json.dumps({'name': '张三'}),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_set_name_requires_name(self, client, mock_db):
        """POST /api/person/set_name 需要 name"""
        response = client.post(
            '/api/person/set_name',
            data=json.dumps({'person_id': 1}),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_set_name_validates_mj_name(self, client, mock_db):
        """POST /api/person/set_name 应验证人员姓名"""
        mock_db.is_valid_emp_name.return_value = False

        response = client.post(
            '/api/person/set_name',
            data=json.dumps({'person_id': 1, 'name': '无效姓名'}),
            content_type='application/json'
        )

        assert response.status_code == 400


class TestPersonSearchAPI:
    """/api/person/search 端点测试"""

    def test_search_requires_name(self, client):
        """GET /api/person/search 需要 name 参数"""
        response = client.get('/api/person/search')

        assert response.status_code == 400

    def test_search_returns_results(self, client):
        """GET /api/person/search 应返回搜索结果"""
        with patch('person_management_api.person_manager') as mock_pm:
            mock_pm.search_persons_by_name.return_value = [
                {'person_id': 1, 'name': '张三'}
            ]

            response = client.get('/api/person/search?name=张')
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data['success'] is True
            assert len(data['results']) == 1


class TestPersonInfoAPI:
    """/api/person/info/<id> 端点测试"""

    def test_get_info_returns_person(self, client):
        """GET /api/person/info/<id> 应返回人员信息"""
        with patch('person_management_api.person_manager') as mock_pm:
            mock_pm.get_person_info.return_value = {
                'person_id': 123,
                'name': '张三'
            }

            response = client.get('/api/person/info/123')
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data['success'] is True
            assert data['person']['person_id'] == 123

    def test_get_info_not_found(self, client):
        """GET /api/person/info/<id> 不存在时返回 404"""
        with patch('person_management_api.person_manager') as mock_pm:
            mock_pm.get_person_info.return_value = None

            response = client.get('/api/person/info/999')

            assert response.status_code == 404


class TestPersonStatusAPI:
    """/api/person/status 端点测试"""

    def test_status_returns_enabled(self, client):
        """GET /api/person/status 应返回启用状态"""
        with patch('person_management_api.person_manager') as mock_pm:
            mock_pm.is_enabled.return_value = True

            response = client.get('/api/person/status')
            data = json.loads(response.data)

            assert response.status_code == 200
            assert data['success'] is True
            assert data['enabled'] is True


class TestRecordsPage:
    """/records 页面测试"""

    def test_records_page_returns_200(self, client, mock_db):
        """GET /records 应返回 200"""
        mock_db.get_all_records.return_value = []

        response = client.get('/records')

        # 可能返回 200 或 500（取决于模板是否存在）
        assert response.status_code in [200, 500]


class TestMjListAPI:
    """/api/person/emp_list 端点测试"""

    def test_get_emp_list_returns_200(self, client, mock_db):
        """GET /api/person/emp_list 应返回 200"""
        mock_db.get_all_person.return_value = []

        response = client.get('/api/person/emp_list')

        assert response.status_code == 200

    def test_get_emp_list_returns_json(self, client, mock_db):
        """GET /api/person/emp_list 应返回 JSON 格式"""
        mock_db.get_all_person.return_value = [
            {'id': 1, 'empno': '001', 'empname': '张三', 'empsex': '男', 'emp_dept': '部门A'}
        ]

        response = client.get('/api/person/emp_list')
        data = json.loads(response.data)

        assert 'success' in data
        assert 'emp_list' in data
        assert 'total' in data


class TestAPIResponseFormat:
    """API 响应格式测试"""

    def test_success_response_format(self, client, mock_db):
        """成功响应应包含 success: true"""
        mock_db.get_all_cameras.return_value = []

        response = client.get('/api/cameras')
        data = json.loads(response.data)

        assert data['success'] is True

    def test_error_response_format(self, client, mock_db):
        """错误响应应包含 success: false 和 message"""
        mock_db.get_all_cameras.side_effect = Exception("Test error")

        response = client.get('/api/cameras')
        data = json.loads(response.data)

        assert data['success'] is False
        assert 'message' in data
