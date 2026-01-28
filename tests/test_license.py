"""
许可证验证系统单元测试
覆盖 specs/授权许可系统.md 的核心需求
"""
import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from license_manager import LicenseManager


class TestMachineCodeGeneration:
    """机器码生成测试"""

    def test_machine_code_is_sha256_format(self):
        """机器码应为 SHA256 格式（64 位十六进制）"""
        manager = LicenseManager()
        machine_code = manager.machine_code

        # SHA256 产生 64 位十六进制字符串
        assert len(machine_code) == 64
        assert all(c in '0123456789abcdef' for c in machine_code)

    def test_machine_code_is_deterministic(self):
        """同一机器的机器码应一致"""
        manager1 = LicenseManager()
        manager2 = LicenseManager()

        assert manager1.machine_code == manager2.machine_code

    def test_get_mac_address_format(self):
        """MAC 地址应为标准格式"""
        mac = LicenseManager.get_mac_address()

        # MAC 地址格式：XX:XX:XX:XX:XX:XX
        if mac:  # 可能在某些环境下获取失败
            assert len(mac) == 17
            parts = mac.split(':')
            assert len(parts) == 6
            for part in parts:
                assert len(part) == 2

    def test_machine_code_uses_multiple_hardware_ids(self):
        """机器码应综合多个硬件标识"""
        with patch.object(LicenseManager, 'get_cpu_id', return_value='CPU123'):
            with patch.object(LicenseManager, 'get_mac_address', return_value='AA:BB:CC:DD:EE:FF'):
                with patch.object(LicenseManager, 'get_disk_serial', return_value='DISK456'):
                    with patch.object(LicenseManager, 'get_motherboard_serial', return_value='BOARD789'):
                        manager = LicenseManager()

                        # 验证机器码是这些值的组合哈希
                        import hashlib
                        expected_input = "CPU123|AA:BB:CC:DD:EE:FF|DISK456|BOARD789"
                        expected_hash = hashlib.sha256(expected_input.encode()).hexdigest()

                        assert manager.machine_code == expected_hash


class TestRSAKeyGeneration:
    """RSA 密钥生成测试"""

    def test_generate_key_pair(self):
        """应能生成有效的 RSA 密钥对"""
        private_key, public_key = LicenseManager.generate_rsa_keys()

        assert private_key is not None
        assert public_key is not None

    def test_key_size_is_2048(self):
        """密钥大小应为 2048 位"""
        private_key, public_key = LicenseManager.generate_rsa_keys()

        assert private_key.key_size == 2048

    def test_save_and_load_private_key(self):
        """私钥应能正确保存和加载"""
        private_key, _ = LicenseManager.generate_rsa_keys()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
            temp_path = f.name

        try:
            LicenseManager.save_private_key(private_key, temp_path)
            loaded_key = LicenseManager.load_private_key(temp_path)

            assert loaded_key.key_size == private_key.key_size
        finally:
            os.unlink(temp_path)

    def test_save_and_load_public_key(self):
        """公钥应能正确保存和加载"""
        _, public_key = LicenseManager.generate_rsa_keys()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
            temp_path = f.name

        try:
            LicenseManager.save_public_key(public_key, temp_path)
            loaded_key = LicenseManager.load_public_key(temp_path)

            assert loaded_key.key_size == public_key.key_size
        finally:
            os.unlink(temp_path)


class TestLicenseGeneration:
    """许可证生成测试"""

    @pytest.fixture
    def temp_keys(self):
        """创建临时密钥对"""
        private_key, public_key = LicenseManager.generate_rsa_keys()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='_private.pem', delete=False) as f:
            private_path = f.name
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_public.pem', delete=False) as f:
            public_path = f.name

        LicenseManager.save_private_key(private_key, private_path)
        LicenseManager.save_public_key(public_key, public_path)

        yield private_path, public_path

        os.unlink(private_path)
        os.unlink(public_path)

    def test_generate_license_contains_required_fields(self, temp_keys):
        """生成的许可证应包含必要字段"""
        private_path, _ = temp_keys
        manager = LicenseManager()

        license_data = manager.generate_license(
            machine_code=manager.machine_code,
            expire_days=365,
            private_key_file=private_path
        )

        assert 'license_info' in license_data
        assert 'signature' in license_data

        info = license_data['license_info']
        assert 'machine_code' in info
        assert 'expire_date' in info
        assert 'issue_date' in info
        assert 'version' in info

    def test_generate_license_respects_expire_days(self, temp_keys):
        """许可证有效期应符合指定天数"""
        private_path, _ = temp_keys
        manager = LicenseManager()

        license_data = manager.generate_license(
            machine_code=manager.machine_code,
            expire_days=30,
            private_key_file=private_path
        )

        expire_date_str = license_data['license_info']['expire_date']
        expire_date = datetime.strptime(expire_date_str, "%Y-%m-%d %H:%M:%S")

        # 有效期应约为 30 天
        days_diff = (expire_date - datetime.now()).days
        assert 29 <= days_diff <= 31


class TestLicenseVerification:
    """许可证验证测试"""

    @pytest.fixture
    def valid_license_setup(self):
        """创建有效许可证的完整设置"""
        private_key, public_key = LicenseManager.generate_rsa_keys()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='_private.pem', delete=False) as f:
            private_path = f.name
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_public.pem', delete=False) as f:
            public_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            license_path = f.name

        LicenseManager.save_private_key(private_key, private_path)
        LicenseManager.save_public_key(public_key, public_path)

        manager = LicenseManager()
        license_data = manager.generate_license(
            machine_code=manager.machine_code,
            expire_days=365,
            private_key_file=private_path
        )
        manager.save_license(license_data, license_path)

        yield manager, license_path, public_path

        os.unlink(private_path)
        os.unlink(public_path)
        os.unlink(license_path)

    def test_valid_license_passes_verification(self, valid_license_setup):
        """有效许可证应通过验证"""
        manager, license_path, public_path = valid_license_setup

        is_valid, message = manager.verify_license(license_path, public_path)

        assert is_valid is True
        assert "验证成功" in message

    def test_missing_license_file_fails(self):
        """缺少许可证文件应验证失败"""
        manager = LicenseManager()

        is_valid, message = manager.verify_license(
            "nonexistent_license.dat",
            "public_key.pem"
        )

        assert is_valid is False
        assert "不存在" in message

    def test_missing_public_key_fails(self, valid_license_setup):
        """缺少公钥文件应验证失败"""
        manager, license_path, _ = valid_license_setup

        is_valid, message = manager.verify_license(
            license_path,
            "nonexistent_public_key.pem"
        )

        assert is_valid is False
        assert "不存在" in message


class TestExpiredLicenseRejection:
    """过期许可证拒绝测试"""

    def test_expired_license_is_rejected(self):
        """过期许可证应被拒绝"""
        private_key, public_key = LicenseManager.generate_rsa_keys()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='_private.pem', delete=False) as f:
            private_path = f.name
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_public.pem', delete=False) as f:
            public_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            license_path = f.name

        try:
            LicenseManager.save_private_key(private_key, private_path)
            LicenseManager.save_public_key(public_key, public_path)

            manager = LicenseManager()

            # 创建已过期的许可证（直接构造）
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            expired_date = datetime.now() - timedelta(days=1)
            license_info = {
                "machine_code": manager.machine_code,
                "expire_date": expired_date.strftime("%Y-%m-%d %H:%M:%S"),
                "issue_date": (datetime.now() - timedelta(days=366)).strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0"
            }

            license_json = json.dumps(license_info, sort_keys=True)
            signature = private_key.sign(
                license_json.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            license_data = {
                "license_info": license_info,
                "signature": signature.hex()
            }

            with open(license_path, 'w', encoding='utf-8') as f:
                json.dump(license_data, f)

            # 验证
            is_valid, message = manager.verify_license(license_path, public_path)

            assert is_valid is False
            assert "过期" in message

        finally:
            os.unlink(private_path)
            os.unlink(public_path)
            os.unlink(license_path)


class TestSignatureTamperRejection:
    """签名篡改拒绝测试"""

    def test_tampered_machine_code_is_rejected(self):
        """篡改机器码应被拒绝"""
        private_key, public_key = LicenseManager.generate_rsa_keys()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='_private.pem', delete=False) as f:
            private_path = f.name
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_public.pem', delete=False) as f:
            public_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            license_path = f.name

        try:
            LicenseManager.save_private_key(private_key, private_path)
            LicenseManager.save_public_key(public_key, public_path)

            manager = LicenseManager()
            license_data = manager.generate_license(
                machine_code=manager.machine_code,
                expire_days=365,
                private_key_file=private_path
            )

            # 篡改机器码
            license_data['license_info']['machine_code'] = 'tampered_code_' + '0' * 50

            with open(license_path, 'w', encoding='utf-8') as f:
                json.dump(license_data, f)

            is_valid, message = manager.verify_license(license_path, public_path)

            assert is_valid is False
            # 可能是机器码不匹配或签名验证失败
            assert "不匹配" in message or "签名" in message or "失败" in message

        finally:
            os.unlink(private_path)
            os.unlink(public_path)
            os.unlink(license_path)

    def test_tampered_signature_is_rejected(self):
        """篡改签名应被拒绝"""
        private_key, public_key = LicenseManager.generate_rsa_keys()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='_private.pem', delete=False) as f:
            private_path = f.name
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_public.pem', delete=False) as f:
            public_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            license_path = f.name

        try:
            LicenseManager.save_private_key(private_key, private_path)
            LicenseManager.save_public_key(public_key, public_path)

            manager = LicenseManager()
            license_data = manager.generate_license(
                machine_code=manager.machine_code,
                expire_days=365,
                private_key_file=private_path
            )

            # 篡改签名
            original_sig = license_data['signature']
            tampered_sig = '00' + original_sig[2:]  # 改变签名的前两个字符
            license_data['signature'] = tampered_sig

            with open(license_path, 'w', encoding='utf-8') as f:
                json.dump(license_data, f)

            is_valid, message = manager.verify_license(license_path, public_path)

            assert is_valid is False
            assert "签名" in message or "失败" in message

        finally:
            os.unlink(private_path)
            os.unlink(public_path)
            os.unlink(license_path)


class TestCheckLicense:
    """check_license 简化接口测试"""

    def test_check_license_returns_boolean(self):
        """check_license 应返回布尔值"""
        manager = LicenseManager()

        # 没有许可证文件时应返回 False
        result = manager.check_license()

        assert isinstance(result, bool)

    def test_check_license_without_files_returns_false(self):
        """没有许可证文件时应返回 False"""
        # 确保文件不存在
        if os.path.exists("license.dat"):
            os.rename("license.dat", "license.dat.bak")
        if os.path.exists("public_key.pem"):
            os.rename("public_key.pem", "public_key.pem.bak")

        try:
            manager = LicenseManager()
            result = manager.check_license()

            assert result is False
        finally:
            # 恢复文件
            if os.path.exists("license.dat.bak"):
                os.rename("license.dat.bak", "license.dat")
            if os.path.exists("public_key.pem.bak"):
                os.rename("public_key.pem.bak", "public_key.pem")


class TestLicenseFileFormat:
    """许可证文件格式测试"""

    def test_license_file_is_json(self):
        """许可证文件应为 JSON 格式"""
        private_key, public_key = LicenseManager.generate_rsa_keys()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='_private.pem', delete=False) as f:
            private_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            license_path = f.name

        try:
            LicenseManager.save_private_key(private_key, private_path)

            manager = LicenseManager()
            license_data = manager.generate_license(
                machine_code=manager.machine_code,
                expire_days=365,
                private_key_file=private_path
            )
            manager.save_license(license_data, license_path)

            # 应能正确读取为 JSON
            with open(license_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            assert 'license_info' in loaded_data
            assert 'signature' in loaded_data

        finally:
            os.unlink(private_path)
            os.unlink(license_path)
