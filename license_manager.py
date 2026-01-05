"""
软件许可证管理模块
提供机器码获取、证书生成和验证功能
"""
import hashlib
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from loguru import logger


class LicenseManager:
    """许可证管理器"""

    # 证书文件路径
    LICENSE_FILE = "license.dat"
    # 公钥文件（用于验证）
    PUBLIC_KEY_FILE = "public_key.pem"

    def __init__(self):
        self.machine_code = self.get_machine_code()

    @staticmethod
    def get_cpu_id():
        """获取CPU ID"""
        try:
            if platform.system() == "Windows":
                # Windows系统使用wmic获取CPU ID
                output = subprocess.check_output(
                    "wmic cpu get ProcessorId",
                    shell=True
                ).decode().strip()
                cpu_id = output.split('\n')[1].strip()
                return cpu_id
            else:
                # Linux系统读取/proc/cpuinfo
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'Serial' in line:
                            return line.split(':')[1].strip()
                # 如果没有Serial，使用其他标识
                return str(uuid.getnode())
        except Exception as e:
            logger.warning(f"获取CPU ID失败: {e}")
            return ""

    @staticmethod
    def get_mac_address():
        """获取MAC地址"""
        try:
            mac = uuid.getnode()
            mac_address = ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))
            return mac_address
        except Exception as e:
            logger.warning(f"获取MAC地址失败: {e}")
            return ""

    @staticmethod
    def get_disk_serial():
        """获取硬盘序列号"""
        try:
            if platform.system() == "Windows":
                # Windows系统获取C盘序列号
                output = subprocess.check_output(
                    "wmic diskdrive get SerialNumber",
                    shell=True
                ).decode().strip()
                lines = output.split('\n')
                if len(lines) > 1:
                    serial = lines[1].strip()
                    return serial
            else:
                # Linux系统
                output = subprocess.check_output(
                    "lsblk -d -o serial",
                    shell=True
                ).decode().strip()
                lines = output.split('\n')
                if len(lines) > 1:
                    return lines[1].strip()
        except Exception as e:
            logger.warning(f"获取硬盘序列号失败: {e}")
        return ""

    @staticmethod
    def get_motherboard_serial():
        """获取主板序列号"""
        try:
            if platform.system() == "Windows":
                output = subprocess.check_output(
                    "wmic baseboard get SerialNumber",
                    shell=True
                ).decode().strip()
                lines = output.split('\n')
                if len(lines) > 1:
                    return lines[1].strip()
            else:
                output = subprocess.check_output(
                    "dmidecode -s baseboard-serial-number",
                    shell=True
                ).decode().strip()
                return output
        except Exception as e:
            logger.warning(f"获取主板序列号失败: {e}")
        return ""

    def get_machine_code(self):
        """
        生成唯一的机器码
        综合CPU ID、MAC地址、硬盘序列号、主板序列号生成
        """
        cpu_id = self.get_cpu_id()
        mac = self.get_mac_address()
        disk_serial = self.get_disk_serial()
        board_serial = self.get_motherboard_serial()

        # 组合所有硬件信息
        hardware_info = f"{cpu_id}|{mac}|{disk_serial}|{board_serial}"

        # 使用SHA256生成机器码
        machine_code = hashlib.sha256(hardware_info.encode()).hexdigest()

        logger.info(f"机器码: {machine_code}")
        return machine_code

    @staticmethod
    def generate_rsa_keys():
        """
        生成RSA密钥对
        返回: (private_key, public_key)
        """
        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # 生成公钥
        public_key = private_key.public_key()

        return private_key, public_key

    @staticmethod
    def save_private_key(private_key, filename="private_key.pem"):
        """保存私钥到文件"""
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        with open(filename, 'wb') as f:
            f.write(pem)

        logger.info(f"私钥已保存到: {filename}")

    @staticmethod
    def save_public_key(public_key, filename="public_key.pem"):
        """保存公钥到文件"""
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        with open(filename, 'wb') as f:
            f.write(pem)

        logger.info(f"公钥已保存到: {filename}")

    @staticmethod
    def load_private_key(filename="private_key.pem"):
        """从文件加载私钥"""
        with open(filename, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        return private_key

    @staticmethod
    def load_public_key(filename="public_key.pem"):
        """从文件加载公钥"""
        with open(filename, 'rb') as f:
            public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )
        return public_key

    def generate_license(self, machine_code, expire_days=365, private_key_file="private_key.pem"):
        """
        生成许可证

        参数:
            machine_code: 机器码
            expire_days: 有效期（天数）
            private_key_file: 私钥文件路径

        返回:
            license_data: 包含签名的许可证数据
        """
        # 加载私钥
        private_key = self.load_private_key(private_key_file)

        # 计算过期时间
        expire_date = datetime.now() + timedelta(days=expire_days)

        # 创建许可证信息
        license_info = {
            "machine_code": machine_code,
            "expire_date": expire_date.strftime("%Y-%m-%d %H:%M:%S"),
            "issue_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        }

        # 将许可证信息转换为JSON字符串
        license_json = json.dumps(license_info, sort_keys=True)

        # 使用私钥签名
        signature = private_key.sign(
            license_json.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # 组合许可证数据和签名
        license_data = {
            "license_info": license_info,
            "signature": signature.hex()
        }

        return license_data

    def save_license(self, license_data, filename="license.dat"):
        """保存许可证到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(license_data, f, indent=2, ensure_ascii=False)

        logger.info(f"许可证已保存到: {filename}")

    def verify_license(self, license_file="license.dat", public_key_file="public_key.pem"):
        """
        验证许可证

        返回:
            (is_valid, message): 验证结果和消息
        """
        try:
            # 检查许可证文件是否存在
            if not os.path.exists(license_file):
                return False, "许可证文件不存在"

            # 检查公钥文件是否存在
            if not os.path.exists(public_key_file):
                return False, "公钥文件不存在"

            # 读取许可证
            with open(license_file, 'r', encoding='utf-8') as f:
                license_data = json.load(f)

            license_info = license_data.get("license_info")
            signature_hex = license_data.get("signature")

            if not license_info or not signature_hex:
                return False, "许可证格式错误"

            # 验证机器码
            if license_info.get("machine_code") != self.machine_code:
                return False, "许可证与当前机器不匹配"

            # 验证过期时间
            expire_date_str = license_info.get("expire_date")
            expire_date = datetime.strptime(expire_date_str, "%Y-%m-%d %H:%M:%S")

            if datetime.now() > expire_date:
                return False, f"许可证已过期（过期时间：{expire_date_str}）"

            # 验证签名
            public_key = self.load_public_key(public_key_file)
            license_json = json.dumps(license_info, sort_keys=True)
            signature = bytes.fromhex(signature_hex)

            try:
                public_key.verify(
                    signature,
                    license_json.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            except Exception as e:
                return False, f"许可证签名验证失败: {e}"

            # 所有验证通过
            days_left = (expire_date - datetime.now()).days
            return True, f"许可证验证成功（剩余有效期：{days_left}天）"

        except Exception as e:
            return False, f"许可证验证失败: {e}"

    def check_license(self):
        """
        检查许可证（简化版本，用于程序启动时调用）

        返回:
            True: 许可证有效
            False: 许可证无效
        """
        is_valid, message = self.verify_license(
            license_file=self.LICENSE_FILE,
            public_key_file=self.PUBLIC_KEY_FILE
        )

        if is_valid:
            logger.info(f"✓ {message}")
        else:
            logger.error(f"✗ {message}")

        return is_valid


if __name__ == "__main__":
    # 测试代码
    manager = LicenseManager()
    print(f"机器码: {manager.machine_code}")
