"""
许可证生成工具
用于为客户生成软件许可证
注意：此文件仅供软件提供商使用，不要发给客户！
"""
import argparse
import os
import sys
from license_manager import LicenseManager


def generate_keys():
    """生成RSA密钥对"""
    print("=" * 60)
    print("生成RSA密钥对...")
    print("=" * 60)

    manager = LicenseManager()

    # 生成密钥对
    private_key, public_key = manager.generate_rsa_keys()

    # 保存密钥
    manager.save_private_key(private_key, "private_key.pem")
    manager.save_public_key(public_key, "public_key.pem")

    print("\n密钥对生成成功！")
    print("- private_key.pem: 私钥文件（请妥善保管，用于生成许可证）")
    print("- public_key.pem: 公钥文件（需要随软件一起发布，用于验证许可证）")
    print("=" * 60)


def show_machine_code():
    """显示当前机器的机器码"""
    print("=" * 60)
    print("获取当前机器码...")
    print("=" * 60)

    manager = LicenseManager()
    print(f"\n机器码: {manager.machine_code}")
    print("\n请将此机器码提供给客户，用于生成许可证。")
    print("=" * 60)


def create_license(machine_code, expire_days=365):
    """
    生成许可证

    参数:
        machine_code: 客户机器码
        expire_days: 有效期（天数）
    """
    print("=" * 60)
    print("生成许可证...")
    print("=" * 60)

    # 检查私钥是否存在
    if not os.path.exists("private_key.pem"):
        print("\n错误: 未找到私钥文件 private_key.pem")
        print("请先运行: python generate_license.py --generate-keys")
        print("=" * 60)
        return

    manager = LicenseManager()

    # 生成许可证
    license_data = manager.generate_license(
        machine_code=machine_code,
        expire_days=expire_days,
        private_key_file="private_key.pem"
    )

    # 保存许可证
    output_file = f"license_{machine_code[:8]}.dat"
    manager.save_license(license_data, output_file)

    print(f"\n许可证生成成功！")
    print(f"- 文件名: {output_file}")
    print(f"- 机器码: {machine_code}")
    print(f"- 有效期: {expire_days} 天")
    print(f"- 签发时间: {license_data['license_info']['issue_date']}")
    print(f"- 过期时间: {license_data['license_info']['expire_date']}")
    print("\n请将以下文件一起发送给客户:")
    print(f"1. {output_file} (许可证文件，需改名为 license.dat)")
    print(f"2. public_key.pem (公钥文件)")
    print("=" * 60)


def verify_license_file(license_file):
    """验证许可证文件"""
    print("=" * 60)
    print(f"验证许可证文件: {license_file}")
    print("=" * 60)

    if not os.path.exists(license_file):
        print(f"\n错误: 许可证文件不存在: {license_file}")
        print("=" * 60)
        return

    if not os.path.exists("public_key.pem"):
        print("\n错误: 未找到公钥文件 public_key.pem")
        print("=" * 60)
        return

    manager = LicenseManager()
    is_valid, message = manager.verify_license(
        license_file=license_file,
        public_key_file="public_key.pem"
    )

    print(f"\n验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    print(f"详细信息: {message}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="软件许可证生成和管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 首次使用，生成密钥对:
   python generate_license.py --generate-keys

2. 查看当前机器的机器码:
   python generate_license.py --show-machine-code

3. 为客户生成许可证（365天有效期）:
   python generate_license.py --create --machine-code <客户的机器码> --days 365

4. 验证许可证文件:
   python generate_license.py --verify license.dat
        """
    )

    parser.add_argument(
        '--generate-keys',
        action='store_true',
        help='生成RSA密钥对（首次使用时需要）'
    )

    parser.add_argument(
        '--show-machine-code',
        action='store_true',
        help='显示当前机器的机器码'
    )

    parser.add_argument(
        '--create',
        action='store_true',
        help='生成许可证'
    )

    parser.add_argument(
        '--machine-code',
        type=str,
        help='客户的机器码'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='许可证有效期（天数），默认365天'
    )

    parser.add_argument(
        '--verify',
        type=str,
        metavar='LICENSE_FILE',
        help='验证许可证文件'
    )

    args = parser.parse_args()

    # 如果没有任何参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 生成密钥对
    if args.generate_keys:
        generate_keys()

    # 显示机器码
    if args.show_machine_code:
        show_machine_code()

    # 生成许可证
    if args.create:
        if not args.machine_code:
            print("错误: 请使用 --machine-code 参数指定客户的机器码")
            return
        create_license(args.machine_code, args.days)

    # 验证许可证
    if args.verify:
        verify_license_file(args.verify)


if __name__ == "__main__":
    main()
