# 软件许可证系统使用指南

本系统采用基于机器码的证书认证机制，确保软件只能在授权的机器上运行。

## 系统组成

1. **license_manager.py** - 许可证管理核心模块
   - 获取机器码（基于CPU ID、MAC地址、硬盘序列号、主板序列号）
   - 生成和验证许可证
   - 使用RSA加密算法确保安全性

2. **generate_license.py** - 许可证生成工具
   - 生成RSA密钥对
   - 为客户生成许可证
   - 验证许可证有效性

3. **main.py** - 主程序（已集成许可证验证）
   - 启动时自动验证许可证
   - 验证失败则拒绝运行

## 使用流程

### 一、软件提供商操作（首次设置）

#### 1. 生成密钥对

首次使用时，需要生成RSA密钥对：

```bash
python generate_license.py --generate-keys
```

这将生成两个文件：
- **private_key.pem** - 私钥文件（请妥善保管，用于生成许可证）
- **public_key.pem** - 公钥文件（随软件一起发布，用于验证许可证）

**重要提醒：**
- `private_key.pem` 必须严格保密，不能泄露给客户
- `public_key.pem` 需要随软件一起发布给客户

---

### 二、为客户生成许可证

#### 1. 获取客户机器码

客户在其机器上运行以下命令获取机器码：

```bash
python generate_license.py --show-machine-code
```

或者直接运行主程序（会在验证失败时显示机器码）：

```bash
python main.py
```

客户将看到类似以下输出：
```
============================================================
获取当前机器码...
============================================================

机器码: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6

请将此机器码提供给客户，用于生成许可证。
============================================================
```

#### 2. 生成许可证

使用客户提供的机器码生成许可证（例如365天有效期）：

```bash
python generate_license.py --create --machine-code <客户的机器码> --days 365
```

示例：
```bash
python generate_license.py --create --machine-code a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6 --days 365
```

系统将生成许可证文件，例如 `license_a1b2c3d4.dat`

#### 3. 发送给客户

将以下文件发送给客户：
1. `license_a1b2c3d4.dat` - 许可证文件（客户需改名为 `license.dat`）
2. `public_key.pem` - 公钥文件（如果客户是首次安装）

---

### 三、客户操作

#### 1. 安装许可证

客户收到许可证文件后，需要：

1. 将许可证文件重命名为 `license.dat`
2. 将 `license.dat` 和 `public_key.pem` 放到软件根目录（与 `main.py` 同级）

目录结构：
```
软件目录/
├── main.py
├── license.dat          ← 许可证文件
├── public_key.pem       ← 公钥文件
├── license_manager.py
└── ...其他文件
```

#### 2. 运行软件

现在可以正常运行软件：

```bash
python main.py
```

如果许可证有效，将看到：
```
============================================================
正在验证软件许可证...
============================================================

============================================================
【许可证验证成功】
============================================================
```

如果许可证无效，将看到错误提示并退出。

---

## 常见问题

### 1. 许可证验证失败的原因

- **许可证文件不存在** - 确认 `license.dat` 文件在正确位置
- **公钥文件不存在** - 确认 `public_key.pem` 文件在正确位置
- **许可证与机器不匹配** - 许可证是为其他机器生成的
- **许可证已过期** - 需要重新申请许可证
- **许可证签名验证失败** - 许可证文件被篡改或损坏

### 2. 如何验证许可证文件

软件提供商或客户可以验证许可证文件：

```bash
python generate_license.py --verify license.dat
```

### 3. 如何延长许可证有效期

需要重新生成许可证：

```bash
python generate_license.py --create --machine-code <客户的机器码> --days 730
```

### 4. 更换硬件后许可证失效怎么办

更换主要硬件（CPU、主板、硬盘）后，机器码会改变，需要：
1. 获取新的机器码
2. 联系软件提供商重新生成许可证

---

## 安全说明

1. **私钥保护**
   - `private_key.pem` 必须严格保密
   - 建议使用专用的密钥管理系统
   - 定期备份私钥到安全位置

2. **许可证分发**
   - 每个客户的许可证都是唯一的
   - 许可证只能在对应的机器上使用
   - 建议通过安全渠道传输许可证文件

3. **公钥发布**
   - `public_key.pem` 随软件一起发布
   - 公钥文件可以公开，不影响安全性

---

## 技术细节

### 机器码生成算法

机器码由以下硬件信息组合生成：
- CPU ID
- MAC地址
- 硬盘序列号
- 主板序列号

使用 SHA256 算法生成64位十六进制字符串。

### 加密算法

- **非对称加密**: RSA-2048
- **签名算法**: PSS + SHA256
- **填充方式**: OAEP

### 许可证结构

```json
{
  "license_info": {
    "machine_code": "机器码",
    "expire_date": "过期时间",
    "issue_date": "签发时间",
    "version": "版本号"
  },
  "signature": "RSA签名（十六进制）"
}
```

---

## 命令速查表

```bash
# 生成密钥对（首次使用）
python generate_license.py --generate-keys

# 查看机器码
python generate_license.py --show-machine-code

# 生成许可证（365天）
python generate_license.py --create --machine-code <机器码> --days 365

# 生成许可证（永久，9999天）
python generate_license.py --create --machine-code <机器码> --days 9999

# 验证许可证
python generate_license.py --verify license.dat

# 查看帮助
python generate_license.py --help
```

---

## 依赖安装

确保已安装所需的Python包：

```bash
pip install cryptography loguru
```

或使用项目的 requirements.txt：

```bash
pip install -r requirements.txt
```

---

## 联系支持

如有问题，请联系软件提供商。
