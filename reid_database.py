"""
ReID数据库操作模块（多数据源支持）
用于连接ReID系统的数据库
"""
import sys
from pathlib import Path
from loguru import logger
import config_loader as config

# ReID系统已整合到caseapp项目中
# 优先使用本地模块（reid_database_modu），如果不存在则尝试外部路径（向后兼容）
caseapp_dir = Path(__file__).parent
reid_database_module_dir = caseapp_dir / 'reid_database_modu'

# 如果本地模块存在，不需要添加外部路径
if not reid_database_module_dir.exists():
    # 向后兼容：如果本地模块不存在，尝试外部路径
    reid_system_path = caseapp_dir / config.REID_CONFIG.get('reid_system_path', '../person-re-identification-system-main')
    reid_system_path = reid_system_path.resolve()
    if reid_system_path.exists():
        sys.path.insert(0, str(reid_system_path))
        logger.debug(f"已添加ReID系统路径到sys.path: {reid_system_path}")
    else:
        logger.warning(f"ReID系统路径不存在: {reid_system_path}")
        logger.info(f"请确保ReID系统已整合到caseapp项目中（检查 {reid_database_module_dir}）")
else:
    logger.debug(f"使用本地ReID数据库模块: {reid_database_module_dir}")

# 延迟导入，在修改配置后再导入
REID_DB_AVAILABLE = None
DatabaseManager = None
DatabaseConfirmationManager = None

def _check_reid_modules_exist():
    """检查ReID模块文件是否存在（不导入）"""
    try:
        caseapp_dir = Path(__file__).parent
        reid_database_module_dir = caseapp_dir / 'reid_database_modu'
        
        # 优先检查本地模块
        if reid_database_module_dir.exists():
            database_init = reid_database_module_dir / '__init__.py'
            db_config_file = reid_database_module_dir / 'db_config.py'
            
            logger.info(f"检查本地ReID数据库模块: {reid_database_module_dir}")
            logger.info(f"  - reid_database_module目录存在: {reid_database_module_dir.exists()}")
            logger.info(f"  - __init__.py存在: {database_init.exists()}")
            logger.info(f"  - db_config.py存在: {db_config_file.exists()}")
            
            if not database_init.exists():
                logger.error(f"本地ReID数据库模块不完整: {database_init} 不存在")
                return False
            return True
        
        # 向后兼容：检查外部路径
        reid_system_path = caseapp_dir / config.REID_CONFIG.get('reid_system_path', '../person-re-identification-system-main')
        reid_system_path = reid_system_path.resolve()
        database_dir = reid_system_path / 'database'
        database_init = database_dir / '__init__.py'
        db_config_file = database_dir / 'db_config.py'
        
        logger.info(f"检查外部ReID系统路径: {reid_system_path}")
        logger.info(f"  - ReID系统目录存在: {reid_system_path.exists()}")
        logger.info(f"  - database目录存在: {database_dir.exists()}")
        logger.info(f"  - __init__.py存在: {database_init.exists()}")
        logger.info(f"  - db_config.py存在: {db_config_file.exists()}")
        
        if not reid_system_path.exists():
            logger.error(f"ReID系统路径不存在: {reid_system_path}")
            logger.error(f"请确保ReID系统已整合到caseapp项目中（检查 {reid_database_module_dir}）")
            return False
        if not database_dir.exists():
            logger.error(f"ReID database目录不存在: {database_dir}")
            return False
        if not database_init.exists():
            logger.error(f"ReID database/__init__.py不存在: {database_init}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"检查ReID模块时出错: {e}", exc_info=True)
        return False

def _import_reid_modules():
    """导入ReID模块（在配置修改后调用）"""
    global REID_DB_AVAILABLE, DatabaseManager, DatabaseConfirmationManager
    if REID_DB_AVAILABLE is not None:
        return REID_DB_AVAILABLE
    
    try:
        # 在导入前，创建一个简单的 Config mock 来满足 DatabaseConfirmationManager 的导入需求
        import config as caseapp_config
        caseapp_dir = Path(__file__).parent
        class MockConfig:
            """简单的 Config mock，只提供必要的属性"""
            RESULTS_DIR = caseapp_dir / 'data' / 'results'
        
        # 临时保存 caseapp 的 database 模块（如果存在）
        _caseapp_database_backup = sys.modules.get('database')
        # 临时移除 caseapp 的 database 模块，让 Python 导入 ReID 系统的
        if 'database' in sys.modules:
            del sys.modules['database']
        
        # 临时替换 sys.modules 中的 config，使用我们的 mock
        original_config = sys.modules.get('config')
        sys.modules['config'] = type(sys)('config')  # 创建一个新模块
        sys.modules['config'].Config = MockConfig
        
        try:
            # 优先尝试从本地模块导入
            caseapp_dir = Path(__file__).parent
            reid_database_module_dir = caseapp_dir / 'reid_database_modu'
            
            if reid_database_module_dir.exists():
                # 从本地模块导入（使用importlib避免模块名冲突）
                logger.debug(f"从本地模块导入: {reid_database_module_dir}")
                import importlib.util
                import types
                
                # 在加载 database_manager.py 之前，需要先设置好 database 模块
                # 因为 database_manager.py 中有 from database.db_config import ...
                _temp_database_backup = sys.modules.get('database')
                
                # 创建临时的 database 包结构
                temp_database_package = types.ModuleType('database')
                sys.modules['database'] = temp_database_package
                
                # 先加载 db_config 模块
                db_config_path = reid_database_module_dir / 'db_config.py'
                if db_config_path.exists():
                    db_config_spec = importlib.util.spec_from_file_location("database.db_config", db_config_path)
                    db_config_module = importlib.util.module_from_spec(db_config_spec)
                    # 将 db_config 模块添加到 database 包中
                    temp_database_package.db_config = db_config_module
                    sys.modules['database.db_config'] = db_config_module
                    db_config_spec.loader.exec_module(db_config_module)
                
                # 加载 models 模块
                models_path = reid_database_module_dir / 'models.py'
                if models_path.exists():
                    models_spec = importlib.util.spec_from_file_location("database.models", models_path)
                    models_module = importlib.util.module_from_spec(models_spec)
                    temp_database_package.models = models_module
                    sys.modules['database.models'] = models_module
                    models_spec.loader.exec_module(models_module)
                
                # 现在加载 database_manager
                db_manager_path = reid_database_module_dir / 'database_manager.py'
                db_manager_spec = importlib.util.spec_from_file_location("database.database_manager", db_manager_path)
                db_manager_module = importlib.util.module_from_spec(db_manager_spec)
                temp_database_package.database_manager = db_manager_module
                sys.modules['database.database_manager'] = db_manager_module
                db_manager_spec.loader.exec_module(db_manager_module)
                DM = db_manager_module.DatabaseManager
                
                # 将 DatabaseManager 也添加到 database 包的顶层，供 db_confirmation_manager 使用
                temp_database_package.DatabaseManager = DM
                
                try:
                    # 导入 db_confirmation_manager（从core模块）
                    from core.db_confirmation_manager import DatabaseConfirmationManager as DCM
                finally:
                    # 恢复原来的 database 模块（但不要覆盖 _caseapp_database_backup，它会在外层 finally 中恢复）
                    if _temp_database_backup:
                        sys.modules['database'] = _temp_database_backup
                    elif 'database' in sys.modules and sys.modules['database'] is temp_database_package:
                        # 如果当前是临时包，删除它（让外层 finally 恢复）
                        del sys.modules['database']
                        # 同时删除子模块
                        for key in list(sys.modules.keys()):
                            if key.startswith('database.'):
                                del sys.modules[key]
            else:
                # 向后兼容：从外部路径导入
                logger.debug("从外部路径导入")
                from database.database_manager import DatabaseManager as DM
                from core.db_confirmation_manager import DatabaseConfirmationManager as DCM
            
            DatabaseManager = DM
            DatabaseConfirmationManager = DCM
            REID_DB_AVAILABLE = True
            logger.debug("ReID数据库模块导入成功")
        finally:
            # 恢复原来的 config 模块（如果存在）
            if original_config is not None:
                sys.modules['config'] = original_config
            else:
                # 如果原来没有 config 模块，移除我们创建的
                if 'config' in sys.modules and sys.modules['config'].__name__ == 'config':
                    del sys.modules['config']
            # 恢复 caseapp 的 database 模块
            if _caseapp_database_backup:
                sys.modules['database'] = _caseapp_database_backup
    except ImportError as e:
        logger.warning(f"ReID数据库模块导入失败: {e}", exc_info=True)
        REID_DB_AVAILABLE = False
    except Exception as e:
        logger.error(f"导入ReID模块时发生错误: {e}", exc_info=True)
        REID_DB_AVAILABLE = False
    return REID_DB_AVAILABLE


class ReIDDatabase:
    """ReID数据库操作类（支持多数据源）"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(ReIDDatabase, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化ReID数据库连接"""
        if self._initialized:
            return
        
        self._initialized = True
        self.enabled = False
        self.db_manager = None
        self.confirmation_manager = None
        
        # 检查ReID模块文件是否存在
        if not _check_reid_modules_exist():
            logger.warning("ReID数据库模块文件不存在")
            return
        
        try:
            caseapp_dir = Path(__file__).parent
            reid_database_module_dir = caseapp_dir / 'reid_database_modu'
            
            # 先修改 database/db_config.py 中的配置（必须在导入 DatabaseManager 之前）
            from urllib.parse import quote_plus
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            # 临时保存 caseapp 的 database 模块（如果存在）
            _caseapp_database_backup = sys.modules.get('database')
            # 临时移除 caseapp 的 database 模块，让 Python 导入 ReID 系统的
            if 'database' in sys.modules:
                del sys.modules['database']
            
            try:
                # 优先尝试从本地模块导入
                if reid_database_module_dir.exists():
                    logger.info(f"使用本地ReID数据库模块: {reid_database_module_dir}")
                    # 使用 importlib 从本地模块导入，避免模块名冲突
                    import importlib.util
                    db_config_path = reid_database_module_dir / 'db_config.py'
                    if db_config_path.exists():
                        spec = importlib.util.spec_from_file_location("reid_db_config", db_config_path)
                        db_config_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(db_config_module)
                    else:
                        raise ImportError(f"db_config.py 不存在: {db_config_path}")
                else:
                    # 向后兼容：从外部路径导入
                    logger.info("本地模块不存在，尝试从外部路径导入")
                    import database.db_config as db_config_module
            except Exception as e:
                logger.error(f"导入db_config模块失败: {e}", exc_info=True)
                raise
            finally:
                # 恢复 caseapp 的 database 模块
                if _caseapp_database_backup:
                    sys.modules['database'] = _caseapp_database_backup
            
            # 更新 DB_CONFIG 字典
            db_config_module.DB_CONFIG['host'] = config.REID_DATABASE_CONFIG['host']
            db_config_module.DB_CONFIG['port'] = config.REID_DATABASE_CONFIG['port']
            db_config_module.DB_CONFIG['user'] = config.REID_DATABASE_CONFIG['user']
            db_config_module.DB_CONFIG['password'] = config.REID_DATABASE_CONFIG['password']
            db_config_module.DB_CONFIG['database'] = config.REID_DATABASE_CONFIG['database']
            db_config_module.DB_CONFIG['charset'] = config.REID_DATABASE_CONFIG['charset']
            
            # 重新创建 DATABASE_URL（对密码进行URL编码）
            encoded_password = quote_plus(db_config_module.DB_CONFIG['password'])
            db_config_module.DATABASE_URL = (
                f"mysql+pymysql://{db_config_module.DB_CONFIG['user']}:{encoded_password}"
                f"@{db_config_module.DB_CONFIG['host']}:{db_config_module.DB_CONFIG['port']}"
                f"/{db_config_module.DB_CONFIG['database']}"
                f"?charset={db_config_module.DB_CONFIG['charset']}"
            )
            
            # 重新创建 engine 和 SessionLocal
            db_config_module.engine = create_engine(
                db_config_module.DATABASE_URL,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            db_config_module.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=db_config_module.engine
            )
            
            logger.info(f"已更新ReID数据库配置: {db_config_module.DB_CONFIG['host']}:{db_config_module.DB_CONFIG['port']}/{db_config_module.DB_CONFIG['database']}")
            
            # 注意：我们只使用 database/db_config.py 中的配置，不依赖 ReID 系统的 config.py
            # DatabaseManager 和 DatabaseConfirmationManager 都使用 database/db_config.py 中的配置
            
            # 现在导入并使用 DatabaseManager（此时配置已更新）
            if not _import_reid_modules():
                error_msg = "无法导入ReID数据库模块"
                logger.error(error_msg)
                raise ImportError(error_msg)
            
            # 初始化数据库管理器
            logger.info("正在初始化DatabaseManager...")
            self.db_manager = DatabaseManager()
            logger.info("✓ DatabaseManager初始化成功")
            
            logger.info("正在初始化DatabaseConfirmationManager...")
            self.confirmation_manager = DatabaseConfirmationManager()
            logger.info("✓ DatabaseConfirmationManager初始化成功")
            
            # 注意：不初始化 ReIDPipeline，因为它需要 ReID 系统的 Config
            # 我们只使用数据库操作功能，不需要完整的 Pipeline
            
            # 测试连接
            logger.info("正在测试数据库连接...")
            test_session = self.db_manager.get_session()
            test_session.close()
            logger.info("✓ 数据库连接测试成功")
            
            self.enabled = True
            logger.info("✓ ReID数据库连接成功")
        except ImportError as e:
            logger.error(f"ReID数据库模块导入失败: {e}", exc_info=True)
            self.enabled = False
        except Exception as e:
            logger.error(f"ReID数据库初始化失败: {e}", exc_info=True)
            self.enabled = False
    
    def get_all_groups(self):
        """获取所有分组"""
        if not self.enabled:
            return []
        try:
            return self.db_manager.get_all_groups()
        except Exception as e:
            logger.error(f"获取分组失败: {e}")
            return []
    
    def set_person_name(self, person_id: int, name: str) -> bool:
        """
        设置人员名称
        
        Args:
            person_id: 人员ID
            name: 人员姓名
            
        Returns:
            成功返回True，失败返回False
        """
        if not self.enabled:
            logger.warning("ReID数据库未启用")
            return False
        
        try:
            merged_id = self.confirmation_manager.set_name(person_id, name)
            logger.info(f"设置人员名称成功: person_id={person_id}, name={name}, merged_id={merged_id}")
            return True
        except Exception as e:
            logger.error(f"设置人员名称失败: {e}")
            return False
    
    def merge_persons(self, person_ids: list, operator: str = "system") -> bool:
        """
        合并多个人员
        
        Args:
            person_ids: 要合并的人员ID列表
            operator: 操作者
            
        Returns:
            成功返回True，失败返回False
        """
        if not self.enabled:
            logger.warning("ReID数据库未启用")
            return False
        
        try:
            merged_id = self.confirmation_manager.merge_groups(
                person_ids=person_ids,
                operator=operator,
                note="从caseapp系统合并"
            )
            logger.info(f"合并人员成功: person_ids={person_ids} -> merged_id={merged_id}")
            return True
        except Exception as e:
            logger.error(f"合并人员失败: {e}")
            return False
    
    def unmerge_persons(self, merged_person_id: int, operator: str = "system") -> bool:
        """
        取消合并
        
        Args:
            merged_person_id: 合并后的人员ID
            operator: 操作者
            
        Returns:
            成功返回True，失败返回False
        """
        if not self.enabled:
            logger.warning("ReID数据库未启用")
            return False
        
        try:
            original_ids = self.confirmation_manager.unmerge_group(
                merged_person_id=merged_person_id,
                operator=operator,
                note="从caseapp系统取消合并"
            )
            logger.info(f"取消合并成功: merged_person_id={merged_person_id} -> {original_ids}")
            return True
        except Exception as e:
            logger.error(f"取消合并失败: {e}")
            return False
    
    def get_person_name(self, person_id: int) -> str:
        """
        获取人员名称
        
        Args:
            person_id: 人员ID
            
        Returns:
            人员姓名，如果未设置返回空字符串
        """
        if not self.enabled:
            return ""
        
        try:
            return self.confirmation_manager.get_name(person_id)
        except Exception as e:
            logger.error(f"获取人员名称失败: {e}")
            return ""
    
    def get_all_persons(self):
        """获取所有人员"""
        if not self.enabled:
            return []
        try:
            return self.db_manager.get_all_persons()
        except Exception as e:
            logger.error(f"获取人员列表失败: {e}")
            return []

