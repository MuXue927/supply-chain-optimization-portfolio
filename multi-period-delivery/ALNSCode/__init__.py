"""
ALNSCode 包初始化

模块定位
    - 使 ALNSCode 成为可导入的 Python 包
    - 提供常用对象的便捷导出（re-export），便于调用方 from ALNSCode import X
    - 保持轻量与健壮：使用惰性/容错导入，避免导入时因可选依赖缺失导致失败

使用说明
    - 仅导出常用/稳定接口；若某些子模块依赖重或可能缺失，则采用 try/except 包裹
    - 导出失败不会影响包的基本可用性（但相应符号不可用）
"""

# 采用“按可用性”构建导出清单，避免列出却不可用的符号
__all__ = []

# 轻量、常用数据 IO 类
try:
    from .InputDataALNS import DataALNS
    __all__.append("DataALNS")
except Exception:
    # 可选依赖缺失或模块内部错误时，跳过该导出
    pass

try:
    from .OutputDataALNS import OutPutData
    __all__.append("OutPutData")
except Exception:
    pass

# 停止准则组合（含逻辑表达式 AND/OR 辅助构造器）
try:
    from .combined_stopping import CombinedStoppingCriterion, AND, OR, create_standard_combined_criterion
    __all__ += ["CombinedStoppingCriterion", "AND", "OR", "create_standard_combined_criterion"]
except Exception:
    pass

# 解决方案一致性检查入口
try:
    from .check_solution import check_solution
    __all__.append("check_solution")
except Exception:
    pass

# 配置对象
try:
    # 导出兼容旧代码的名为 ALNSConfig 的符号，但实际引用的是模块级 default_config 实例
    from .alns_config import default_config as ALNSConfig
    __all__.append("ALNSConfig")
except Exception:
    pass
