"""
optutility
==========

模块定位
    提供与优化日志输出相关的轻量级工具类 `LogPrinter`，用于在算法运行过程中
    统一输出带耗时前缀的彩色 / 纯文本日志（自动检测终端能力）。
    设计目标：最小依赖、可在任意模块中快速实例化并使用、避免全局副作用。

核心组件
    - 常量:
        BigNum        一般用于设置较大的默认数（占位）
        MYEPS         负向微小量 (判定浮点“略小于0”场景)
        POSITIVEEPS   正向微小量
    - LogPrinter:
        * 自动记录起始时间
        * 根据 stdout 是否是 TTY 判断是否启用 rich 彩色输出
        * 提供普通消息 + 标题块两种输出形式
        * 非彩色模式下移除 ANSI 转义序列，保证日志可重定向落盘

逻辑评审
    1. set_output_mode:
         - 正确判断 sys.stdout.isatty() 决定 rich 行为
         - 非 TTY 模式关闭颜色与终端控制符，便于持久化
    2. strip_ansi:
         - 原实现的正则遗漏 ESC 前缀 (ESC = \\x1B)，会误删除普通大写字母
           (原模式 '(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])' 会匹配大量普通字符)
         - 现修复为标准 ANSI 转义匹配：'\\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])'
    3. print:
         - 记录相对起始时间 (s) 前缀 + 富文本或纯文本输出
    4. print_title:
         - 输出三行（上分隔线 / 居中标题 / 下分隔线），增强可视化分段
    5. 模块未在加载时创建全局实例，避免副作用；调用侧需自行实例化:
            printer = LogPrinter(time.time())
            printer.print("Start")

改进内容
    - 移除未使用的 datetime 导入
    - 增加完整模块文档与类/方法中文注释
    - 修复 ANSI 转义正则 (防止误删普通字符)
    - 将 ANSI 正则编译提升为模块级常量，避免重复编译
    - 增加类型注解，提高可读性

保持不变
    - 对外 API: LogPrinter(start_time).print / print_title
    - 时间计算与颜色参数风格
"""

# =========================
# 标准库
# =========================
import time
import sys
import re
from dataclasses import dataclass, field
from typing import Optional

# =========================
# 第三方库
# =========================
from rich.console import Console

# =========================
# 常量 (与其他模块可能共享的判定阈值)
# =========================
BigNum = 99999
MYEPS = -0.0001
POSITIVEEPS = 0.00001

# ANSI 转义序列匹配（标准形式：ESC + 控制指令）
# 说明:
#   \x1B           ESC 字符
#   (?: ... | ... ) 非捕获组
#   @-Z\\-_        单字符控制序列
#   \[ ... @-~     CSI 序列：ESC [ 参数 中间字符 结束符
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


@dataclass
class LogPrinter:
    """
    统一日志输出工具:
        - 根据终端环境自动决定是否使用 rich 彩色输出
        - 输出统一带“相对起始耗时”前缀
        - 在非 TTY 环境下去除 ANSI 转义，适用于日志文件 / CI
    """
    start_time: float
    console: Console = field(init=False)
    use_rich_formatting: bool = field(init=False)

    def __post_init__(self):
        self.set_output_mode()

    # ------------------------------------------------------------------
    # 环境检测与 Console 初始化
    # ------------------------------------------------------------------
    def set_output_mode(self, force_plain: bool = False) -> None:
        """
        设置输出模式:
            force_plain=True 时强制关闭彩色输出 (调试或文件写入场景)
        """
        is_tty = sys.stdout.isatty() and not force_plain
        self.use_rich_formatting = is_tty
        if is_tty:
            # 交互式终端：启用颜色与富文本
            self.console = Console()
        else:
            # 非交互环境：禁用终端特定格式，确保输出纯文本
            self.console = Console(force_terminal=False, color_system=None)

    # ------------------------------------------------------------------
    # ANSI 清理
    # ------------------------------------------------------------------
    def strip_ansi(self, text: str) -> str:
        """
        去除 ANSI 转义序列:
            - 保证在非彩色输出模式下日志不携带控制字符
        """
        return ANSI_ESCAPE_PATTERN.sub('', text)

    # ------------------------------------------------------------------
    # 普通日志输出
    # ------------------------------------------------------------------
    def print(self, msg: str, color: str = 'bold blue') -> None:
        """
        输出单行日志:
            格式: "<耗时秒>s <消息>"
            彩色模式: 使用 rich 样式
            纯文本模式: 清理 ANSI 后输出
        """
        elapsed = time.time() - self.start_time
        formatted_msg = f'{elapsed:.1f}s {msg}'
        if self.use_rich_formatting:
            self.console.print(formatted_msg, style=color)
        else:
            self.console.print(self.strip_ansi(formatted_msg))

    # ------------------------------------------------------------------
    # 标题块输出
    # ------------------------------------------------------------------
    def print_title(self, msg: str, color: str = 'bold blue', stars_len: int = 75) -> None:
        """
        输出块状标题:
            上/中/下三行，突出分隔
        """
        line = "*" * stars_len
        centered_msg = msg.center(stars_len)
        if self.use_rich_formatting:
            self.console.print(line, style=color)
            self.console.print(centered_msg, style=color)
            self.console.print(line, style=color)
        else:
            self.console.print(self.strip_ansi(line))
            self.console.print(self.strip_ansi(centered_msg))
            self.console.print(self.strip_ansi(line))

# 不在模块级创建默认实例，避免意外共享状态
# 使用方式:
#   from .optutility import LogPrinter
#   printer = LogPrinter(time.time())
#   printer.print_title("ALNS START")
#   printer.print("Iteration 1")
