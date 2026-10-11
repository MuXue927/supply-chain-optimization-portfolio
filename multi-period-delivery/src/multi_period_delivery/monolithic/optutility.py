import datetime
import time
import sys
import re
from rich.console import Console
from dataclasses import dataclass, field
from gurobipy import GRB

BigNum = 99999
MYEPS = -0.0001
POSITIVEEPS = 0.00001

# A callback is a user function that is called periodically by the Gurobi optimizer in order to allow the user
# to query or modify the state of the optimization. More precisely, if you pass a function that takes two
# arguments (model and where) as the argument to Model.optimize or Model.computeIIS, your function will be called
# during the optimization. Your callback function can then call Model.cbGet to query the optimizer for details
# on the state of the optimization.
# Model.cbGet() --> Query the optimizer from within the user callback.
# cbGet(what) --> what: Integer code that indicates what type of information is being requested by the callback.
# The set of valid codes depends on the where value that is passed into the user callback function.


def my_callback(model, where):
    # GRB.Callback.MIPSOL --> Found a new MIP incumbent.
    # GRB.Callback.MIPSOL_OBJ --> Objective value for new solution.
    if where == GRB.Callback.MIP:
        obj_bst = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        if abs(obj_bst - obj_bnd) < 0.1 * (1.0 + abs(obj_bst)):
            print("Stop early - 10% gap achieved")
            model.terminate()
    elif where == GRB.Callback.MIPSOL and model.cbGet(GRB.Callback.MIPSOL_OBJ) > MYEPS:
        # 发现新解，并且新解的目标函数值大于我规定的精度
        model.terminate()

# Model.terminate() --> Generate a request to terminate the current optimization. This method can be called at any
# time during an optimization (from a callback, from another thread, from an interrupt handler, etc.). Note that,
# in general, the request won't be acted upon immediately.
# When the optimization stops, the Status attribute will be equal to GRB_INTERRUPTED.

@dataclass
class LogPrinter:
    start_time: float
    console: Console = field(init=False)
    use_rich_formatting: bool = field(init=False)


    # The generated __init__() code will call a method named __post_init__(), if __post_init__() is defined
    # on the class. It will normally be called as self.__post_init__(). However, if any InitVar fields are defined,
    # they will also be passed to __post_init__() in the order they were defined in the class. If no __init__()
    # method is generated, then __post_init__() will not automatically be called.

    def __post_init__(self):
        self.set_output_mode()

    def set_output_mode(self, force_plain=False):
        is_tty = sys.stdout.isatty() and not force_plain
        self.use_rich_formatting = is_tty
        if is_tty:
            # 如果是teletype，启用全部功能
            self.console = Console()
        else:
            # 如果不是，则禁用终端特定的格式化
            # force_terminal=False确保rich不会使用终端特定的格式化
            # color_system=None表示完全禁用字体颜色
            self.console = Console(force_terminal=False, color_system=None)

    def strip_ansi(self, text):
        # ======================================================================================
        # 下面这个正则表达式非常全面，几乎可以匹配所有常见的ANSI转义序列。使用这个模式
        # 可以有效地识别和删除文本中的ANSI转义序列,从而得到纯文本内容
        # re.compile(pattern, flags=0)
        # 将正则表达式的样式编译为一个正则对象，可用于匹配，表达式的行为可以通过指定flags值来修改
        # \x1B -- 这匹配ASCII转义字符,其十六进制值是1B。在ANSI转义序列中,这个字符总是出现在序列的开始。
        # (?:...) -- 这是一个非捕获组。它用于分组,但不会创建一个单独的捕获组。
        # [@-Z\\-_] -- 这表示一个字符集，可以匹配从@到Z的任何大写字母、反斜杠\、以及下划线_
        # | 表示“或”操作符，分隔两个可能的模式
        # \[ 表示匹配一个左方括号，方括号在正则表达式中有特殊含义，所以需要转义
        # [0-?]* 表示匹配0次或者1次或者多次字符，范围从0到?
        # [ -/]* 表示匹配0次或者1次或者多次字符，范围从空格到斜杠/
        # [@-~]* 表示匹配0次或者1次或者多次字符，范围从@到~
        # ======================================================================================
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    # # print optimization log with info of consuming time and exact time.
    def print(self, msg: str, color: str = 'bold blue'):
        # global START_TIME  # indicates START_TIME is a global variable and can be used out of the scope of print() func.
        end_time = time.time()
        # # datetime.now() --> 返回表示当前地方时的date和time对象，该方法会在可能的情况下提供比通过 time.time() 时间戳所获时间值更高的精度
        # now = datetime.datetime.now()
        formatted_msg = f'{end_time - self.start_time :.1f}s ' + msg
        if self.use_rich_formatting:
            self.console.print(formatted_msg, style=color)
        else:
            cleaned_msg = self.strip_ansi(formatted_msg)
            print(cleaned_msg)

        # self.console.print(now.strftime(
        #     f"%Y-%m-%d %H:%M:%S consumed time={end_time - self.start_time :.2f}s ")
        #                    + msg, style=color)

    def print_title(self, msg: str, color: str = 'bold blue', stars_len=75):
        line = "*" * stars_len
        centered_msg = msg.center(stars_len)
        centered_line = line.center(stars_len)
        if self.use_rich_formatting:
            self.console.print(centered_line, style=color)
            self.console.print(centered_msg, style=color)
            self.console.print(centered_line, style=color)
        else:
            print(self.strip_ansi(centered_line))
            print(self.strip_ansi(centered_msg))
            print(self.strip_ansi(centered_line))

# 不要在模块级别创建实例
# log_printer = LogPrinter(time.time())
