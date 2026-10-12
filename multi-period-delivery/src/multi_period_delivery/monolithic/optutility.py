import re
import sys
import time
from dataclasses import dataclass, field

from gurobipy import GRB
from rich.console import Console

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
        # Report a new incumbent only when it improves by the configured tolerance.
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
            # Enable full terminal formatting when attached to a TTY.
            self.console = Console()
        else:
            # Disable terminal-specific formatting for redirected output.
            # force_terminal=False prevents Rich from emitting terminal escapes.
            # color_system=None disables ANSI colors entirely.
            self.console = Console(force_terminal=False, color_system=None)

    def strip_ansi(self, text):
        # ======================================================================================
        # This pattern matches common ANSI escape sequences so they can be
        # removed before returning plain text.
        # re.compile(pattern, flags=0)
        # Compile the ANSI escape pattern. The pattern combines:
        # - ESC for the start of the sequence;
        # - a non-capturing group for common escape forms;
        # - character classes for parameter bytes and final bytes.
        # ======================================================================================
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    # # print optimization log with info of consuming time and exact time.
    def print(self, msg: str, color: str = 'bold blue'):
        # global START_TIME  # indicates START_TIME is a global variable and can be used out of the scope of print() func.
        end_time = time.time()
        # datetime.now() provides a higher-resolution local timestamp than a raw
        # time.time() value.
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

# Do not create a logger instance at module import time.
# log_printer = LogPrinter(time.time())
