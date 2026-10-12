"""
optutility
==========

Module role
    Provides `LogPrinter`, a lightweight utility for optimization logs. It
    emits colored or plain text output with an elapsed-time prefix and detects
    terminal capabilities automatically. The design goal is minimal
    dependencies, easy instantiation from any module, and no global side
    effects.

Core components
    - Constants:
        BigNum        General-purpose large default value (placeholder)
        MYEPS         Small negative tolerance for floating-point checks
        POSITIVEEPS   Small positive tolerance
    - LogPrinter:
        * Records the starting time automatically.
        * Enables rich colored output based on whether stdout is a TTY.
        * Provides both ordinary messages and title blocks.
        * Removes ANSI escape sequences in non-color mode so logs can be
          redirected to files.

Design review
    1. set_output_mode:
         - Uses sys.stdout.isatty() to decide whether rich output is
           appropriate.
         - Disables color and terminal control codes in non-TTY mode for
           reliable persistence.
    2. strip_ansi:
         - The original regex omitted the ESC prefix (ESC = \\x1B), which could
           match ordinary uppercase letters.
         - The standard ANSI escape pattern is now used:
           '\\x1B(?:[@-Z\\\\-_]|\\[[0-?]*[ -/]*[@-~])'
    3. print:
         - Adds an elapsed-seconds prefix and emits rich or plain text.
    4. print_title:
         - Emits three lines (top separator / centered title / bottom
           separator) to improve visual separation.
    5. The module does not create a global instance on import, avoiding side
       effects. Callers instantiate it explicitly:
            printer = LogPrinter(time.time())
            printer.print("Start")

Improvements
    - Removed an unused datetime import.
    - Added complete module documentation and class/method documentation.
    - Fixed the ANSI escape regex to avoid deleting ordinary characters.
    - Moved ANSI regex compilation to a module-level constant.
    - Added type annotations for readability.

Unchanged behavior
    - Public API: LogPrinter(start_time).print / print_title.
    - Timing calculations and color parameter style.
"""

# =========================
# Standard library
# =========================
import re
import sys
import time
from dataclasses import dataclass, field

# =========================
# Third-party libraries
# =========================
from rich.console import Console

# =========================
# Constants (possible shared tolerance thresholds)
# =========================
BigNum = 99999
MYEPS = -0.0001
POSITIVEEPS = 0.00001

# ANSI escape sequence matching (standard form: ESC + control command).
# Notes:
#   \x1B           ESC character
#   (?: ... | ... ) non-capturing group
#   @-Z\\-_        single-character control sequence
#   \[ ... @-~     CSI sequence: ESC [ parameters intermediates final byte
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


@dataclass
class LogPrinter:
    """
    Unified log output helper:
        - Automatically selects rich colored output based on the terminal.
        - Prefixes every message with elapsed time since start.
        - Removes ANSI escapes in non-TTY environments for log files / CI.
    """
    start_time: float
    console: Console = field(init=False)
    use_rich_formatting: bool = field(init=False)

    def __post_init__(self):
        self.set_output_mode()

    # ------------------------------------------------------------------
    # Environment detection and Console initialization.
    # ------------------------------------------------------------------
    def set_output_mode(self, force_plain: bool = False) -> None:
        """
        Configure output mode. force_plain=True forces plain text output for
        debugging or file-writing scenarios.
        """
        is_tty = sys.stdout.isatty() and not force_plain
        self.use_rich_formatting = is_tty
        if is_tty:
            # Interactive terminal: enable color and rich text.
            self.console = Console()
        else:
            # Non-interactive environment: disable terminal-specific
            # formatting to ensure plain text output.
            self.console = Console(force_terminal=False, color_system=None)

    # ------------------------------------------------------------------
    # ANSI cleanup.
    # ------------------------------------------------------------------
    def strip_ansi(self, text: str) -> str:
        """
        Remove ANSI escape sequences so logs contain no control characters in
        plain-text mode.
        """
        return ANSI_ESCAPE_PATTERN.sub('', text)

    # ------------------------------------------------------------------
    # Ordinary log output.
    # ------------------------------------------------------------------
    def print(self, msg: str, color: str = 'bold blue') -> None:
        """
        Emit one log line:
            Format: "<elapsed seconds>s <message>"
            Color mode: use rich styling.
            Plain mode: strip ANSI escapes before output.
        """
        elapsed = time.time() - self.start_time
        formatted_msg = f'{elapsed:.1f}s {msg}'
        if self.use_rich_formatting:
            self.console.print(formatted_msg, style=color)
        else:
            self.console.print(self.strip_ansi(formatted_msg))

    # ------------------------------------------------------------------
    # Title block output.
    # ------------------------------------------------------------------
    def print_title(self, msg: str, color: str = 'bold blue', stars_len: int = 75) -> None:
        """
        Emit a block title with top, middle, and bottom lines for clear
        separation.
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

# Do not create a default instance at module level to avoid accidental shared
# state. Usage:
#   from .optutility import LogPrinter
#   printer = LogPrinter(time.time())
#   printer.print_title("ALNS START")
#   printer.print("Iteration 1")
