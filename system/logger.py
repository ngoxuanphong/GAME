import warnings
from typing import Optional, Type

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

min_level = 30

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(
    string: str, color: str, bold: bool = False, highlight: bool = False
) -> str:
    """Returns string surrounded by appropriate terminal colour codes to print colourised text.

    Args:
        string: The message to colourise
        color: Literal values are gray, red, green, yellow, blue, magenta, cyan, white, crimson
        bold: If to bold the string
        highlight: If to highlight the string

    Returns:
        Colourised string
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    attrs = ";".join(attr)
    return f"\x1b[{attrs}m{string}\x1b[0m"

def warn(
    msg: str,
    *args: object,
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 1,
):
    """Raises a warning to the user if the min_level <= WARN.

    Args:
        msg: The message to warn the user
        *args: Additional information to warn the user
        category: The category of warning
        stacklevel: The stack level to raise to
    """
    if min_level <= WARN:
        warnings.warn(
            colorize(f"WARN: {msg % args}", "yellow"),
            category=category,
            stacklevel=stacklevel + 1,
        )