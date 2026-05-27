import os
import sys
import re
import logging
from datetime import datetime

# ANSI Codes for Terminal Colors
DIM = '\033[90m'
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

class Colors:
    DIM = DIM
    CYAN = CYAN
    GREEN = GREEN
    YELLOW = YELLOW
    RED = RED
    RESET = RESET
    BOLD = BOLD

def get_timestamp() -> str:
    """Returns a dim-colored timestamp [HH:MM:SS]."""
    return f"{DIM}[{datetime.now().strftime('%H:%M:%S')}]{RESET}"

def kai_print(*args, **kwargs):
    """
    Custom print function wrapper that formats logs beautifully with
    subtle timestamps and color-coded module tags.
    """
    kwargs.setdefault('flush', True)
    if not args:
        import builtins
        builtins.print(*args, **kwargs)
        return

    # Join args as string
    message = " ".join(str(arg) for arg in args)

    # Don't prefix timestamp to separators or empty whitespace lines
    stripped = message.strip()
    if not stripped or all(c in "=-_*~+" for c in stripped):
        import builtins
        builtins.print(message, **kwargs)
        return

    # Dynamically extract and color bracketed tags (e.g. [YOLO], [Control])
    match = re.match(r'^\[([^\]]+)\]\s*(.*)$', message)
    timestamp = get_timestamp()

    if match:
        tag, rest = match.groups()
        tag_lower = tag.lower()
        
        # Color coding map
        if "yolo" in tag_lower:
            tag_color = CYAN
        elif "image server" in tag_lower:
            tag_color = GREEN
        elif "control" in tag_lower:
            tag_color = BOLD + CYAN
        elif "mqtt" in tag_lower:
            tag_color = YELLOW
        elif "power" in tag_lower:
            tag_color = RED
        elif "monitoring" in tag_lower or "inactivity" in tag_lower:
            tag_color = BOLD + YELLOW
        else:
            tag_color = BOLD

        formatted_message = f"{timestamp} {tag_color}[{tag}]{RESET} {rest}"
    else:
        formatted_message = f"{timestamp} {message}"

    import builtins
    builtins.print(formatted_message, **kwargs)

class CustomFormatter(logging.Formatter):
    """Custom Formatter for python standard logging."""
    def format(self, record):
        timestamp = get_timestamp()
        
        # Choose colored label for standard logging levels
        if record.levelno == logging.WARNING:
            level_color = YELLOW
        elif record.levelno >= logging.ERROR:
            level_color = RED
        elif record.levelno == logging.INFO:
            level_color = CYAN
        else:
            level_color = DIM
            
        level_str = f"{level_color}[{record.levelname}]{RESET}"
        
        # Format the actual message
        message = record.getMessage()
        return f"{timestamp} {level_str} {message}"

def configure_logging(level=logging.INFO):
    """Configures the standard root logging handler with our custom format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
