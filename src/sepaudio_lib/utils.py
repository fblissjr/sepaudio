import re
from pathlib import Path

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename component."""
    name = str(name)
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '-', name).strip('-')
    return name if name else "untitled"

def is_url(string: str) -> bool:
    """Basic check if a string is a URL."""
    s = str(string)
    return s.startswith('http://') or s.startswith('https://')