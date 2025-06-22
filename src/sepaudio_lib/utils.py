import re
from pathlib import Path

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a valid filename component."""
    name = str(name) # Ensure it's a string
    name = re.sub(r'[^\w\s-]', '', name) # Remove invalid chars
    name = re.sub(r'[-\s]+', '-', name).strip('-') # Replace spaces/hyphens with single hyphen
    return name if name else "untitled"

def is_url(string_to_check: str) -> bool:
    """Basic check if a string is a URL."""
    s = str(string_to_check) # Ensure it's a string
    return s.startswith('http://') or s.startswith('https://')