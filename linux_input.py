"""
linux_input.py

Key injection for Linux, using pynput (which drives the X11 XTest extension
under the hood).
"""

from pynput.keyboard import Controller, Key

_keyboard = Controller()

KEY_MAP = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "z": "z",
    "x": "x",
    "c": "c",
    "space": Key.space,
}


def key_down(key_name: str):
    key = KEY_MAP.get(key_name)
    if key is None:
        raise ValueError(f"Unknown key '{key_name}' - add it to KEY_MAP in linux_input.py")
    _keyboard.press(key)


def key_up(key_name: str):
    key = KEY_MAP.get(key_name)
    if key is None:
        raise ValueError(f"Unknown key '{key_name}' - add it to KEY_MAP in linux_input.py")
    _keyboard.release(key)
