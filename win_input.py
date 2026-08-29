"""
win_input.py

Key injection for Windows, via ctypes SendInput using hardware scan codes.

Many games (anything using DirectInput or raw input, which is common for
engine-based games) ignore the "keybd_event"-style synthetic key presses
that libraries like pynput send by default on Windows - the game's input
layer just never sees them. Sending scan codes through SendInput instead
is the standard workaround: it looks like a real hardware key press to
anything reading below the Win32 message layer, so it reaches games that
higher-level key-simulation methods miss.

Same key_down(name) / key_up(name) interface as linux_input.py, so either
module can be swapped in without changing calling code - see
input_backend.py, which picks between them automatically.
"""

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# Hardware scan codes (Set 1) for each key - NOT virtual key codes. Scan
# codes are what SendInput needs for KEYEVENTF_SCANCODE to work; see the
# module docstring for why that matters over the VK-code approach.
KEY_MAP = {
    "w": 0x11,
    "a": 0x1E,
    "s": 0x1F,
    "d": 0x20,
    "z": 0x2C,
    "x": 0x2D,
    "c": 0x2E,
    "space": 0x39,
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _InputUnion),
    ]


def _send_scan_code(scan_code: int, is_key_up: bool):
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if is_key_up else 0)
    extra = ctypes.pointer(wintypes.ULONG(0))
    ki = KEYBDINPUT(wVk=0, wScan=scan_code, dwFlags=flags, time=0, dwExtraInfo=extra)
    inp = INPUT(type=INPUT_KEYBOARD, union=_InputUnion(ki=ki))
    n_sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if n_sent != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def key_down(key_name: str):
    scan_code = KEY_MAP.get(key_name)
    if scan_code is None:
        raise ValueError(f"Unknown key '{key_name}' - add it to KEY_MAP in win_input.py")
    _send_scan_code(scan_code, is_key_up=False)


def key_up(key_name: str):
    scan_code = KEY_MAP.get(key_name)
    if scan_code is None:
        raise ValueError(f"Unknown key '{key_name}' - add it to KEY_MAP in win_input.py")
    _send_scan_code(scan_code, is_key_up=True)
