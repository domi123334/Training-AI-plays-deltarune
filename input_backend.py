"""
input_backend.py

Picks the right key-injection module for the current OS, so the rest of
the project (deltarune_env.py, heuristic_dodge.py) can just do

    from input_backend import key_down, key_up

without caring whether it's running on Linux or Windows.

  - Linux (and anything else)  -> linux_input.py  (pynput / X11 XTest)
  - Windows                    -> win_input.py    (ctypes SendInput, scan codes)

linux_input.py remains the main/default backend; win_input.py is only
picked up automatically when platform.system() reports "Windows".
"""

import platform

if platform.system() == "Windows":
    from win_input import key_down, key_up
else:
    from linux_input import key_down, key_up

__all__ = ["key_down", "key_up"]
