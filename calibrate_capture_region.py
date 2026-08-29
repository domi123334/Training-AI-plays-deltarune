"""
calibrate_capture_region.py

Helper for finding and verifying CAPTURE_REGION in deltarune_env.py, so you
don't have to guess coordinates by hand.

USAGE
-----
1. List currently open window titles (so you know the exact string to
   search for - window titles are often not exactly "DELTARUNE"):

    python3 calibrate_capture_region.py --list

2. Once you see the right title in that list, find its region:

    python3 calibrate_capture_region.py --title "DELTARUNE"

   This prints a CAPTURE_REGION dict you can paste directly into
   deltarune_env.py, AND saves a screenshot of exactly that region to
   capture_region_preview.png so you can visually confirm it's actually
   framing the game window and not, say, half the taskbar.

Requires `xdotool` (sudo apt install xdotool) and `mss` (already in
requirements.txt).
"""

import argparse
import subprocess
import sys

import mss
import mss.tools


def list_window_titles():
    result = subprocess.run(
        ["xdotool", "search", "--name", ""],
        capture_output=True, text=True,
    )
    window_ids = result.stdout.strip().splitlines()
    if not window_ids:
        print("No windows found. Is xdotool installed and are you on X11/XWayland?")
        return

    print("Open window titles:")
    for wid in window_ids:
        name_result = subprocess.run(
            ["xdotool", "getwindowname", wid],
            capture_output=True, text=True,
        )
        title = name_result.stdout.strip()
        if title:
            print(f"  {title}")


def find_window_region(title_substring: str) -> dict:
    result = subprocess.run(
        ["xdotool", "search", "--name", title_substring],
        capture_output=True, text=True,
    )
    window_ids = result.stdout.strip().splitlines()
    if not window_ids:
        print(f"No window found matching '{title_substring}'.")
        print("Run with --list to see all open window titles.")
        sys.exit(1)

    window_id = window_ids[0]
    geo_result = subprocess.run(
        ["xdotool", "getwindowgeometry", "--shell", window_id],
        capture_output=True, text=True,
    )
    values = dict(line.split("=") for line in geo_result.stdout.strip().splitlines())
    return {
        "left": int(values["X"]),
        "top": int(values["Y"]),
        "width": int(values["WIDTH"]),
        "height": int(values["HEIGHT"]),
    }


def save_preview(region: dict, out_path: str = "capture_region_preview.png"):
    with mss.mss() as sct:
        shot = sct.grab(region)
        mss.tools.to_png(shot.rgb, shot.size, output=out_path)
    print(f"Saved a preview screenshot of this region to: {out_path}")
    print("Open it and confirm it's tightly framing the game window - if it's")
    print("cutting off part of the screen or including desktop/taskbar, the")
    print("window may have moved since detection, or the title match may be")
    print("catching the wrong window (rerun with --list to check).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List all open window titles")
    parser.add_argument("--title", type=str, help="Substring of the Deltarune window title to search for")
    args = parser.parse_args()

    if args.list:
        list_window_titles()
        return

    if not args.title:
        parser.error("Provide --title \"...\" or use --list to see available window titles first.")

    region = find_window_region(args.title)
    print("CAPTURE_REGION = " + repr(region))
    print()
    print("Paste this into deltarune_env.py (and heuristic_dodge.py if you're")
    print("using that script too) to replace the existing CAPTURE_REGION line.")
    print()
    save_preview(region)


if __name__ == "__main__":
    main()
