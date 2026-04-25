from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import certifi
from PIL import Image, ImageDraw, ImageFont


LAB_DIR = Path(__file__).resolve().parent
DEPTH_ANYTHING_DIR = LAB_DIR / "Depth-Anything"
OUTPUT_DIR = LAB_DIR / "outputs"
REPO_URL = "https://github.com/LiheYoung/Depth-Anything.git"
LOCAL_LAB_PYTHON = Path("A:/envs/tf210/python.exe")
PYTHON_EXE = str(LOCAL_LAB_PYTHON if LOCAL_LAB_PYTHON.exists() else Path(sys.executable))


COMMANDS = [
    [
        PYTHON_EXE,
        "run.py",
        "--encoder",
        "vits",
        "--img-path",
        "assets/examples/demo1.png",
        "--outdir",
        "../outputs/grayscale",
        "--pred-only",
        "--grayscale",
    ],
    [
        PYTHON_EXE,
        "run.py",
        "--encoder",
        "vits",
        "--img-path",
        "assets/examples/demo2.png",
        "--outdir",
        "../outputs/color",
        "--pred-only",
    ],
    [
        PYTHON_EXE,
        "run_video.py",
        "--encoder",
        "vits",
        "--video-path",
        "assets/examples_video/davis_dolphins.mp4",
        "--outdir",
        "../outputs/video",
    ],
]


EXPECTED_OUTPUTS = [
    OUTPUT_DIR / "grayscale" / "demo1_depth.png",
    OUTPUT_DIR / "color" / "demo2_depth.png",
    OUTPUT_DIR / "video" / "davis_dolphins_video_depth.mp4",
]


def ensure_depth_anything_repo() -> None:
    if (DEPTH_ANYTHING_DIR / "run.py").exists():
        return

    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(DEPTH_ANYTHING_DIR)],
        cwd=LAB_DIR,
        check=True,
    )


def run_command(command: list[str]) -> tuple[str, str]:
    env = os.environ.copy()
    env["SSL_CERT_FILE"] = certifi.where()
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    display_command = " ".join(command)
    print(f"\n$ {display_command}")
    result = subprocess.run(
        command,
        cwd=DEPTH_ANYTHING_DIR,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()
    return result.stdout, result.stderr


def create_video_preview() -> Path:
    video_path = OUTPUT_DIR / "video" / "davis_dolphins_video_depth.mp4"
    preview_path = OUTPUT_DIR / "video" / "davis_dolphins_video_depth_preview.png"
    script = (
        "import cv2, sys\n"
        "video_path, preview_path = sys.argv[1], sys.argv[2]\n"
        "cap = cv2.VideoCapture(video_path)\n"
        "ok, frame = cap.read()\n"
        "cap.release()\n"
        "if not ok:\n"
        "    raise RuntimeError(f'Could not read a preview frame from {video_path}')\n"
        "cv2.imwrite(preview_path, frame)\n"
    )
    subprocess.run([PYTHON_EXE, "-c", script, str(video_path), str(preview_path)], check=True)
    return preview_path


def render_terminal_screenshot(log_text: str) -> Path:
    screenshot_path = OUTPUT_DIR / "terminal_commands.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    wrapped_lines: list[str] = []
    for line in log_text.splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=118) or [""])

    line_height = 18
    padding = 24
    width = 1280
    height = max(360, padding * 2 + line_height * (len(wrapped_lines) + 1))
    image = Image.new("RGB", (width, height), color=(18, 24, 30))
    draw = ImageDraw.Draw(image)

    y = padding
    for line in wrapped_lines:
        fill = (136, 220, 255) if line.startswith("$") else (232, 238, 243)
        draw.text((padding, y), line, font=font, fill=fill)
        y += line_height

    image.save(screenshot_path)
    return screenshot_path


def validate_outputs(preview_path: Path) -> None:
    missing = [path for path in EXPECTED_OUTPUTS + [preview_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected outputs: " + ", ".join(map(str, missing)))

    for image_path in EXPECTED_OUTPUTS[:2] + [preview_path]:
        with Image.open(image_path) as image:
            print(f"{image_path.name}: {image.size[0]}x{image.size[1]} {image.mode}")

    video_path = EXPECTED_OUTPUTS[2]
    print(f"{video_path.name}: {video_path.stat().st_size} bytes")


def main() -> None:
    ensure_depth_anything_repo()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_sections = [
        "Lab 09 Depth Anything terminal commands",
        f"Python: {PYTHON_EXE}",
        f"Working directory: {DEPTH_ANYTHING_DIR}",
    ]

    for command in COMMANDS:
        stdout, stderr = run_command(command)
        log_sections.append("\n$ " + " ".join(command))
        if stdout.strip():
            log_sections.append(stdout.strip())
        if stderr.strip():
            log_sections.append(stderr.strip())

    preview_path = create_video_preview()
    log_sections.append("\nGenerated outputs:")
    for path in EXPECTED_OUTPUTS + [preview_path]:
        log_sections.append(str(path.relative_to(LAB_DIR)))

    log_text = "\n".join(log_sections) + "\n"
    (OUTPUT_DIR / "terminal_commands.txt").write_text(log_text, encoding="utf-8")
    screenshot_path = render_terminal_screenshot(log_text)
    validate_outputs(preview_path)
    print(f"Terminal command screenshot: {screenshot_path}")


if __name__ == "__main__":
    main()
