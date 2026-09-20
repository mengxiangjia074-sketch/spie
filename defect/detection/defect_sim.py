#!/usr/bin/env python3
"""PCB defect simulation: remove a boxed component to fake a "missing part".

The GUI lets an operator load a captured local (large-zoom) PCB image, draw a
bounding box around one component, and produce a board where that component is
gone.  Four backends are supported:

  * ``codex`` (default) — real model image generation via the ChatGPT Pro
    ``image_gen`` tool, invoked through the logged-in ``codex`` CLI.  This is the
    path that actually produces a bitmap edit; it does not depend on the relay
    key, so it keeps working even when the relay has image generation disabled.
  * ``edits`` — the OpenAI **Images API** (``POST /v1/images/edits``) with a
    transparency mask (the transparent region = the box).  Needs a relay key with
    image-generation permission.
  * ``responses`` — the OpenAI **Responses API** (``POST /v1/responses``): the box
    is annotated with a red rectangle and the prompt tells the model to remove the
    component inside it.
  * ``inpaint`` — local OpenCV inpainting (no API, no network).  A deterministic
    fallback that fills the boxed region with surrounding board texture.

The API/Codex paths downscale/pad the image, call the backend, then composite only
the boxed region back onto the full-resolution original so nothing else changes.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

DEFAULT_PROMPT = (
    "Remove the electronic component inside the red rectangle (a surface-mount "
    "resistor or capacitor) so the board looks like the component is missing. "
    "Keep the two silver solder pads and any silkscreen markings intact and "
    "visible. Replace only the component body with bare printed-circuit-board "
    "surface that exactly matches the surrounding board: same green color, same "
    "fine grain texture, same lighting, sharp and crisp with no blur or smudge. "
    "The filled area must blend seamlessly into the surrounding board. Do not "
    "alter anything outside the red rectangle."
)

DEFAULT_BASE_URL = "http://107.182.173.201:8080/v1"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_ACTOR = "local-image-extension"
CODEX_AGENT_MODEL = "gpt-5.6-luna"
CODEX_IMAGE_PROBE_PROMPT = (
    "Check the tools exposed in this Codex session without calling any tool. "
    "Reply with exactly IMAGE_GEN_AVAILABLE if an image generation or image editing "
    "tool is exposed and callable; otherwise reply with exactly IMAGE_GEN_UNAVAILABLE."
)


class DefectSimError(RuntimeError):
    """Raised for any defect-simulation failure (bad input, network, API)."""


@dataclass
class DefectSimConfig:
    """Everything needed to talk to the image-editing endpoint."""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    prompt: str = DEFAULT_PROMPT
    wire_api: str = "codex"  # "codex" | "edits" | "responses" | "inpaint"
    size: str = "auto"
    quality: str = "auto"
    actor_authorization: str = DEFAULT_ACTOR  # "" disables the header
    max_side: int = 2048
    timeout: float = 300.0
    n: int = 1
    inpaint_radius: int = 3
    inpaint_feather: int = 12


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _multipart_body(
    fields: dict[str, str],
    files: dict[str, tuple[str, str, bytes]],
) -> tuple[bytes, str]:
    """Build a multipart/form-data body for the edits request."""
    boundary = "----LensDetect" + uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        parts.append(f"{value}\r\n".encode("utf-8"))
    for name, (filename, content_type, data) in files.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'.encode("utf-8")
        )
        parts.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _auth_headers(cfg: DefectSimConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    if cfg.api_key.strip():
        headers["Authorization"] = f"Bearer {cfg.api_key.strip()}"
    if cfg.actor_authorization.strip():
        headers["x-openai-actor-authorization"] = cfg.actor_authorization.strip()
    return headers


def _endpoint_url(cfg: DefectSimConfig, path: str) -> str:
    """Build a full endpoint URL, tolerating base URLs with or without ``/v1``."""
    base = cfg.base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{path}"
    return f"{base}/v1/{path}"


def _post_json(url: str, payload: dict[str, Any], cfg: DefectSimConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **_auth_headers(cfg)},
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DefectSimError(f"API 返回 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DefectSimError(f"API 网络错误: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise DefectSimError(f"API 响应不是合法 JSON: {exc}") from exc


def _download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "LensDetect/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise DefectSimError(f"无法下载生成结果: {exc.reason}") from exc


def _find_images(payload: Any) -> list[dict[str, Any]]:
    """Recursively collect ``b64_json`` / ``url`` image nodes from the response."""
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "b64_json" in node and isinstance(node["b64_json"], str):
                found.append(node)
            elif "url" in node and isinstance(node["url"], str):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


def _image_from_node(node: dict[str, Any], timeout: float) -> Image.Image:
    if node.get("b64_json"):
        raw = base64.b64decode(node["b64_json"])
    elif node.get("url"):
        raw = _download(str(node["url"]), timeout)
    else:
        raise DefectSimError(f"无法识别的图像字段: {node}")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except OSError as exc:
        raise DefectSimError(f"无法解码 API 返回的图像: {exc}") from exc


def call_responses(cfg: DefectSimConfig, image_bytes: bytes) -> Image.Image:
    """POST an image-editing request to the Responses API and decode the result."""
    if not cfg.api_key.strip():
        raise DefectSimError("未配置 API Key，请在缺陷模拟页填写。")
    url = _endpoint_url(cfg, "responses")
    payload: dict[str, Any] = {
        "model": cfg.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": cfg.prompt},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(image_bytes).decode("ascii"),
                        "detail": "high",
                    },
                ],
            }
        ],
        "tools": [
            {
                "type": "image_generation",
                "sizes": [cfg.size],
                "quality": cfg.quality,
            }
        ],
    }
    response = _post_json(url, payload, cfg)
    nodes = _find_images(response)
    if not nodes:
        raise DefectSimError(f"API 未返回图像数据: {json.dumps(response)[:600]}")
    return _image_from_node(nodes[0], cfg.timeout)


def call_edits(cfg: DefectSimConfig, image_bytes: bytes, mask_bytes: bytes) -> Image.Image:
    """POST image+mask to the Images edits endpoint and decode the returned image."""
    if not cfg.api_key.strip():
        raise DefectSimError("未配置 API Key，请在缺陷模拟页填写。")
    url = _endpoint_url(cfg, "images/edits")
    fields = {
        "model": cfg.model,
        "prompt": cfg.prompt,
        "n": str(cfg.n),
        "size": cfg.size,
        "response_format": "b64_json",
    }
    files = {
        "image": ("image.png", "image/png", image_bytes),
        "mask": ("mask.png", "image/png", mask_bytes),
    }
    body, content_type = _multipart_body(fields, files)
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    for name, value in _auth_headers(cfg).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DefectSimError(f"API 返回 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DefectSimError(f"API 网络错误: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise DefectSimError(f"API 响应不是合法 JSON: {exc}") from exc
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise DefectSimError(f"API 未返回图像数据: {payload}")
    return _image_from_node(data[0], cfg.timeout)


def _newest_png_mtime(directory: Path) -> float:
    """Return the latest mtime among PNG files under *directory* (recursively)."""
    newest = 0.0
    if directory.is_dir():
        for path in directory.rglob("*.png"):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    return newest


def _codex_login_status() -> tuple[bool, str]:
    """Return whether Codex has a ChatGPT login and the raw status text.

    The ``image_gen`` tool only exists when codex is signed in to a ChatGPT
    (Pro) account; other providers (OpenAI API key, Anthropic, ...) do not
    expose image generation.  This lets us fail early with a clear Chinese
    message instead of forwarding codex's English error.
    """
    try:
        proc = subprocess.run(
            ["codex", "login", "status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as exc:
        return False, f"无法运行 codex: {exc}"
    except subprocess.TimeoutExpired:
        return False, "codex 登录状态检查超时。"
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0 and "chatgpt" in text.lower(), text.strip()


def check_codex_image_capability(timeout: float = 90.0) -> tuple[bool, str]:
    """Probe the current Codex session instead of inferring image access from login.

    Login is only a prerequisite: account, rollout, model, or client policy can still
    leave ``image_gen`` unavailable.  The probe asks the same non-interactive Codex
    environment used by defect generation to report its exposed tools.  It does not
    generate an image or consume an image-generation request.
    """
    if shutil.which("codex") is None:
        return False, "未找到 codex CLI。请先安装 Codex。"
    logged_in, status = _codex_login_status()
    if not logged_in:
        detail = status or "未登录"
        return False, f"Codex 未登录 ChatGPT 账号：{detail}"
    try:
        proc = subprocess.run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-rules",
                "-m",
                CODEX_AGENT_MODEL,
                "-c",
                'model_reasoning_effort="low"',
                CODEX_IMAGE_PROBE_PROMPT,
            ],
            capture_output=True,
            text=True,
            timeout=max(10.0, float(timeout)),
        )
    except OSError as exc:
        return False, f"无法运行 Codex 权限探针：{exc}"
    except subprocess.TimeoutExpired:
        return False, "Codex 生图权限检测超时。"
    stdout = (proc.stdout or "").strip()
    output = (stdout + "\n" + (proc.stderr or "")).strip()
    probe_lines = {line.strip() for line in stdout.splitlines()}
    if proc.returncode == 0 and "IMAGE_GEN_AVAILABLE" in probe_lines:
        return True, "当前 Codex 会话已提供 image_gen，可使用模型生图。"
    tail = output[-500:] or f"codex 退出码 {proc.returncode}"
    return False, f"当前 Codex 会话未提供 image_gen。探针输出：{tail}"


def _codex_image_failure_message(output: str) -> str:
    """Classify a failed Codex image call without misreporting auth state."""
    lowered = output.lower()
    tail = output[-800:]
    if "image generation failed" in lowered and "403 forbidden" in lowered:
        return (
            "Codex 已登录 ChatGPT，且会话已提供 image_gen，但图像生成服务返回 "
            "403 Forbidden。当前账号或工作区没有可用的图像生成权限；换终端或重复登录"
            "不会解决此权限拒绝。请确认登录的是具有图像生成功能的账号，或联系工作区"
            "管理员/OpenAI 支持开通权限。\n\nCodex 原始输出：\n"
            f"{tail}"
        )
    if "invalid_api_key" in lowered or "invalid api key" in lowered:
        return (
            "Codex 请求仍在使用无效的 API Key，而不是当前 ChatGPT 账号认证。"
            "请检查 Codex provider 配置。\n\nCodex 原始输出：\n"
            f"{tail}"
        )
    if "not exposed" in lowered or "image_gen_unavailable" in lowered:
        return (
            "当前 Codex 会话未提供 image_gen 工具。请确认 Codex 已使用 ChatGPT "
            "账号登录，并使用该账号支持的官方模型。\n\nCodex 原始输出：\n"
            f"{tail}"
        )
    return f"Codex 未生成图像。输出尾部：\n{tail}"


def call_codex_edit(
    cfg: DefectSimConfig,
    image: Image.Image,
    box: tuple[int, int, int, int],
    work_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> Image.Image:
    """Remove a boxed component with the ChatGPT Pro ``image_gen`` tool.

    This is the real model-image-generation path: it uses the operator's logged-in
    ChatGPT account (``codex exec``) rather than the relay key, so it works even
    when the relay has image generation disabled.

    The built-in ``image_gen`` edit redraws the whole input image, so sending it
    the full board makes it repaint far too much and smear details.  Instead we
    crop a small context window around the box, attach only that window to the
    prompt, and paste the boxed region back onto the original at full resolution.
    Returns the full-resolution board with only the box changed.

    ``work_dir`` is a writable directory inside the project (NOT a system temp
    dir): codex's read-only sandbox can only read its working directory, and on
    Windows the system temp folder is outside that, which made the agent unable
    to read the attached image.  We run codex with ``cwd`` set to ``work_dir`` so
    the input image is always inside the sandbox's readable workspace.

    ``on_progress`` (optional) receives short stage messages so a GUI can show
    that the (typically ~1.5 minute) model call is still running.
    """

    def report(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    if shutil.which("codex") is None:
        raise DefectSimError(
            "未找到 codex CLI。此方式需要安装并登录 codex（ChatGPT 账号）。"
        )
    logged_in, _ = _codex_login_status()
    if not logged_in:
        raise DefectSimError(
            "当前 codex 未登录 ChatGPT 账号，无法使用内置的图像生成（image_gen）工具。\n"
            "请先在命令行运行 `codex login` 并登录 ChatGPT（Pro），"
            "或把「修复方式」改为「本地 OpenCV 修复」做离线兜底。"
        )
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    gen_dir = codex_home / "generated_images"
    before = _newest_png_mtime(gen_dir)

    report("正在裁剪框选区域…")
    width, height = image.size
    left, top, right, bottom = box
    bw, bh = right - left, bottom - top
    # Context margin: enough surrounding board so the model can match texture,
    # but small enough that the model only redraws a focused region.
    margin = max(1, int(round(0.5 * max(bw, bh))))
    cl = max(0, left - margin)
    ct = max(0, top - margin)
    cr = min(width, right + margin)
    cb = min(height, bottom + margin)
    window = image.crop((cl, ct, cr, cb))

    # Scale the window so its long side is a good size for image_gen (keeps
    # texture detail while staying under the model's practical limits).
    win_w, win_h = window.size
    target = int(cfg.max_side)
    scale = min(1.0, target / max(win_w, win_h)) if max(win_w, win_h) > target else 1.0
    scaled = window.resize((max(1, int(win_w * scale)), max(1, int(win_h * scale))), Image.LANCZOS)

    instruction = (
        f"{cfg.prompt} "
        f"The attached image shows a single surface-mount component (a resistor or "
        f"capacitor) roughly in the centre between two silver solder pads. Use your "
        f"built-in image_gen tool to EDIT the attached image: remove only the "
        f"component body in the centre and fill that exact spot with the surrounding "
        f"bare printed-circuit-board surface so the component looks missing. Keep "
        f"the two silver solder pads, the silkscreen and everything else unchanged. "
        f"Save the edited image as a PNG file and report its absolute path."
    )
    timeout = max(cfg.timeout, 900.0)

    # Work inside the project output dir (readable by codex's sandbox), never a
    # system temp dir.  One subdir per run so we can clean up without races.
    job_dir = Path(work_dir) / f"codex_job_{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        input_path = job_dir / "input.png"
        input_path.write_bytes(_png_bytes(scaled))
        cmd = [
            "codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
            "-m", CODEX_AGENT_MODEL,
            "-c", 'model_reasoning_effort="low"',
            instruction,
            "-i", "input.png",
        ]
        report("正在调用 ChatGPT 模型生图（目标约 1 分钟）…")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=str(job_dir)
            )
        except subprocess.TimeoutExpired as exc:
            raise DefectSimError(f"codex 生图超时（>{int(timeout)}s）。") from exc
        except OSError as exc:
            raise DefectSimError(f"无法运行 codex: {exc}") from exc
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

    report("AI 已生成，正在写回原图…")
    edited_window: Image.Image | None = None
    for _ in range(10):
        newest = _newest_png_mtime(gen_dir)
        if newest > before + 0.5:
            for path in sorted(gen_dir.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
                if path.stat().st_mtime > before + 0.5:
                    edited_window = Image.open(path).convert("RGB")
                    break
            if edited_window is not None:
                break
        time.sleep(1.0)

    if edited_window is None:
        tail = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise DefectSimError(_codex_image_failure_message(tail))

    # Resize the model output back to the window size and crop out the box region.
    if edited_window.size != (win_w, win_h):
        edited_window = edited_window.resize((win_w, win_h), Image.LANCZOS)
    box_in_window = (left - cl, top - ct, right - cl, bottom - ct)
    piece = edited_window.crop(box_in_window).resize((bw, bh), Image.LANCZOS)

    output = image.copy()
    output.paste(piece, (left, top))
    return output


def normalize_box(box: Any, width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp a user box to image bounds and enforce a minimum size."""
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError) as exc:
        raise DefectSimError("框选区域无效。") from exc
    if len(values) != 4:
        raise DefectSimError("框选区域无效。")
    x0, y0, x1, y1 = values
    left = max(0.0, min(x0, x1))
    top = max(0.0, min(y0, y1))
    right = min(float(width), max(x0, x1))
    bottom = min(float(height), max(y0, y1))
    if right - left < 8.0 or bottom - top < 8.0:
        raise DefectSimError("框选区域太小，请框住一个完整的元器件。")
    return (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))


def _prepare_canvas(
    image: Image.Image, box: tuple[int, int, int, int], max_side: int
) -> tuple[Image.Image, dict[str, Any]]:
    """Downscale to max_side and pad to a square canvas; record the geometry."""
    width, height = image.size
    side = max(1, int(max_side))
    scale = min(1.0, side / max(width, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas_side = max(new_w, new_h)
    canvas = Image.new("RGB", (canvas_side, canvas_side), (255, 255, 255))
    ox = (canvas_side - new_w) // 2
    oy = (canvas_side - new_h) // 2
    canvas.paste(resized, (ox, oy))

    left, top, right, bottom = box
    sl, st = int(round(left * scale)), int(round(top * scale))
    sr, sb = int(round(right * scale)), int(round(bottom * scale))
    canvas_box = (ox + sl, oy + st, ox + sr, oy + sb)

    meta = {
        "source_size": [width, height],
        "scale": scale,
        "canvas_size": [canvas_side, canvas_side],
        "canvas_offset": [ox, oy],
        "resized_size": [new_w, new_h],
        "box": [left, top, right, bottom],
        "box_resized": [sl, st, sr, sb],
        "box_canvas": list(canvas_box),
    }
    return canvas, meta


def _mask_for_box(canvas: Image.Image, box_canvas: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("RGBA", canvas.size, (255, 255, 255, 255))
    ImageDraw.Draw(mask).rectangle(box_canvas, fill=(0, 0, 0, 0))
    return mask


def _annotate_box(canvas: Image.Image, box_canvas: tuple[int, int, int, int]) -> Image.Image:
    """Draw a clear red rectangle around the boxed region on the canvas."""
    annotated = canvas.copy()
    draw = ImageDraw.Draw(annotated)
    side = max(canvas.size)
    width = max(3, side // 256)
    draw.rectangle(box_canvas, outline=(255, 40, 40), width=width)
    return annotated


def _composite(original: Image.Image, result: Image.Image, meta: dict[str, Any]) -> Image.Image:
    """Paste only the edited box region back onto the full-resolution original."""
    ox, oy = meta["canvas_offset"]
    new_w, new_h = meta["resized_size"]
    if result.size != (meta["canvas_size"][0], meta["canvas_size"][1]):
        result = result.resize(
            (meta["canvas_size"][0], meta["canvas_size"][1]), Image.LANCZOS
        )
    crop = result.crop((ox, oy, ox + new_w, oy + new_h))
    sl, st, sr, sb = meta["box_resized"]
    piece = crop.crop((sl, st, sr, sb))
    left, top, right, bottom = meta["box"]
    piece = piece.resize((right - left, bottom - top), Image.LANCZOS)
    output = original.copy()
    output.paste(piece, (left, top))
    return output


def inpaint_missing(
    image: Image.Image,
    box: tuple[int, int, int, int],
    radius: int = 3,
    feather: int = 12,
) -> Image.Image:
    """Remove the boxed region locally with OpenCV inpainting (no API).

    The box interior is filled with the surrounding texture via
    ``cv2.inpaint`` (Telea) on a slightly dilated hard mask, and the result is
    blended back with a Gaussian-softened mask so the boundary has no hard seam.
    Runs at the image's full resolution.
    """
    arr = np.asarray(image.convert("RGB"))
    height, width = arr.shape[:2]
    left, top, right, bottom = box

    hard = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(hard, (left, top), (max(left, right - 1), max(top, bottom - 1)), 255, -1)

    pad = max(1, int(feather))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    dilated = cv2.dilate(hard, kernel)

    filled = cv2.inpaint(arr, dilated, max(1, int(radius)), cv2.INPAINT_TELEA)

    soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigmaX=max(1.0, feather / 2.0))
    soft = np.clip(soft / 255.0, 0.0, 1.0)

    blended = arr.astype(np.float32) * (1.0 - soft[..., None]) + filled.astype(np.float32) * soft[..., None]
    return Image.fromarray(np.clip(np.rint(blended), 0, 255).astype(np.uint8))


def simulate_defect(
    image_path: Path,
    box: Any,
    output_dir: Path,
    cfg: DefectSimConfig,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Run the full missing-component simulation for one boxed region.

    ``on_progress`` (optional) receives short stage messages so the GUI can keep
    the operator informed while the model call runs.
    """

    def report(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    image_path = Path(image_path)
    if not image_path.is_file():
        raise DefectSimError(f"图像不存在: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except OSError as exc:
        raise DefectSimError(f"无法读取图像 {image_path.name}: {exc}") from exc

    box = normalize_box(box, image.width, image.height)
    wire_api = (cfg.wire_api or "codex").strip().lower()

    if wire_api in ("inpaint", "local", "opencv"):
        report("正在用本地 OpenCV 修复抹去框选区域…")
        edited = inpaint_missing(image, box, cfg.inpaint_radius, cfg.inpaint_feather)
        meta = {
            "source_size": [image.width, image.height],
            "box": list(box),
            "method": "opencv_inpaint",
            "inpaint_radius": cfg.inpaint_radius,
            "inpaint_feather": cfg.inpaint_feather,
        }
    elif wire_api in ("codex", "chatgpt", "image_gen"):
        edited = call_codex_edit(cfg, image, box, output_dir, on_progress=on_progress)
        meta = {
            "source_size": [image.width, image.height],
            "box": list(box),
            "method": "codex_image_gen",
            "codex_agent_model": CODEX_AGENT_MODEL,
        }
    else:
        report("正在调用图像编辑 API 抹去框选区域…")
        canvas, meta = _prepare_canvas(image, box, cfg.max_side)
        box_canvas = tuple(meta["box_canvas"])
        if wire_api in ("responses", "response"):
            result = call_responses(cfg, _png_bytes(_annotate_box(canvas, box_canvas)))
        elif wire_api in ("edits", "edit", "images", "image"):
            result = call_edits(
                cfg, _png_bytes(canvas), _png_bytes(_mask_for_box(canvas, box_canvas))
            )
        else:
            raise DefectSimError(f"未知的 wire_api: {cfg.wire_api}")
        edited = _composite(image, result, meta)
        meta["method"] = wire_api

    report("正在保存结果图…")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    stem = f"{image_path.stem}_defect_{stamp}"
    edited_path = output_dir / f"{stem}.png"
    metadata_path = output_dir / f"{stem}.json"

    edited.save(edited_path)
    report = {
        "type": "pcb_defect_simulation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image": str(image_path.resolve()),
        "model": cfg.model,
        "wire_api": wire_api,
        "prompt": cfg.prompt,
        "box_px": meta["box"],
        "edited_image": str(edited_path),
        "preparation": meta,
    }
    metadata_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "edited": edited_path,
        "metadata": metadata_path,
    }
