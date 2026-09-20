#!/usr/bin/env python3
"""根据 Altium 贴片坐标文件(Pick & Place)生成元器件排布图。

用法:
    python3 plot_pnp_layout.py                              # 使用默认的 P3S2 坐标文件
    python3 plot_pnp_layout.py 坐标文件.txt -o layout.png    # 指定文件与输出
    python3 plot_pnp_layout.py --layer top --no-labels      # 只画顶层、不标位号

坐标文件需为 Altium 导出的 "Pick and Place Locations" 文本格式,
包含 Designator / Comment / Layer / Footprint / Center-X(mm) / Center-Y(mm) / Rotation 列。

说明:
- Rotation 按 Altium 习惯视为 PCB 坐标系(Y 轴向上)中的逆时针角度。
- 封装尺寸来自内置估算表与命名规则解析, 未知封装回退为默认小尺寸并在
  控制台列出, 可在 FOOTPRINT_SIZES 中补充。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "P3S2_XW_L3_V1.1B_YJZZB(385).txt"
DEFAULT_OUTPUT = None  # 默认为 <输入文件名>_layout.png

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

# ---- 板级校准表(仅针对本板/本封装库的修正, 通用规则见下方封装家族引擎) ----
# 封装名 -> (宽 mm, 高 mm), 为 Rotation=0 时沿封装库原始方向的尺寸。
FOOTPRINT_SIZES: dict[str, tuple[float, float]] = {
    "0603C": (1.6, 0.8),
    "0603R": (1.6, 0.8),
    "C0805": (2.0, 1.25),
    "F1206": (3.2, 1.6),
    "SOT23A": (2.9, 1.5),
    "SOT-23-6": (2.9, 1.6),
    "SOIC_8P": (4.9, 3.9),  # 实测: rot=180 时为横向(底面照片核对)
    "SOIC": (9.9, 3.9),  # 本板上 SOIC 封装为 UA7/UA9, 均为 16 脚 (AM26LV31E/32E)
    "μSOP8": (3.0, 4.9),
    "SOIC14": (8.7, 3.9),
    "FBGA_48": (6.0, 8.0),
    "T": (0.9, 0.9),
    "J30J-25ZKN-J": (22.4, 9.6),
    "J63A-2H2-025-231-TH": (14.7, 6.9),
    "J63A-2H2-037-231-TH": (18.5, 6.9),
    "IMX178LLJ-C": (15.0, 12.0),
    "LTC4362CDCB-2#TRMPBF": (2.0, 2.0),
    "SCH16TK01004": (13.4, 11.8),
}

# ---- 通用封装家族引擎: 按 封装家族+引脚数 推算常见 SMD 封装体尺寸 ----
# SOP 家族: 引脚数 -> (长mm 沿引脚排列方向, 宽mm 跨引脚方向), 默认横放
SOP_PINS = {
    6: (3.9, 2.6), 8: (4.9, 3.9), 14: (8.7, 3.9), 16: (9.9, 3.9),
    20: (12.8, 7.5), 24: (15.4, 7.5), 28: (17.9, 7.5), 32: (20.7, 7.5),
}
TSSOP_PINS = {
    8: (3.0, 3.0), 14: (5.0, 4.4), 16: (5.0, 4.4), 20: (6.5, 4.4),
    24: (7.8, 4.4), 28: (9.7, 4.4), 38: (12.5, 4.4), 48: (12.5, 6.1), 64: (17.0, 6.1),
}
SSOP_PINS = {16: (6.2, 5.3), 20: (7.2, 5.3), 24: (8.2, 5.3), 28: (10.2, 5.3)}
MSOP_PINS = {8: (4.9, 3.0), 10: (4.9, 3.0), 12: (4.9, 3.0)}
SOP_FAMILY = {"SOP": SOP_PINS, "SOIC": SOP_PINS, "SO": SOP_PINS,
              "VSOIC": SOP_PINS, "TSOP": SOP_PINS,
              "TSSOP": TSSOP_PINS, "SSOP": SSOP_PINS, "VSSOP": SSOP_PINS,
              "MSOP": MSOP_PINS, "MSOP8": MSOP_PINS}
# QFN/QFP: 引脚数 -> 边长 mm (正方形)
QFN_PINS = {8: 3.0, 10: 3.0, 12: 3.0, 16: 3.0, 20: 4.0, 24: 4.0, 32: 5.0,
            40: 6.0, 48: 7.0, 56: 8.0, 64: 9.0}
QFP_PINS = {32: 7.0, 44: 10.0, 48: 7.0, 64: 10.0, 80: 12.0, 100: 14.0, 144: 20.0}
# SOT/SOD 小封装
SOT_SIZES = {
    "SOT23": (2.9, 1.3), "SOT25": (2.9, 1.6), "SOT26": (2.9, 1.6),
    "SOT323": (2.0, 1.25), "SOT563": (1.6, 0.8), "SOT89": (4.5, 2.5),
    "SOT223": (6.5, 3.5), "SOT143": (2.9, 1.5),
}
SOD_SIZES = {
    "SOD323": (1.7, 1.3), "SOD523": (1.2, 0.8), "SOD723": (1.0, 0.6),
    "SOD123": (2.65, 1.5), "SMA": (4.3, 2.6), "SMB": (5.3, 3.6), "SMC": (7.1, 6.0),
}
IMPERIAL_CODES = {
    "0402": (1.0, 0.5), "0603": (1.6, 0.8), "0805": (2.0, 1.25),
    "1206": (3.2, 1.6), "1210": (3.2, 2.5), "2010": (5.0, 2.5), "2512": (6.3, 3.2),
}
METRIC_CODES = {"1005": "0402", "1608": "0603", "2012": "0805", "3216": "1206",
                "3225": "1210", "5025": "2010", "6332": "2512"}

DEFAULT_SIZE = (3.0, 2.0)

# Altium 的 Rotation 是相对封装库原始方向的; 表内封装在库里默认竖放(长边沿 Y 轴),
# 需 +90 才是板上实际方向。依据底面照片实测: SOIC/SOIC14 按 rot=90 均为横向摆放。
FOOTPRINT_ROTATION_OFFSETS: dict[str, int] = {
    "SOIC": 90,
    "SOIC14": 90,
}

# 位号前缀 -> 分类
CATEGORY_RULES = [
    (re.compile(r"^(A?R|RN)"), "resistor"),
    (re.compile(r"^FB"), "inductor"),
    (re.compile(r"^C"), "capacitor"),
    (re.compile(r"^(ESD|LED|D)"), "diode"),
    (re.compile(r"^(Q|TR)"), "transistor"),
    (re.compile(r"^(TP|T)"), "testpoint"),
    (re.compile(r"^F"), "fuse"),
    (re.compile(r"^(X|J|P|CN|K)"), "connector"),
    (re.compile(r"^(SW|S)"), "switch"),
    (re.compile(r"^(U|UA|UB|B|G|M|N|IC)"), "ic"),
]
CATEGORY_OVERRIDE = {"D5": "ic"}  # 位号特殊修正
CATEGORY_META = {  # key -> (中文名, 填充色)
    "resistor": ("电阻", (231, 106, 92)),
    "capacitor": ("电容", (96, 150, 214)),
    "inductor": ("电感/磁珠", (232, 158, 74)),
    "diode": ("二极管", (105, 178, 108)),
    "transistor": ("三极管", (163, 122, 199)),
    "ic": ("IC / 模块", (78, 121, 167)),
    "connector": ("连接器", (150, 116, 92)),
    "testpoint": ("测试点", (176, 190, 197)),
    "fuse": ("保险丝", (222, 92, 144)),
    "switch": ("开关", (129, 199, 132)),
    "other": ("其他", (158, 158, 49)),
}
LAYER_COLORS = {"TopLayer": (30, 30, 34), "BottomLayer": (30, 30, 34)}

ROW_RE = re.compile(
    r"^(?P<des>\S+)\s+(?P<comment>.*?)\s*(?P<layer>TopLayer|BottomLayer)\s+"
    r"(?P<fp>\S+)\s+(?P<cx>-?\d+(?:\.\d+)?)\s+(?P<cy>-?\d+(?:\.\d+)?)\s+"
    r"(?P<rot>\d+)\s*(?P<desc>.*)$"
)


@dataclass
class Part:
    designator: str
    comment: str
    layer: str
    footprint: str
    cx: float
    cy: float
    rotation: float
    size: tuple[float, float]
    category: str


def parse_pnp(path: Path, encoding: str) -> list[Part]:
    text = path.read_bytes().decode(encoding)
    lines = text.splitlines()
    try:
        header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Designator"))
    except StopIteration:
        raise RuntimeError(f"未找到表头行(Designator ...): {path}")
    parts, unknown = [], set()
    for ln in lines[header_idx + 1 :]:
        if not ln.strip():
            continue
        m = ROW_RE.match(ln)
        if not m:
            continue
        g = m.groupdict()
        fp = g["fp"]
        size, known = resolve_footprint_size(fp)
        if not known:
            unknown.add(fp)
        rotation = (float(g["rot"]) + FOOTPRINT_ROTATION_OFFSETS.get(fp, 0)) % 360
        parts.append(
            Part(
                designator=g["des"],
                comment=g["comment"].strip(),
                layer=g["layer"],
                footprint=fp,
                cx=float(g["cx"]),
                cy=float(g["cy"]),
                rotation=rotation,
                size=size,
                category=classify(g["des"]),
            )
        )
    if unknown:
        print(f"[提示] 以下封装未收录尺寸, 已按 {DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}mm 绘制, "
              f"可在 FOOTPRINT_SIZES 中补充: {', '.join(sorted(unknown))}", file=sys.stderr)
    if not parts:
        raise RuntimeError(f"未能从文件解析到任何元器件行: {path}")
    return parts


def resolve_footprint_size(fp: str) -> tuple[tuple[float, float], bool]:
    """返回 ((宽mm, 高mm), 是否已知尺寸)。按优先级: 板级表 -> 命名规则 -> 封装家族。"""
    if fp in FOOTPRINT_SIZES:
        return FOOTPRINT_SIZES[fp], True
    # JLC 风格: SOD-323_L1.7-W1.3-...
    m = re.search(r"L(\d+(?:\.\d+)?)-W(\d+(?:\.\d+)?)", fp)
    if m:
        return (float(m.group(1)), float(m.group(2))), True
    # 乘号风格: SOP4_-_3.2*2.5 / FCS325_11*13.5
    m = re.search(r"(\d+(?:\.\d+)?)\*(\d+(?:\.\d+)?)", fp)
    if m:
        return (float(m.group(1)), float(m.group(2))), True
    # BGA 命名: ..._625X625X192 -> 6.25 x 6.25 mm
    m = re.search(r"(\d{3,4})X\1(?=X|$|[^0-9])", fp)
    if m:
        v = int(m.group(1)) / 100.0
        return (v, v), True
    # SOT 家族 (先于 SOP): SOT-23 / SOT23-5 / SOT-223 ... (SOT23-5/6 按 SOT25/26)
    m = re.search(r"(?:^|[^A-Z])SOT[-_ ]?(\d{2,3})(?:[-_ ]?([3-8]))?(?=[^0-9]|$)", fp)
    if m:
        # SOT23-5/-6 实为 SOT25/SOT26 (5/6 脚加宽)
        key = f"SOT2{m.group(2)}" if m.group(2) in ("5", "6") else f"SOT{m.group(1)}"
        size = SOT_SIZES.get(key)
        if size:
            return size, True
    # SOD / SMA / SMB / SMC 二极管 (去掉 -_ 空格后匹配)
    nfp = re.sub(r"[-_ ]", "", fp)
    for name in ("SMC", "SMB", "SMA", "SOD123", "SOD323", "SOD523", "SOD723"):
        if re.search(rf"(?:^|[^A-Z]){name}(?=[^A-Z0-9]|$)", nfp):
            return SOD_SIZES[name], True
    # QFN / DFN (正方形, 按引脚数)
    m = re.search(r"(?:^|[^A-Z])(?:W|V|L|U)?Q?FN[-_ ]?(\d{1,3})(?=[^0-9]|$)",
                  re.sub(r"[-_ ]", "", fp).upper().replace("DFN", "QFN"))
    if m:
        side = QFN_PINS.get(int(m.group(1)))
        if side:
            return (side, side), True
    # QFP (正方形, 按引脚数)
    m = re.search(r"(?:^|[^A-Z])(?:L|T|E|P)?QFP[-_ ]?(\d{1,3})(?=[^0-9]|$)", fp.upper())
    if m:
        side = QFP_PINS.get(int(m.group(1)))
        if side:
            return (side, side), True
    # SOP 家族: SOIC8 / SOP-16 / TSSOP-20 / MSOP10 ...
    m = re.search(r"(?:^|[^A-Z])(SOIC|TSSOP|VSSOP|VSOIC|TSOP|SSOP|MSOP|SOP|SO)"
                  r"[-_ ]?(\d{1,3})(?=[^0-9]|$)", fp)
    if m:
        table = SOP_FAMILY.get(m.group(1))
        if table:
            size = table.get(int(m.group(2)))
            if size:
                return size, True
    # 英制封装代码: 0603 / C0805 / F1206 / 0603C ...
    m = re.match(r"^[A-Z]*(0402|0603|0805|1206|1210|2010|2512)[A-Z]*$", fp)
    if m:
        return IMPERIAL_CODES[m.group(1)], True
    # 公制封装代码: 1608 / 3216 ...
    m = re.match(r"^[A-Z]*(1005|1608|2012|3216|3225|5025|6332)[A-Z]*$", fp)
    if m:
        return IMPERIAL_CODES[METRIC_CODES[m.group(1)]], True
    return DEFAULT_SIZE, False


def load_size_overrides(csv_path: Path) -> None:
    """从 CSV 导入精确尺寸, 格式: footprint,w_mm,h_mm,rotation_offset(可选)。
    覆盖内置表, 优先级最高。可由 Altium 脚本导出生成。"""
    import csv
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith(("#", "footprint", "Footprint")):
                continue
            fp, w, h = row[0].strip(), float(row[1]), float(row[2])
            FOOTPRINT_SIZES[fp] = (w, h)
            if len(row) > 3 and row[3].strip():
                FOOTPRINT_ROTATION_OFFSETS[fp] = int(row[3])


def classify(designator: str) -> str:
    if designator in CATEGORY_OVERRIDE:
        return CATEGORY_OVERRIDE[designator]
    for pattern, cat in CATEGORY_RULES:
        if pattern.match(designator):
            return cat
    return "other"


# ---------------------------------------------------------------- 渲染 ---

def load_font(size: int, bold: bool = False):
    candidates = FONT_CANDIDATES if not bold else FONT_CANDIDATES[::-1]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def darken(rgb, factor=0.55):
    return tuple(int(c * factor) for c in rgb)


def part_corners(p: Part) -> list[tuple[float, float]]:
    """元器件外框四角的世界坐标(mm, Y 向上), 已含旋转。"""
    w, h = p.size
    th = math.radians(p.rotation)
    cos_t, sin_t = math.cos(th), math.sin(th)
    corners = []
    for lx, ly in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        corners.append((p.cx + lx * cos_t - ly * sin_t, p.cy + lx * sin_t + ly * cos_t))
    return corners


def board_extent(parts: list[Part], margin_mm: float = 1.0):
    xs, ys = [], []
    for p in parts:
        for x, y in part_corners(p):
            xs.append(x)
            ys.append(y)
    return min(xs) - margin_mm, min(ys) - margin_mm, max(xs) + margin_mm, max(ys) + margin_mm


class Panel:
    """单面(顶层或底层)的绘制面板, 含 mm -> 像素换算。"""

    def __init__(self, draw: ImageDraw.ImageDraw, extent, ppm: float, ssa: int,
                 origin_px: tuple[int, int], inner_pad: int):
        self.draw = draw
        self.x0, self.y0, self.x1, self.y1 = extent  # mm
        self.ppm = ppm
        self.ssa = ssa
        self.ox, self.oy = origin_px  # 板框左上角在画布上的像素位置
        pad = inner_pad * ssa
        self.board_px = (self.ox + pad, self.oy + pad,
                         self.ox + (self.x1 - self.x0) * ppm * ssa - pad,
                         self.oy + (self.y1 - self.y0) * ppm * ssa - pad)

    def to_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return (self.ox + (x_mm - self.x0) * self.ppm * self.ssa,
                self.oy + (self.y1 - y_mm) * self.ppm * self.ssa)


def draw_panel(panel: Panel, parts: list[Part], layer: str, show_labels: bool,
               fonts: dict, large_font_px: int):
    d = panel.draw
    # 背景与网格
    panel_right = panel.ox + (panel.x1 - panel.x0) * panel.ppm * panel.ssa
    panel_bottom = panel.oy + (panel.y1 - panel.y0) * panel.ppm * panel.ssa
    d.rectangle([panel.ox, panel.oy, panel_right, panel_bottom], fill=(252, 252, 250))
    step = 5
    gx = math.ceil(panel.x0 / step) * step
    while gx <= panel.x1:
        p0, p1 = panel.to_px(gx, panel.y1), panel.to_px(gx, panel.y0)
        major = gx % 10 == 0
        d.line([p0, p1], fill=(225, 225, 220) if not major else (208, 208, 202),
               width=1 * panel.ssa)
        gx += step
    gy = math.ceil(panel.y0 / step) * step
    while gy <= panel.y1:
        p0, p1 = panel.to_px(panel.x0, gy), panel.to_px(panel.x1, gy)
        major = gy % 10 == 0
        d.line([p0, p1], fill=(225, 225, 220) if not major else (208, 208, 202),
               width=1 * panel.ssa)
        gy += step
    # 坐标原点
    ox, oy = panel.to_px(0, 0)
    d.line([ox - 6 * panel.ssa, oy, ox + 6 * panel.ssa, oy], fill=(120, 120, 120), width=panel.ssa)
    d.line([ox, oy - 6 * panel.ssa, ox, oy + 6 * panel.ssa], fill=(120, 120, 120), width=panel.ssa)

    # 板框
    bx0, by0, bx1, by1 = panel.board_px
    d.rectangle([bx0, by0, bx1, by1], outline=(90, 90, 90), width=2 * panel.ssa)

    # 先画大件(连接器/IC), 后画小件, 避免小件被完全盖住
    layer_parts = sorted(
        (p for p in parts if p.layer == layer),
        key=lambda p: -(p.size[0] * p.size[1]),
    )
    label_font = fonts["label"]
    sub_font = fonts["sub"]
    for p in layer_parts:
        color = CATEGORY_META[p.category][1]
        pts = [panel.to_px(x, y) for x, y in part_corners(p)]
        if p.footprint == "T":  # 测试点画成圆
            r = p.size[0] / 2 * panel.ppm * panel.ssa
            cx, cy = panel.to_px(p.cx, p.cy)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                      outline=darken(color), width=panel.ssa)
        else:
            d.polygon(pts, fill=color, outline=darken(color))
        if show_labels:
            cx, cy = panel.to_px(p.cx, p.cy)
            d.text((cx, cy), p.designator, font=label_font, fill=(20, 20, 20),
                   anchor="mm")
            # 较大的 IC 追加型号
            if p.size[0] * p.size[1] >= 25 and p.comment:
                name = p.comment[:18]
                d.text((cx, cy + large_font_px * 0.9 * panel.ssa), name, font=sub_font,
                       fill=(60, 60, 60), anchor="mm")

    # 面板标题与刻度
    title = "顶层 TopLayer" if layer == "TopLayer" else "底层 BottomLayer"
    d.text((panel.ox + 8 * panel.ssa, panel.oy - 24 * panel.ssa),
           f"{title}  ({len(layer_parts)} 个)", font=fonts["panel_title"],
           fill=LAYER_COLORS[layer], anchor="lt")
    tick_font = fonts["tick"]
    tx = math.ceil(panel.x0 / 10) * 10
    while tx <= panel.x1:
        px, py = panel.to_px(tx, panel.y0)
        d.text((px, py + 3 * panel.ssa), f"{tx:g}", font=tick_font,
               fill=(110, 110, 110), anchor="ma")
        tx += 10
    ty = math.ceil(panel.y0 / 10) * 10
    while ty <= panel.y1:
        px, py = panel.to_px(panel.x0, ty)
        d.text((px - 3 * panel.ssa, py), f"{ty:g}", font=tick_font,
               fill=(110, 110, 110), anchor="rm")
        ty += 10
    return len(layer_parts)


def render(parts: list[Part], out_path: Path, ppm: float, show_labels: bool,
           layers: list[str], title: str, board_mm: tuple[float, float, float, float]):
    ssa = 2  # 超采样倍数, 抗锯齿
    extent = board_extent(parts)
    board_w_mm = extent[2] - extent[0]
    board_h_mm = extent[3] - extent[1]

    pad = 26          # 画布外边距(逻辑像素)
    inner = 10        # 板框与面板边缘间距
    title_h = 84
    panel_gap = 46
    # 每个面板按本面元器件范围紧贴裁剪, 去掉大面积空白
    layer_extents = {}
    panel_sizes = {}
    for layer in layers:
        lp = [p for p in parts if p.layer == layer]
        ext = board_extent(lp)
        layer_extents[layer] = ext
        panel_sizes[layer] = (int((ext[2] - ext[0]) * ppm) + inner * 2,
                              int((ext[3] - ext[1]) * ppm) + inner * 2)
    panel_w = max(w for w, _ in panel_sizes.values())

    n_panels = len(layers)
    canvas_w = pad + panel_w + pad
    canvas_h = pad + title_h + sum(h for _, h in panel_sizes.values()) \
        + (n_panels - 1) * panel_gap + pad
    canvas = Image.new("RGB", (canvas_w * ssa, canvas_h * ssa), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    large = max(10, int(ppm * 0.55))
    fonts = {
        "title": load_font(30 * ssa, bold=True),
        "subtitle": load_font(15 * ssa),
        "panel_title": load_font(17 * ssa, bold=True),
        "label": load_font(large * ssa),
        "sub": load_font(max(9, int(large * 0.62)) * ssa),
        "tick": load_font(11 * ssa),
    }

    # 总标题
    draw.text((pad * ssa, 18 * ssa), title, font=fonts["title"], fill=(25, 25, 25))
    n_top = sum(1 for p in parts if p.layer == "TopLayer")
    n_bot = len(parts) - n_top
    subtitle = (f"共 {len(parts)} 个元器件   顶层 {n_top} / 底层 {n_bot}   "
                f"板框约 {board_w_mm - 6:.0f} x {board_h_mm - 6:.0f} mm   "
                f"比例 1mm = {ppm:g}px")
    draw.text((pad * ssa, 56 * ssa), subtitle, font=fonts["subtitle"], fill=(90, 90, 90))

    oy = pad + title_h
    for layer in layers:
        panel = Panel(draw, layer_extents[layer], ppm, ssa, (pad * ssa, oy * ssa), inner)
        draw_panel(panel, parts, layer, show_labels, fonts, large)
        oy += panel_sizes[layer][1] + panel_gap

    canvas = canvas.resize((canvas_w, canvas_h), Image.LANCZOS)
    canvas.save(out_path)
    print(f"已生成: {out_path}  ({canvas_w}x{canvas_h}px)")


def main():
    ap = argparse.ArgumentParser(description="根据 Altium 贴片坐标文件生成元器件排布图")
    ap.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                    help=f"坐标文件路径 (默认: {DEFAULT_INPUT.name})")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                    help="输出图片路径 (默认: <输入文件名>_layout.png)")
    ap.add_argument("--px-per-mm", type=float, default=18.0, help="每毫米像素数 (默认 18)")
    ap.add_argument("--layer", choices=["both", "top", "bottom"], default="both",
                    help="绘制哪一面 (默认 both, 顶层+底层同图)")
    ap.add_argument("--no-labels", action="store_true", help="不绘制位号文字")
    ap.add_argument("--sizes", type=Path, default=None,
                    help="精确尺寸 CSV (footprint,w,h,rot_offset 可选), 覆盖内置估算")
    ap.add_argument("--encoding", default="gb18030", help="坐标文件编码 (默认 gb18030)")
    args = ap.parse_args()

    if not args.input.exists():
        ap.error(f"坐标文件不存在: {args.input}")
    if args.sizes:
        if not args.sizes.exists():
            ap.error(f"尺寸 CSV 不存在: {args.sizes}")
        load_size_overrides(args.sizes)
    out = args.output or args.input.with_name(args.input.stem + "_layout.png")

    parts = parse_pnp(args.input, args.encoding)
    layers = {"both": ["TopLayer", "BottomLayer"], "top": ["TopLayer"],
              "bottom": ["BottomLayer"]}[args.layer]
    title = f"元器件排布图  {args.input.name}"
    render(parts, out, args.px_per_mm, not args.no_labels, layers, title,
           board_extent(parts))


if __name__ == "__main__":
    main()
