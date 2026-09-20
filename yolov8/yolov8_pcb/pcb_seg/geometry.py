"""Geometry helpers shared by dataset preparation and tiled inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


Point = tuple[float, float]


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def generate_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if tile_size < 64:
        raise ValueError("tile_size must be at least 64")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    return [
        Tile(x, y, min(width, x + tile_size), min(height, y + tile_size))
        for y in _axis_starts(height, tile_size, overlap)
        for x in _axis_starts(width, tile_size, overlap)
    ]


def _clip_edge(
    points: Sequence[Point],
    inside,
    intersect,
) -> list[Point]:
    if not points:
        return []
    output: list[Point] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def clip_polygon_to_tile(points: Iterable[Point], tile: Tile) -> list[Point]:
    """Clip a polygon to a tile and return coordinates local to that tile."""
    polygon = [(float(x), float(y)) for x, y in points]

    def vertical(a: Point, b: Point, x: float) -> Point:
        if b[0] == a[0]:
            return x, a[1]
        ratio = (x - a[0]) / (b[0] - a[0])
        return x, a[1] + ratio * (b[1] - a[1])

    def horizontal(a: Point, b: Point, y: float) -> Point:
        if b[1] == a[1]:
            return a[0], y
        ratio = (y - a[1]) / (b[1] - a[1])
        return a[0] + ratio * (b[0] - a[0]), y

    polygon = _clip_edge(
        polygon, lambda p: p[0] >= tile.x1, lambda a, b: vertical(a, b, tile.x1)
    )
    polygon = _clip_edge(
        polygon, lambda p: p[0] <= tile.x2, lambda a, b: vertical(a, b, tile.x2)
    )
    polygon = _clip_edge(
        polygon, lambda p: p[1] >= tile.y1, lambda a, b: horizontal(a, b, tile.y1)
    )
    polygon = _clip_edge(
        polygon, lambda p: p[1] <= tile.y2, lambda a, b: horizontal(a, b, tile.y2)
    )
    return [(x - tile.x1, y - tile.y1) for x, y in polygon]


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2.0


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )
    return intersection / max(1e-9, area_a + area_b - intersection)
