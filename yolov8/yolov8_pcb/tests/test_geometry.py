from pcb_seg.geometry import Tile, box_iou, clip_polygon_to_tile, generate_tiles, polygon_area


def test_generate_tiles_covers_image_edges():
    tiles = generate_tiles(3840, 2160, 1024, 256)
    assert len(tiles) == 15
    assert tiles[0] == Tile(0, 0, 1024, 1024)
    assert tiles[-1] == Tile(2816, 1136, 3840, 2160)


def test_clip_polygon_to_tile_returns_local_coordinates():
    tile = Tile(100, 100, 200, 200)
    clipped = clip_polygon_to_tile([(50, 120), (150, 120), (150, 180), (50, 180)], tile)
    assert polygon_area(clipped) == 3000.0
    assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in clipped)


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
