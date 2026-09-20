def build_sample_points(
    min_addr,
    max_addr,
    steps,
    error_type=ValueError,
    range_name="sample",
):
    if steps < 1:
        raise error_type("sample steps must be at least 1")
    if min_addr > max_addr:
        raise error_type("{} min must be <= {} max".format(range_name, range_name))
    if steps == 1 or min_addr == max_addr:
        return [int(min_addr)]

    span = max_addr - min_addr
    points = []
    for index in range(steps):
        point = int(round(min_addr + (span * index / float(steps - 1))))
        point = max(min_addr, min(max_addr, point))
        if point not in points:
            points.append(point)
    return points
