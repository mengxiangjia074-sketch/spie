from . import paths
from .json_io import load_json_object


MOTOR_ORDER = ("zoom", "focus", "iris")
DEFAULT_JSON_FILE = paths.DEFAULT_LENS_FILE


def parse_device_number(value):
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("auto", "default"):
            return None

    return parse_position("device", value)


def parse_position(name, value):
    if isinstance(value, dict):
        for key in ("position", "address", "addr", "value"):
            if key in value:
                return parse_position(name, value[key])
        raise ValueError(
            "{} must contain one of: position, address, addr, value".format(name)
        )

    if isinstance(value, bool):
        raise ValueError("{} must be an integer address, not a boolean".format(name))

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("{} must not be an empty string".format(name))
        try:
            return int(text, 0)
        except ValueError as exc:
            raise ValueError(
                "{} must be an integer address, got {!r}".format(name, value)
            ) from exc

    raise ValueError("{} must be an integer address".format(name))


def resolve_lens_json_path(json_file, default_json_file=DEFAULT_JSON_FILE):
    return paths.resolve_project_default_file(json_file, default_json_file)


def read_device_number(config):
    device_number = config.get("device", config.get("device_number"))
    return parse_device_number(device_number)


def read_lens_targets(config):
    targets = {}
    for name in MOTOR_ORDER:
        if name in config and config[name] is not None:
            targets[name] = parse_position(name, config[name])
    return targets


def load_target_config(json_path):
    config = load_json_object(json_path)
    targets = read_lens_targets(config)
    if not targets:
        raise ValueError("JSON must contain at least one of: zoom, focus, iris")
    return read_device_number(config), targets

