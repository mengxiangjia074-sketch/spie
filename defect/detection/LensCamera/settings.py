import json


class SettingsError(ValueError):
    pass


def load_jsonnet_settings(settings_path, settings_name=None):
    settings_name = settings_name or str(settings_path)
    try:
        import _jsonnet
    except ImportError:
        return load_jsonnet_settings_fallback(settings_path, settings_name)

    rendered = _jsonnet.evaluate_file(str(settings_path))
    return json.loads(rendered)


def load_jsonnet_settings_fallback(settings_path, settings_name=None):
    settings_name = settings_name or str(settings_path)
    with open(str(settings_path), "r", encoding="utf-8-sig") as file_obj:
        text = file_obj.read()

    stripped = strip_jsonnet_comments(text)
    stripped = remove_trailing_commas(stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SettingsError(
            "{} uses Jsonnet syntax that needs the Jsonnet Python binding. "
            "Install it with: python -m pip install jsonnet".format(settings_name)
        ) from exc


def strip_jsonnet_comments(text):
    result = []
    index = 0
    in_string = False
    escape = False
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue

        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (
                text[index] == "*" and text[index + 1] == "/"
            ):
                index += 1
            index += 2
            continue

        result.append(char)
        index += 1

    return "".join(result)


def remove_trailing_commas(text):
    result = []
    index = 0
    in_string = False
    escape = False
    while index < len(text):
        char = text[index]

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue

        result.append(char)
        index += 1

    return "".join(result)


class Settings:
    def __init__(self, settings_path, settings_name=None):
        self.settings_path = settings_path
        self.settings_name = settings_name or str(settings_path)
        self.values = self._load()

    def _load(self):
        if not self.settings_path.is_file():
            raise SettingsError("{} not found".format(self.settings_name))

        settings = load_jsonnet_settings(self.settings_path, self.settings_name)
        if not isinstance(settings, dict):
            raise SettingsError("{} root must be a JSON object".format(self.settings_name))
        return settings

    def value(self, name):
        if name not in self.values:
            raise SettingsError(
                "{} missing required setting {!r}".format(self.settings_name, name)
            )

        item = self.values[name]
        if isinstance(item, dict) and "value" in item:
            return item["value"]
        return item

