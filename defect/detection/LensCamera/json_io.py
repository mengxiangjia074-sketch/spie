import json


def load_json_object(json_path, root_description="JSON"):
    with open(str(json_path), "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        raise ValueError("{} root must be an object".format(root_description))

    return data


def write_json(path, data, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=indent, sort_keys=False)
        file_obj.write("\n")

