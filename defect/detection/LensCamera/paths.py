from pathlib import Path


# The package lives at <project>/detection/LensCamera, so the project root is
# three levels above this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = PROJECT_ROOT
ORIGINAL_CWD = Path.cwd()
CONTROLLER_DIR = PACKAGE_DIR / "LensConnect_Controller"
DEFAULT_LENS_FILE = "lens_target.json"


def resolve_path(path_text, default_to_project=False):
    path = Path(path_text)
    if path.is_absolute():
        return path

    base_dir = PROJECT_ROOT if default_to_project else ORIGINAL_CWD
    return base_dir / path


def resolve_project_path(path_text):
    """Resolve a relative path against the project root, independent of CWD.

    Output paths must never depend on the directory a script is run from, so
    calibration results always land in the same place (e.g. working_data/...).
    """
    return resolve_path(path_text, default_to_project=True)


def resolve_project_default_file(path_text, default_file):
    path = Path(path_text)
    if path.is_absolute():
        return path

    if path_text == default_file:
        for candidate in (
            PROJECT_ROOT / default_file,
            PROJECT_ROOT / "config" / "autofocus" / default_file,
        ):
            if candidate.is_file():
                return candidate

    return ORIGINAL_CWD / path
