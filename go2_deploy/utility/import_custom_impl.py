import importlib.util
import os
from pathlib import Path


def get_custom_impl_module(name=None):
    if name is None:
        name = os.environ.get("GO2_CUSTOM_IMPLEMENTATION", "go2trot_base.py")
    path_name = Path(name)
    if path_name.name != name or "\\" in name:
        raise ValueError("Custom implementation must be a filename, not a path")
    if not path_name.suffix:
        name += ".py"
        path_name = Path(name)
    if path_name.suffix != ".py" or not path_name.stem.isidentifier():
        raise ValueError("Custom implementation must be a Python module filename")

    repo_path = Path(__file__).resolve().parent.parent
    path = repo_path / "custom_implementation" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load file at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


custom_impl = get_custom_impl_module()
