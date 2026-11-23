import importlib.util
from pathlib import Path
import os

def get_custom_impl_module(name="go2trot_base.py"):
    repo_path = Path(__file__).resolve().parent.parent  
    path = Path(repo_path / "custom_implementation" / name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load file at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

implementations = {
    "trot": "go2trot_base.py",
    "phase1": "phase1/go2osc_deploy.py",
    "phase2": "phase2/go2osc_deploy.py",
}

phase = os.environ.get("GO2_PHASE", "trot")

custom_impl = get_custom_impl_module(implementations[phase])