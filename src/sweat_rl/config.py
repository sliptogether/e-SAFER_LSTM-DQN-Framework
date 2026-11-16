from pathlib import Path
import yaml

def load_config(path):
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    if str(cfg.get("project", {}).get("device", "cpu")).lower() != "cpu":
        raise ValueError("This release implements CPU execution; set project.device to cpu.")
    cfg["_root"] = str(path.parent.parent)
    return cfg

def project_path(cfg, *parts):
    return Path(cfg["_root"], *parts)
