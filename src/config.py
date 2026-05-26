import yaml

_REQUIRED_KEYS = [
    "embedding_model",
    "generation_model",
    "chroma_path",
    "chunk_size",
    "chunk_overlap",
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")

    return cfg
