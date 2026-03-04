from pathlib import Path


def resolve_checkpoint_path(base_dir: Path, cfg_value: str) -> str:
    """统一解析 checkpoint 路径。

    - 支持历史配置里误写的 r'...'/r"..." 文本
    - 若配置为绝对路径，则直接使用
    - 若配置为相对路径，则拼接到 base_dir
    """
    if cfg_value is None:
        return ""

    raw = str(cfg_value).strip()
    if (raw.startswith("r'") and raw.endswith("'")) or (raw.startswith('r"') and raw.endswith('"')):
        raw = raw[2:-1]

    candidate = Path(raw.strip()).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())
