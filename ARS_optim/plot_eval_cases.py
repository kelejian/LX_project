import argparse
import math
import textwrap
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.settings import FEATURE_ORDER


METRIC_SUFFIXES = {
    "HIC",
    "Dmax",
    "Nij",
    "Phead",
    "Pchest",
    "Pneck",
    "JointRisk",
    "AIS_head",
    "AIS_chest",
    "AIS_neck",
    "MAIS",
}
UNIT_MAP = {
    "impact_velocity": "km/h",
    "impact_angle": "deg",
    "overlap": "-",
    "LL1": "kN",
    "LL2": "kN",
    "BTF": "ms",
    "LLATTF": "ms",
    "AFT": "ms",
    "SP": "mm",
    "SH": "mm",
    "RA": "deg",
    "HIC": "-",
    "Dmax": "mm",
    "Nij": "-",
    "Phead": "prob",
    "Pchest": "prob",
    "Pneck": "prob",
    "JointRisk": "prob",
    "AIS_head": "level",
    "AIS_chest": "level",
    "AIS_neck": "level",
    "MAIS": "level",
}
CONTROL_FIG_NAME = "01_control_compare.png"
SCALAR_FIG_NAME = "02_injury_scalar_compare.png"
AIS_FIG_NAME = "03_ais_compare.png"
RISK_FIG_NAME = "04_risk_compare.png"
OPT_STAGE_CANDIDATES = ("Opt1", "Opt2")
RESULT_METRIC_NAMES = (
    "HIC",
    "Dmax",
    "Nij",
    "Phead",
    "Pchest",
    "Pneck",
    "JointRisk",
    "AIS_head",
    "AIS_chest",
    "AIS_neck",
    "MAIS",
)
TRUE_METRIC_COLUMNS = [
    "True_HIC", "True_Dmax", "True_Nij",
    "True_Phead", "True_Pchest", "True_Pneck", "True_JointRisk",
    "True_AIS_head", "True_AIS_chest", "True_AIS_neck", "True_MAIS",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-case comparison charts from ARS_optim evaluation CSV")
    parser.add_argument("--eval_csv", required=True, type=str, help="Absolute path to evaluation_results.csv or custom eval csv")
    parser.add_argument("--case_ids", required=True, nargs="+", help="One or more case_id values to plot, e.g. --case_ids (1,2,6)")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    return parser.parse_args()


def _safe_case_folder_name(case_id: str) -> str:
    text = str(case_id)
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
    return f"case_{safe}"


def _require_columns(df: pd.DataFrame, columns: Iterable[str], scope: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{scope} 缺少必需列: {missing}")


def _detect_stage_names_and_controls(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    base_control_names = {column[5:] for column in df.columns if column.startswith("Base_")} - METRIC_SUFFIXES
    if not base_control_names:
        raise ValueError("未能从结果 CSV 中识别到任何可调 control 参数列")
    stages = ["Base"]
    for metric_name in RESULT_METRIC_NAMES:
        if f"Base_{metric_name}" not in df.columns:
            raise ValueError(f"评估结果 CSV 缺少 Base 阶段指标列: Base_{metric_name}")

    for stage_name in OPT_STAGE_CANDIDATES:
        prefix = f"{stage_name}_"
        stage_control_names = {column[len(prefix):] for column in df.columns if column.startswith(prefix)} - METRIC_SUFFIXES
        has_stage_metrics = any(f"{stage_name}_{metric_name}" in df.columns for metric_name in RESULT_METRIC_NAMES)
        if not stage_control_names and not has_stage_metrics:
            continue
        if stage_control_names != base_control_names:
            base_only = sorted(base_control_names - stage_control_names)
            stage_only = sorted(stage_control_names - base_control_names)
            raise ValueError(
                f"评估结果 CSV 的 Base_/{stage_name}_ control 列不一致："
                f" 仅 Base 存在={base_only or []}, 仅 {stage_name} 存在={stage_only or []}"
            )
        _require_columns(df, [f"{stage_name}_{name}" for name in sorted(base_control_names)], f"{stage_name} control 绘图")
        _require_columns(df, [f"{stage_name}_{metric_name}" for metric_name in RESULT_METRIC_NAMES], f"{stage_name} 指标绘图")
        stages.append(stage_name)

    if len(stages) == 1:
        raise ValueError("评估结果 CSV 中不存在任何 Opt1_/Opt2_ 阶段列")
    control_names = sorted(base_control_names, key=lambda name: FEATURE_ORDER.index(name) if name in FEATURE_ORDER else name)
    return stages, control_names


def _detect_context_names(df: pd.DataFrame, control_names: List[str]) -> List[str]:
    control_set = set(control_names)
    context_names = []
    for name in FEATURE_ORDER:
        has_context_column = name in df.columns
        has_control_pair = name in control_set
        if has_context_column and has_control_pair:
            raise ValueError(f"参数 {name} 同时以 context 原列和 Base_/Opt1_/Opt2_ control 对出现，结果 CSV 结构存在歧义")
        if not has_context_column and not has_control_pair:
            raise ValueError(f"参数 {name} 既没有 context 原列，也没有成对的 Base_/Opt1_/Opt2_ control 列")
        if has_context_column:
            context_names.append(name)
    return context_names


def _format_context_text(row: pd.Series, context_names: List[str]) -> str:
    items = []
    for name in context_names:
        value = row[name]
        unit = UNIT_MAP.get(name, "-")
        if pd.isna(value):
            rendered = f"{name}=NaN"
        elif unit == "-":
            rendered = f"{name}={value:g}"
        else:
            rendered = f"{name}={value:g} {unit}"
        items.append(rendered)
    return "\n".join(textwrap.wrap(", ".join(items), width=120, break_long_words=False, break_on_hyphens=False))


def _subplot_grid(count: int) -> tuple[int, int]:
    cols = min(3, max(1, count))
    rows = int(math.ceil(count / cols))
    return rows, cols


def _plot_bar_group(ax, labels: List[str], values: List[float], title: str, unit: str, value_fmt: str) -> None:
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(values)], width=0.6)
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            value_fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _save_control_figure(
    row: pd.Series,
    control_names: List[str],
    context_names: List[str],
    stage_names: List[str],
    output_dir: Path,
    dpi: int,
) -> None:
    required_columns = [f"{stage_name}_{name}" for stage_name in stage_names for name in control_names]
    _require_columns(row.to_frame().T, required_columns, "control 绘图")
    rows, cols = _subplot_grid(len(control_names))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for axis, control_name in zip(axes_flat, control_names):
        unit = UNIT_MAP.get(control_name, "-")
        labels = stage_names
        values = [float(row[f"{stage_name}_{control_name}"]) for stage_name in stage_names]
        _plot_bar_group(axis, labels, values, f"{control_name} ({unit})", unit, "{:.4g}")
    for axis in axes_flat[len(control_names):]:
        axis.axis("off")
    fig.suptitle(f"case_id={row['case_id']}\n{_format_context_text(row, context_names)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / CONTROL_FIG_NAME, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_metric_figure(
    row: pd.Series,
    metrics: List[str],
    stage_names: List[str],
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    rows, cols = _subplot_grid(len(metrics))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for axis, metric_name in zip(axes_flat, metrics):
        labels = list(stage_names)
        values = [float(row[f"{stage_name}_{metric_name}"]) for stage_name in stage_names]
        true_column = f"True_{metric_name}"
        if true_column in row.index and not pd.isna(row[true_column]):
            labels.append("True")
            values.append(float(row[true_column]))
        unit = UNIT_MAP.get(metric_name, "-")
        value_fmt = "{:.4g}" if unit != "level" else "{:.0f}"
        _plot_bar_group(axis, labels, values, f"{metric_name} ({unit})", unit, value_fmt)
    for axis in axes_flat[len(metrics):]:
        axis.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_case(
    row: pd.Series,
    control_names: List[str],
    context_names: List[str],
    stage_names: List[str],
    csv_dir: Path,
    dpi: int,
) -> None:
    case_dir = csv_dir / _safe_case_folder_name(str(row["case_id"]))
    case_dir.mkdir(parents=True, exist_ok=True)
    case_title = f"case_id={row['case_id']}\n{_format_context_text(row, context_names)}"

    _save_control_figure(
        row=row,
        control_names=control_names,
        context_names=context_names,
        stage_names=stage_names,
        output_dir=case_dir,
        dpi=dpi,
    )
    _save_metric_figure(
        row=row,
        metrics=["HIC", "Dmax", "Nij"],
        stage_names=stage_names,
        output_path=case_dir / SCALAR_FIG_NAME,
        title=case_title,
        dpi=dpi,
    )
    _save_metric_figure(
        row=row,
        metrics=["AIS_head", "AIS_chest", "AIS_neck", "MAIS"],
        stage_names=stage_names,
        output_path=case_dir / AIS_FIG_NAME,
        title=case_title,
        dpi=dpi,
    )
    _save_metric_figure(
        row=row,
        metrics=["Phead", "Pchest", "Pneck", "JointRisk"],
        stage_names=stage_names,
        output_path=case_dir / RISK_FIG_NAME,
        title=case_title,
        dpi=dpi,
    )


def main() -> None:
    args = parse_args()
    eval_csv_path = Path(args.eval_csv).resolve()
    if not eval_csv_path.is_file():
        raise FileNotFoundError(f"评估结果 CSV 不存在: {eval_csv_path}")

    df = pd.read_csv(eval_csv_path)
    _require_columns(df, ["case_id"], "评估结果绘图")
    if df["case_id"].astype(str).duplicated().any():
        duplicated = sorted(df.loc[df["case_id"].astype(str).duplicated(), "case_id"].astype(str).unique().tolist())
        raise ValueError(f"评估结果 CSV 中存在重复 case_id，无法唯一定位绘图对象: {duplicated}")
    # 这里不再兼容旧版缺列结果，也不在绘图脚本里临时补算派生指标。
    # 绘图工具只接受“当前 run_eval 规范直接生成的完整结果 CSV”；
    # 若 True_ 列不完整，应回到评估脚本修复，而不是在绘图阶段静默兜底。
    if any(column.startswith("True_") for column in df.columns):
        _require_columns(df, TRUE_METRIC_COLUMNS, "测试集真值绘图")

    stage_names, control_names = _detect_stage_names_and_controls(df)
    context_names = _detect_context_names(df, control_names)
    _require_columns(df, context_names, "评估结果绘图")

    selected_case_ids = [str(case_id) for case_id in args.case_ids]
    df_case_index = df.assign(_case_id_key=df["case_id"].astype(str)).set_index("_case_id_key", drop=False)
    found_any = False
    for case_id in selected_case_ids:
        if case_id not in df_case_index.index:
            print(f"[WARNING] case_id={case_id} 不存在，已跳过")
            continue
        found_any = True
        _plot_case(
            row=df_case_index.loc[case_id],
            control_names=control_names,
            context_names=context_names,
            stage_names=stage_names,
            csv_dir=eval_csv_path.parent,
            dpi=int(args.dpi),
        )
        print(f"[INFO] case_id={case_id} 绘图完成")

    if not found_any:
        raise ValueError("指定的 case_id 全部不存在，未生成任何图像")


if __name__ == "__main__":
    main()
