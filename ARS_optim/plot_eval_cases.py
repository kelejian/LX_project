import argparse
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
AIS_METRICS = {"AIS_head", "AIS_chest", "AIS_neck", "MAIS"}
TRUE_METRIC_COLUMNS = [
    "True_HIC",
    "True_Dmax",
    "True_Nij",
    "True_Phead",
    "True_Pchest",
    "True_Pneck",
    "True_JointRisk",
    "True_AIS_head",
    "True_AIS_chest",
    "True_AIS_neck",
    "True_MAIS",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-case comparison charts from ARS_optim evaluation CSV")
    parser.add_argument("--eval_csv", required=True, type=str, help="Absolute path to evaluation_results.csv")
    parser.add_argument("--case_ids", nargs="*", default=None, help="Explicit case_id list to plot")
    parser.add_argument("--topn_joint_risk", type=int, default=0, help="Plot top-N cases with the largest JointRisk reduction")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, columns: Iterable[str], scope: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{scope} 缺少必需列: {missing}")


def _detect_stage_names_and_controls(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    base_control_names = {column[5:] for column in df.columns if column.startswith("Base_")} - METRIC_SUFFIXES
    if not base_control_names:
        raise ValueError("结果 CSV 中未识别到任何 Base_ 可调 control 列")

    stages = ["Base"]
    _require_columns(df, [f"Base_{metric_name}" for metric_name in RESULT_METRIC_NAMES], "Base 指标绘图")
    for stage_name in OPT_STAGE_CANDIDATES:
        prefix = f"{stage_name}_"
        stage_control_names = {column[len(prefix):] for column in df.columns if column.startswith(prefix)} - METRIC_SUFFIXES
        has_stage_metrics = any(f"{stage_name}_{metric_name}" in df.columns for metric_name in RESULT_METRIC_NAMES)
        if not stage_control_names and not has_stage_metrics:
            continue
        if stage_control_names != base_control_names:
            raise ValueError(
                f"Base_ 与 {stage_name}_ 的 control 列不一致: "
                f"Base_only={sorted(base_control_names - stage_control_names)}, "
                f"{stage_name}_only={sorted(stage_control_names - base_control_names)}"
            )
        _require_columns(df, [f"{stage_name}_{name}" for name in sorted(base_control_names)], f"{stage_name} control 绘图")
        _require_columns(df, [f"{stage_name}_{metric_name}" for metric_name in RESULT_METRIC_NAMES], f"{stage_name} 指标绘图")
        stages.append(stage_name)

    if len(stages) == 1:
        raise ValueError("结果 CSV 中不存在任何 Opt1_/Opt2_ 阶段列")
    control_names = sorted(base_control_names, key=lambda name: FEATURE_ORDER.index(name) if name in FEATURE_ORDER else name)
    return stages, control_names


def _detect_context_names(df: pd.DataFrame, control_names: List[str]) -> List[str]:
    control_set = set(control_names)
    context_names = []
    for name in FEATURE_ORDER:
        has_context_column = name in df.columns
        has_control_pair = name in control_set
        if has_context_column and has_control_pair:
            raise ValueError(f"参数 {name} 同时作为 context 原列和 control 成对列存在，结果 CSV 语义冲突")
        if not has_context_column and not has_control_pair:
            raise ValueError(f"参数 {name} 既不在 context 原列中，也不在 Base_/Opt*_control 列中")
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


def _plot_bar_group(ax, labels: List[str], values: List[float], title: str, unit: str, value_fmt: str, force_level_axis: bool = False) -> None:
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(values)]
    bars = ax.bar(labels, values, color=colors, width=0.6)
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
    if force_level_axis:
        ax.set_ylim(0, 5)
        ax.set_yticks(range(6))


def _save_control_figure(
    row: pd.Series,
    case_prefix: str,
    control_names: List[str],
    context_names: List[str],
    stage_names: List[str],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(control_names), figsize=(5.2 * len(control_names), 4.6), squeeze=False)
    for axis, control_name in zip(axes[0], control_names):
        unit = UNIT_MAP.get(control_name, "-")
        labels = list(stage_names)
        values = [float(row[f"{stage_name}_{control_name}"]) for stage_name in stage_names]
        _plot_bar_group(axis, labels, values, f"{control_name} ({unit})", unit, "{:.4g}")
    fig.suptitle(f"case_id={row['case_id']}\n{_format_context_text(row, context_names)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_dir / f"{case_prefix}_{CONTROL_FIG_NAME}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_metric_figure(
    row: pd.Series,
    metrics: List[str],
    stage_names: List[str],
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.6), squeeze=False)
    for axis, metric_name in zip(axes[0], metrics):
        labels = list(stage_names)
        values = [float(row[f"{stage_name}_{metric_name}"]) for stage_name in stage_names]
        true_column = f"True_{metric_name}"
        if true_column in row.index and not pd.isna(row[true_column]):
            labels.append("True")
            values.append(float(row[true_column]))
        unit = UNIT_MAP.get(metric_name, "-")
        value_fmt = "{:.0f}" if metric_name in AIS_METRICS else "{:.4g}"
        _plot_bar_group(
            axis,
            labels,
            values,
            f"{metric_name} ({unit})",
            unit,
            value_fmt,
            force_level_axis=metric_name in AIS_METRICS,
        )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_case(
    row: pd.Series,
    control_names: List[str],
    context_names: List[str],
    stage_names: List[str],
    output_dir: Path,
    dpi: int,
) -> None:
    safe_case_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(row["case_id"]))
    case_prefix = f"case{safe_case_id}"
    case_title = f"case_id={row['case_id']}\n{_format_context_text(row, context_names)}"
    _save_control_figure(row, case_prefix, control_names, context_names, stage_names, output_dir, dpi)
    _save_metric_figure(row, ["HIC", "Dmax", "Nij"], stage_names, output_dir / f"{case_prefix}_{SCALAR_FIG_NAME}", case_title, dpi)
    _save_metric_figure(row, ["AIS_head", "AIS_chest", "AIS_neck", "MAIS"], stage_names, output_dir / f"{case_prefix}_{AIS_FIG_NAME}", case_title, dpi)
    _save_metric_figure(row, ["Phead", "Pchest", "Pneck", "JointRisk"], stage_names, output_dir / f"{case_prefix}_{RISK_FIG_NAME}", case_title, dpi)


def _select_topn_case_ids(df: pd.DataFrame, stage_names: List[str], topn_joint_risk: int) -> List[str]:
    if topn_joint_risk <= 0:
        return []
    preferred_stage = "Opt2" if "Opt2" in stage_names else "Opt1"
    reduction_col = f"Reduction_{preferred_stage}_JointRisk"
    _require_columns(df, [reduction_col], "topN JointRisk 选图")
    ranked = (
        df.loc[df[reduction_col].notna(), ["case_id", reduction_col]]
        .sort_values(by=reduction_col, ascending=False)
        .head(topn_joint_risk)
    )
    if ranked.empty:
        raise ValueError(f"结果 CSV 中没有可用于 topN 选图的 {reduction_col} 有效值")
    return ranked["case_id"].astype(str).tolist()


def _plot_selected_cases(
    df_case_index: pd.DataFrame,
    selected_case_ids: List[str],
    control_names: List[str],
    context_names: List[str],
    stage_names: List[str],
    output_dir: Path,
    dpi: int,
) -> None:
    found_any = False
    for case_id in selected_case_ids:
        if case_id not in df_case_index.index:
            print(f"[WARNING] case_id={case_id} 不存在，已跳过")
            continue
        if not found_any:
            # 只有真正命中至少一个可绘制 case 时才创建目录，避免“全是无效 case_id”时留下空文件夹。
            output_dir.mkdir(parents=True, exist_ok=False)
        found_any = True
        _plot_case(
            row=df_case_index.loc[case_id],
            control_names=control_names,
            context_names=context_names,
            stage_names=stage_names,
            output_dir=output_dir,
            dpi=dpi,
        )
        print(f"[INFO] case_id={case_id} 绘图完成 -> {output_dir}")
    if not found_any:
        raise ValueError(f"{output_dir.name} 未生成任何图片：指定 case_id 全部不存在")


def main() -> None:
    args = parse_args()
    explicit_case_ids = [str(case_id) for case_id in (args.case_ids or [])]
    if not explicit_case_ids and int(args.topn_joint_risk) <= 0:
        raise ValueError("至少需要指定 --case_ids 或 --topn_joint_risk 之一")

    eval_csv_path = Path(args.eval_csv).resolve()
    if not eval_csv_path.is_file():
        raise FileNotFoundError(f"评估结果 CSV 不存在: {eval_csv_path}")

    df = pd.read_csv(eval_csv_path)
    _require_columns(df, ["case_id"], "评估结果绘图")
    if df["case_id"].astype(str).duplicated().any():
        duplicated = sorted(df.loc[df["case_id"].astype(str).duplicated(), "case_id"].astype(str).unique().tolist())
        raise ValueError(f"结果 CSV 中存在重复 case_id，无法唯一定位绘图对象: {duplicated}")
    if any(column.startswith("True_") for column in df.columns):
        _require_columns(df, TRUE_METRIC_COLUMNS, "测试集真值绘图")

    stage_names, control_names = _detect_stage_names_and_controls(df)
    context_names = _detect_context_names(df, control_names)
    _require_columns(df, context_names, "评估结果绘图")

    df_case_index = df.assign(_case_id_key=df["case_id"].astype(str)).set_index("_case_id_key", drop=False)
    parent_dir = eval_csv_path.parent

    if explicit_case_ids:
        _plot_selected_cases(
            df_case_index=df_case_index,
            selected_case_ids=explicit_case_ids,
            control_names=control_names,
            context_names=context_names,
            stage_names=stage_names,
            output_dir=parent_dir / "plots_case_ids",
            dpi=int(args.dpi),
        )

    if int(args.topn_joint_risk) > 0:
        top_case_ids = _select_topn_case_ids(df, stage_names, int(args.topn_joint_risk))
        _plot_selected_cases(
            df_case_index=df_case_index,
            selected_case_ids=top_case_ids,
            control_names=control_names,
            context_names=context_names,
            stage_names=stage_names,
            output_dir=parent_dir / f"plots_top{int(args.topn_joint_risk)}_joint_risk",
            dpi=int(args.dpi),
        )


if __name__ == "__main__":
    main()
