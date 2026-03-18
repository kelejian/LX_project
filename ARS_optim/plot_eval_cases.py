import argparse
import math
import textwrap
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.metrics.injury_risk import (
    AIS_cal_chest,
    AIS_cal_head,
    AIS_cal_neck,
    Injury_prob_cal_chest,
    Injury_prob_cal_head,
    Injury_prob_cal_neck,
)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-case comparison charts from ARS_optim evaluation CSV")
    parser.add_argument("--eval_csv", required=True, type=str, help="Path to evaluation_results.csv or custom eval csv")
    parser.add_argument("--case_ids", required=True, nargs="+", help="One or more case_id values to plot")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    return parser.parse_args()


def _safe_case_folder_name(case_id: str) -> str:
    text = str(case_id)
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
    return f"case_{safe}"


def _warn(message: str) -> None:
    print(f"[WARNING] {message}")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], scope: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{scope} 缺少必需列: {missing}")


def _detect_control_names(df: pd.DataFrame) -> List[str]:
    base_names = {column[5:] for column in df.columns if column.startswith("Base_")}
    opt_names = {column[4:] for column in df.columns if column.startswith("Opt_")}
    control_names = sorted((base_names & opt_names) - METRIC_SUFFIXES, key=lambda name: FEATURE_ORDER.index(name) if name in FEATURE_ORDER else name)
    if not control_names:
        raise ValueError("未能从结果 CSV 中识别到任何可调 control 参数列")
    return control_names


def _detect_context_names(df: pd.DataFrame, control_names: List[str]) -> List[str]:
    return [name for name in FEATURE_ORDER if name in df.columns and name not in control_names]


def _ensure_true_columns(df: pd.DataFrame) -> pd.DataFrame:
    """为旧版测试集评估 CSV 补齐 True_ 风险/AIS 列，便于统一绘图。

    旧版 run_eval 结果表里只有 True_HIC / True_Dmax / True_Nij 与 True_AIS_*，
    没有 True_Phead / True_Pchest / True_Pneck / True_JointRisk。
    这里仅在检测到真值损伤标量但缺少派生列时，按与 run_eval 相同的口径即时补算，
    使绘图工具既能处理新结果，也能处理用户已经保存的历史评估目录。
    """
    has_any_true = any(column.startswith("True_") for column in df.columns)
    if not has_any_true:
        return df

    required_truth_scalars = ["True_HIC", "True_Dmax", "True_Nij"]
    _require_columns(df, required_truth_scalars, "测试集真值绘图")
    if "OT" not in df.columns:
        raise ValueError("测试集真值绘图缺少 OT 列，无法计算胸部真值风险/AIS")

    result_df = df.copy()
    ot_array = pd.to_numeric(result_df["OT"], errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(ot_array).any():
        raise ValueError("OT 列存在缺失或非法值，无法计算真值胸部风险/AIS")

    if "True_Phead" not in result_df.columns:
        result_df["True_Phead"] = np.asarray(
            Injury_prob_cal_head(result_df["True_HIC"].to_numpy(dtype=np.float32)),
            dtype=np.float32,
        )
    if "True_Pchest" not in result_df.columns:
        result_df["True_Pchest"] = np.asarray(
            Injury_prob_cal_chest(result_df["True_Dmax"].to_numpy(dtype=np.float32), OT=ot_array),
            dtype=np.float32,
        )
    if "True_Pneck" not in result_df.columns:
        result_df["True_Pneck"] = np.asarray(
            Injury_prob_cal_neck(result_df["True_Nij"].to_numpy(dtype=np.float32)),
            dtype=np.float32,
        )
    if "True_JointRisk" not in result_df.columns:
        result_df["True_JointRisk"] = 1.0 - (
            (1.0 - result_df["True_Phead"])
            * (1.0 - result_df["True_Pchest"])
            * (1.0 - result_df["True_Pneck"])
        )
    if "True_AIS_head" not in result_df.columns:
        result_df["True_AIS_head"] = AIS_cal_head(result_df["True_HIC"].to_numpy(dtype=np.float32))
    if "True_AIS_chest" not in result_df.columns:
        result_df["True_AIS_chest"] = AIS_cal_chest(result_df["True_Dmax"].to_numpy(dtype=np.float32), ot_array)
    if "True_AIS_neck" not in result_df.columns:
        result_df["True_AIS_neck"] = AIS_cal_neck(result_df["True_Nij"].to_numpy(dtype=np.float32))
    if "True_MAIS" not in result_df.columns:
        result_df["True_MAIS"] = np.maximum.reduce(
            [
                result_df["True_AIS_head"].to_numpy(dtype=np.float32),
                result_df["True_AIS_chest"].to_numpy(dtype=np.float32),
                result_df["True_AIS_neck"].to_numpy(dtype=np.float32),
            ]
        )
    return result_df


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


def _annotate_bars(ax, bars, value_fmt: str) -> None:
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


def _plot_bar_group(ax, labels: List[str], values: List[float], title: str, unit: str, value_fmt: str) -> None:
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"][: len(values)], width=0.6)
    _annotate_bars(ax, bars, value_fmt=value_fmt)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _save_control_figure(row: pd.Series, control_names: List[str], context_names: List[str], output_dir: Path, dpi: int) -> None:
    _require_columns(row.to_frame().T, [f"Base_{name}" for name in control_names] + [f"Opt_{name}" for name in control_names], "control 绘图")
    rows, cols = _subplot_grid(len(control_names))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for axis, control_name in zip(axes_flat, control_names):
        unit = UNIT_MAP.get(control_name, "-")
        values = [float(row[f"Base_{control_name}"]), float(row[f"Opt_{control_name}"])]
        _plot_bar_group(axis, ["Base", "Opt"], values, f"{control_name} ({unit})", unit, "{:.4g}")
    for axis in axes_flat[len(control_names):]:
        axis.axis("off")
    fig.suptitle(f"case_id={row['case_id']}\n{_format_context_text(row, context_names)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / CONTROL_FIG_NAME, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_metric_figure(
    row: pd.Series,
    metrics: List[str],
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    rows, cols = _subplot_grid(len(metrics))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for axis, metric_name in zip(axes_flat, metrics):
        labels = ["Base", "Opt"]
        values = [float(row[f"Base_{metric_name}"]), float(row[f"Opt_{metric_name}"])]
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


def _plot_case(row: pd.Series, control_names: List[str], context_names: List[str], csv_dir: Path, dpi: int) -> None:
    case_dir = csv_dir / _safe_case_folder_name(str(row["case_id"]))
    case_dir.mkdir(parents=True, exist_ok=True)
    case_title = f"case_id={row['case_id']}\n{_format_context_text(row, context_names)}"

    _save_control_figure(row=row, control_names=control_names, context_names=context_names, output_dir=case_dir, dpi=dpi)
    _save_metric_figure(
        row=row,
        metrics=["HIC", "Dmax", "Nij"],
        output_path=case_dir / SCALAR_FIG_NAME,
        title=case_title,
        dpi=dpi,
    )
    _save_metric_figure(
        row=row,
        metrics=["AIS_head", "AIS_chest", "AIS_neck", "MAIS"],
        output_path=case_dir / AIS_FIG_NAME,
        title=case_title,
        dpi=dpi,
    )
    _save_metric_figure(
        row=row,
        metrics=["Phead", "Pchest", "Pneck", "JointRisk"],
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
    df = _ensure_true_columns(df)

    control_names = _detect_control_names(df)
    context_names = _detect_context_names(df, control_names)
    required_metric_columns = [
        "Base_HIC", "Base_Dmax", "Base_Nij",
        "Opt_HIC", "Opt_Dmax", "Opt_Nij",
        "Base_AIS_head", "Base_AIS_chest", "Base_AIS_neck", "Base_MAIS",
        "Opt_AIS_head", "Opt_AIS_chest", "Opt_AIS_neck", "Opt_MAIS",
        "Base_Phead", "Base_Pchest", "Base_Pneck", "Base_JointRisk",
        "Opt_Phead", "Opt_Pchest", "Opt_Pneck", "Opt_JointRisk",
    ]
    _require_columns(df, required_metric_columns, "评估结果绘图")

    selected_case_ids = [str(case_id) for case_id in args.case_ids]
    df_case_index = df.assign(_case_id_key=df["case_id"].astype(str)).set_index("_case_id_key", drop=False)
    found_any = False
    for case_id in selected_case_ids:
        if case_id not in df_case_index.index:
            _warn(f"case_id={case_id} 不存在，已跳过")
            continue
        found_any = True
        _plot_case(
            row=df_case_index.loc[case_id],
            control_names=control_names,
            context_names=context_names,
            csv_dir=eval_csv_path.parent,
            dpi=int(args.dpi),
        )
        print(f"[INFO] case_id={case_id} 绘图完成")

    if not found_any:
        raise ValueError("指定的 case_id 全部不存在，未生成任何图像")


if __name__ == "__main__":
    main()