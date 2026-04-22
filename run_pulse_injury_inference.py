"""\
run_pulse_injury_inference.py

批处理推理工具。输入一个包含碰撞参数的CSV文件，先使用PulsePredict模型
预测碰撞波形，再将波形传给InjuryPredict模型预测损伤指标。
输出目录结构参见脚本帮助，默认位于 DATA_DIR/inference_outputs。

输入CSV必须满足以下要求：
  * 扩展名为 .csv
  * 第一列为 case_id（整数），后续按 FEATURE_ORDER 定义的顺序包含特征值
      impact_velocity, impact_angle, overlap, LL1, ..., RA, is_driver_side, OT
  * 特征列应为数字类型，is_driver_side 0/1，OT 1/2/3。
  * 若缺省 case_id 列且列数正好等于 len(FEATURE_ORDER)+1，则会按顺序重命名。
  * 不允许重复 case_id。

脚本依赖项目 common 和各子模块中的模型定义，需在 LX_project 根目录执行。
"""
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.data_utils.processor import UnifiedDataProcessor
from common.metrics import AIS_cal_chest, AIS_cal_head, AIS_cal_neck
from common.settings import DATA_DIR, PULSE_PREDICT_DIR, INJURY_PREDICT_DIR, FEATURE_ORDER, NORMALIZATION_CONFIG_PATH, WAVEFORM_LENGTH
from InjuryPredict.utils.models import InjuryPredictModel
from PulsePredict.model.model import HybridPulseCNN


DEFAULT_OUTPUT_ROOT = DATA_DIR / "inference_outputs"
DEFAULT_PULSE_RUN_DIR = PULSE_PREDICT_DIR / "saved" / "models" / "HybridPulseCNN" / "0415_161324"
DEFAULT_INJURY_RUN_DIR = INJURY_PREDICT_DIR / "runs" / "InjuryPredictModel_03280055"
PULSE_FEATURE_NAMES = ["impact_velocity", "impact_angle", "overlap"]

INPUT_CSV_FILE = Path(r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\LX_project\ARS_optim\saved_eval\cases_for_sledtest_0331.csv")  # 可在此处修改输入CSV路径，或通过命令行参数覆盖

def parse_args() -> argparse.Namespace:
    # 解析命令行参数，返回一个命名空间对象
    parser = argparse.ArgumentParser(
        description="Batch inference for PulsePredict and InjuryPredict from a case parameter CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV_FILE,
        help="Input CSV path."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for inference outputs.",
    )
    parser.add_argument(
        "--pulse-checkpoint",
        type=Path,
        default=DEFAULT_PULSE_RUN_DIR / "model_best.pth",
        help="PulsePredict checkpoint path.",
    )
    parser.add_argument(
        "--pulse-config",
        type=Path,
        default=DEFAULT_PULSE_RUN_DIR / "config.json",
        help="PulsePredict config JSON path.",
    )
    parser.add_argument(
        "--injury-checkpoint",
        type=Path,
        default=DEFAULT_INJURY_RUN_DIR / "best_val_loss.pth",
        help="InjuryPredict checkpoint path.",
    )
    parser.add_argument(
        "--injury-record",
        type=Path,
        default=DEFAULT_INJURY_RUN_DIR / "TrainingRecord.json",
        help="InjuryPredict TrainingRecord.json path.",
    )
    parser.add_argument(
        "--normalization-config",
        type=Path,
        default=NORMALIZATION_CONFIG_PATH,
        help="Unified normalization config JSON path.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Inference device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for both stages.",
    )
    parser.add_argument(
        "--skip-waveform-plots",
        action="store_true",
        help="Do not save per-case waveform plots.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    # 根据用户指定的名称决定运行设备，支持 'auto'、'cpu'、'cuda'
    # 在auto模式下会优先使用可用的CUDA，否则回退到CPU
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_name(name: str) -> str:
    # 清理输入名称，替换非法字符并去除首尾下划线或点，返回安全的文件/目录名
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized or "inference"


def build_output_dir(input_csv: Path, output_root: Path) -> Path:
    # 为本次推理构建唯一的输出目录，目录名根据输入CSV名称和时间戳生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{sanitize_name(input_csv.stem)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def ensure_file(path: Path, label: str) -> None:
    # 检查文件是否存在，若不存在则抛出异常
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_json(path: Path) -> Dict:
    # 读取并解析JSON文件，返回Python字典
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_input_dataframe(input_csv: Path) -> pd.DataFrame:
    # 从CSV加载输入参数表格，并进行基本校验和类型转换
    if input_csv.suffix.lower() != ".csv":
        raise ValueError("Only CSV input is supported.")

    df = pd.read_csv(input_csv)
    expected_columns = ["case_id", *FEATURE_ORDER]

    if "case_id" not in df.columns:
        if df.shape[1] < len(expected_columns):
            raise ValueError(
                "Input CSV must include a case_id column or provide columns in the expected positional order."
            )
        renamed = df.iloc[:, : len(expected_columns)].copy()
        renamed.columns = expected_columns
        df = renamed

    missing = [column for column in FEATURE_ORDER if column not in df.columns]
    if missing:
        if df.shape[1] >= len(expected_columns):
            renamed = df.iloc[:, : len(expected_columns)].copy()
            renamed.columns = expected_columns
            df = renamed
            missing = [column for column in FEATURE_ORDER if column not in df.columns]
        if missing:
            raise ValueError(f"Input CSV is missing required feature columns: {missing}")

    df = df[["case_id", *FEATURE_ORDER]].copy()
    df["case_id"] = pd.to_numeric(df["case_id"], errors="raise").astype(np.int64)
    if df["case_id"].duplicated().any():
        duplicates = df.loc[df["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"Duplicate case_id values found: {duplicates[:10]}")

    for column in FEATURE_ORDER:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["is_driver_side"] = df["is_driver_side"].astype(np.int64)
    df["OT"] = df["OT"].astype(np.int64)
    return df


def load_pulse_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> HybridPulseCNN:
    # 根据配置和checkpoint加载PulsePredict模型并切换到评估模式
    config = load_json(config_path)
    model = HybridPulseCNN(**config["arch"]["args"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_injury_model(record_path: Path, checkpoint_path: Path, device: torch.device) -> InjuryPredictModel:
    # 从训练记录和checkpoint加载损伤预测模型
    record = load_json(record_path)
    model = InjuryPredictModel(**record["hyperparameters"]["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def batched_indices(total_size: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    # 生成分批次的索引范围，用于按batch_size遍历数据
    for start in range(0, total_size, batch_size):
        yield start, min(start + batch_size, total_size)


def extract_pulse_output(model_output: torch.Tensor, model: HybridPulseCNN) -> torch.Tensor:
    # 有些模型返回多个张量，或者包装在tuple/list中；提取第一个作为脉冲数据
    if hasattr(model, "get_metrics_output"):
        return model.get_metrics_output(model_output)
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def predict_pulses(
    pulse_model: HybridPulseCNN,
    processor: UnifiedDataProcessor,
    inputs_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # 对输入参数执行脉冲预测，返回三通道波形数据 np.ndarray 数组
    pulse_raw = inputs_df[PULSE_FEATURE_NAMES].to_numpy(dtype=np.float32)
    pulse_norm = processor.process_by_name(pulse_raw, PULSE_FEATURE_NAMES, inverse=False).astype(np.float32)
    # 归一化波形直接走伤害推理链路；物理波形只在需要落盘或可视化时才使用。

    predicted_norm_batches: List[np.ndarray] = []
    predicted_physical_batches: List[np.ndarray] = []
    with torch.no_grad():
        for start, end in batched_indices(len(inputs_df), batch_size):
            batch = torch.from_numpy(pulse_norm[start:end]).to(device)
            model_output = pulse_model(batch)
            pulse_output = extract_pulse_output(model_output, pulse_model)
            pulse_output_np = pulse_output.detach().cpu().numpy()
            predicted_norm_batches.append(pulse_output_np.astype(np.float32, copy=False))
            pulse_physical = processor.process_waveform(pulse_output_np, inverse=True).astype(np.float32)
            predicted_physical_batches.append(pulse_physical)

    return (
        np.concatenate(predicted_norm_batches, axis=0),
        np.concatenate(predicted_physical_batches, axis=0),
    )


def predict_injuries(
    injury_model: InjuryPredictModel,
    processor: UnifiedDataProcessor,
    inputs_df: pd.DataFrame,
    predicted_waveforms_norm_xyz: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    # 给定预测波形，运用损伤预测模型得到HIC15/Dmax/Nij并返回数组
    scalar_raw = inputs_df[FEATURE_ORDER].to_numpy(dtype=np.float32)
    x_continuous, x_discrete = processor.process_all_features(scalar_raw, inverse=False)
    # PulsePredict 与 InjuryPredict 共享同一套 waveform 归一化配置，因此这里直接复用模型原始输出。
    x_wave_xy = predicted_waveforms_norm_xyz[:, :2, :].astype(np.float32, copy=False)

    prediction_batches: List[np.ndarray] = []
    with torch.no_grad():
        for start, end in batched_indices(len(inputs_df), batch_size):
            x_acc = torch.from_numpy(x_wave_xy[start:end]).to(device)
            x_att_continuous = torch.from_numpy(x_continuous[start:end].astype(np.float32)).to(device)
            x_att_discrete = torch.from_numpy(x_discrete[start:end].astype(np.int64)).to(device)
            batch_pred, _, _ = injury_model(x_acc, x_att_continuous, x_att_discrete)
            prediction_batches.append(batch_pred.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(prediction_batches, axis=0)

def build_waveform_wide_dataframe(case_ids: np.ndarray, waveforms_xyz: np.ndarray) -> pd.DataFrame:
    """将波形数据转为宽格式：每个case生成四列

    列名为 {case_id}_time_ms, {case_id}_acc_x, {case_id}_acc_y, {case_id}_acc_z。
    输出行数等于波形长度 WAVEFORM_LENGTH，便于每个case在水平上并排查看。
    """
    num_cases = len(case_ids)
    length = WAVEFORM_LENGTH
    time_ms = np.arange(1, length + 1, dtype=np.int32)

    dfs: List[pd.DataFrame] = []
    for idx, cid in enumerate(case_ids):
        prefix = f"{cid}_"
        df = pd.DataFrame(
            {
                prefix + "time_ms": time_ms,
                prefix + "acc_x": waveforms_xyz[idx, 0, :],
                prefix + "acc_y": waveforms_xyz[idx, 1, :],
                prefix + "acc_z": waveforms_xyz[idx, 2, :],
            }
        )
        dfs.append(df)
    # 按列拼接
    wide_df = pd.concat(dfs, axis=1)
    return wide_df


def plot_predicted_waveform(
    case_id: int,
    params: Dict[str, float],
    waveform_xyz: np.ndarray,
    output_path: Path,
) -> None:
    # 为单个case绘制三轴预测波形并保存为PNG图片
    time_ms = np.arange(1, waveform_xyz.shape[1] + 1)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    titles = [
        ("Predicted X-direction Acceleration", "Acceleration (m/s^2)"),
        ("Predicted Y-direction Acceleration", "Acceleration (m/s^2)"),
        ("Predicted Z-direction Rotational Acceleration", "Angular Acceleration (rad/s^2)"),
    ]
    for index, axis in enumerate(axes):
        axis.plot(time_ms, waveform_xyz[index], color="#1f77b4", linewidth=2)
        axis.set_ylabel(titles[index][1])
        axis.set_title(titles[index][0])
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (ms)")
    figure.suptitle(
        (
            f"Case ID: {case_id}\n"
            f"Velocity: {params['impact_velocity']:.2f} km/h, "
            f"Angle: {params['impact_angle']:.2f} deg, "
            f"Overlap: {params['overlap']:.2f}"
        ),
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_waveform_plots(inputs_df: pd.DataFrame, predicted_waveforms_xyz: np.ndarray, plots_dir: Path) -> None:
    # 为每个输入样本调用plot_predicted_waveform并保存图片
    plots_dir.mkdir(parents=True, exist_ok=True)
    for row, waveform in zip(inputs_df.to_dict("records"), predicted_waveforms_xyz):
        case_id = int(row["case_id"])
        plot_predicted_waveform(case_id, row, waveform, plots_dir / f"case_{case_id}.png")


def build_injury_results_dataframe(inputs_df: pd.DataFrame, injury_predictions: np.ndarray) -> pd.DataFrame:
    # 根据模型预测结果计算AIS等级和MAIS
    result_df = inputs_df.copy()
    result_df["HIC15_pred"] = injury_predictions[:, 0]
    result_df["Dmax_pred"] = injury_predictions[:, 1]
    result_df["Nij_pred"] = injury_predictions[:, 2]
    result_df["AIS_head_pred"] = AIS_cal_head(result_df["HIC15_pred"].to_numpy()).astype(np.int64)
    result_df["AIS_chest_pred"] = AIS_cal_chest(
        result_df["Dmax_pred"].to_numpy(),
        result_df["OT"].to_numpy(dtype=np.int64),
    ).astype(np.int64)
    result_df["AIS_neck_pred"] = AIS_cal_neck(result_df["Nij_pred"].to_numpy()).astype(np.int64)
    result_df["MAIS_pred"] = np.maximum.reduce(
        [
            result_df["AIS_head_pred"].to_numpy(),
            result_df["AIS_chest_pred"].to_numpy(),
            result_df["AIS_neck_pred"].to_numpy(),
        ]
    ).astype(np.int64)
    return result_df


def save_manifest(output_path: Path, manifest: Dict) -> None:
    # 将执行摘要写入JSON清单文件
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    # 主入口函数，按步骤执行输入校验、模型加载、推理和结果保存
    args = parse_args()

    ensure_file(args.input_csv, "Input CSV")
    ensure_file(args.pulse_checkpoint, "Pulse checkpoint")
    ensure_file(args.pulse_config, "Pulse config")
    ensure_file(args.injury_checkpoint, "Injury checkpoint")
    ensure_file(args.injury_record, "Injury record")
    ensure_file(args.normalization_config, "Normalization config")

    device = resolve_device(args.device)
    processor = UnifiedDataProcessor(config_path=args.normalization_config)
    if not processor.load_config():
        raise RuntimeError(f"Failed to load normalization config: {args.normalization_config}")

    inputs_df = load_input_dataframe(args.input_csv)
    output_dir = build_output_dir(args.input_csv, args.output_root)
    waveform_dir = output_dir / "waveforms"
    waveform_plot_dir = waveform_dir / "plots"
    injury_dir = output_dir / "injuries"
    waveform_dir.mkdir(parents=True, exist_ok=True)
    injury_dir.mkdir(parents=True, exist_ok=True)

    pulse_model = load_pulse_model(args.pulse_config, args.pulse_checkpoint, device)
    injury_model = load_injury_model(args.injury_record, args.injury_checkpoint, device)

    print(f"[1/4] Loaded {len(inputs_df)} cases from {args.input_csv}")
    print(f"[2/4] Running pulse inference on {device}")
    predicted_waveforms_norm_xyz, predicted_waveforms_xyz = predict_pulses(
        pulse_model=pulse_model,
        processor=processor,
        inputs_df=inputs_df,
        device=device,
        batch_size=args.batch_size,
    )

    # 根据需求生成宽格式CSV，每四列对应一个case
    waveform_csv_path = waveform_dir / f"pred_pulses_of_{args.input_csv.stem}.csv"
    waveform_df = build_waveform_wide_dataframe(inputs_df["case_id"].to_numpy(dtype=np.int64), predicted_waveforms_xyz)
    waveform_df.to_csv(waveform_csv_path, index=False)

    if not args.skip_waveform_plots:
        save_waveform_plots(inputs_df, predicted_waveforms_xyz, waveform_plot_dir)

    print(f"[3/4] Running injury inference on {device}")
    injury_predictions = predict_injuries(
        injury_model=injury_model,
        processor=processor,
        inputs_df=inputs_df,
        predicted_waveforms_norm_xyz=predicted_waveforms_norm_xyz,
        device=device,
        batch_size=args.batch_size,
    )
    injury_df = build_injury_results_dataframe(inputs_df, injury_predictions)
    injury_csv_path = injury_dir / f"pred_injuries_of_{args.input_csv.stem}.csv"
    injury_df.to_csv(injury_csv_path, index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(args.input_csv),
        "output_dir": str(output_dir),
        "device": str(device),
        "num_cases": int(len(inputs_df)),
        "normalization_config": str(args.normalization_config),
        "pulse": {
            "checkpoint": str(args.pulse_checkpoint),
            "config": str(args.pulse_config),
            "waveform_csv": str(waveform_csv_path),
            "plots_dir": str(waveform_plot_dir),
        },
        "injury": {
            "checkpoint": str(args.injury_checkpoint),
            "record": str(args.injury_record),
            "injury_csv": str(injury_csv_path),
        },
    }
    save_manifest(output_dir / "manifest.json", manifest)

    print(f"[4/4] Finished. Output directory: {output_dir}")
    print(f"         Waveforms CSV: {waveform_csv_path}")
    print(f"         Injuries CSV:  {injury_csv_path}")


if __name__ == "__main__":
    main()
