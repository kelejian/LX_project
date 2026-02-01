import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import warnings
warnings.filterwarnings('ignore')
import joblib
import optuna
from optuna.storages import RDBStorage
import argparse
import json
'''
将保存在本地 .pkl 文件中的 Optuna study 恢复并加载到 SQLite 数据库中。
这对于从中断的、未连接数据库的运行中恢复，或迁移历史运行记录非常有用。
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从.pkl文件恢复Optuna study到SQLite数据库。")
    parser.add_argument("--pkl_file", '-f', type=str, required=True, help="需要恢复的 .pkl 文件路径。")
    parser.add_argument("--db_path", '-d', type=str, default="sqlite:///./runs/optuna_study.db", help="SQLite数据库文件路径。")
    parser.add_argument("--study_name", '-n', type=str, required=True, help="在数据库中为研究指定的新名称。")
    args = parser.parse_args()

    # 1. 加载要恢复的 .pkl 文件
    try:
        study_to_restore = joblib.load(args.pkl_file)
        # 检查加载的对象是否为 Optuna 的 Study 对象
        if isinstance(study_to_restore, optuna.study.Study):
            trials = study_to_restore.trials
            print(f"从 '{args.pkl_file}' 中成功加载 {len(trials)} 个试验。")
        else:
            raise TypeError("加载的 .pkl 文件不包含一个有效的 Optuna Study 对象。")

    except FileNotFoundError:
        print(f"错误: 文件 '{args.pkl_file}' 未找到。")
        exit()
    except Exception as e:
        print(f"加载 .pkl 文件时发生错误: {e}")
        exit()

    # 2. 定义多目标优化的方向。这个列表必须与训练脚本中使用的 directions 完全一致！
    # 目标是最大化四个准确率指标。
    study_directions = ["maximize", "maximize", "maximize", "maximize"]

    # 3. 连接到数据库并创建或加载研究
    storage = RDBStorage(args.db_path)
    new_study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=study_directions,  # 使用与训练脚本匹配的多目标方向
        load_if_exists=True # 如果同名研究已存在，则加载它，而不是报错
    )
    print(f"成功创建或加载研究 '{args.study_name}' 于 '{args.db_path}'。")

    # 4. 将从 .pkl 文件中加载的 trial 数据逐个添加到新的 study 中
    added_trials_count = 0
    for trial in trials:
        try:
            # add_trial 会检查试验是否已存在，避免重复添加
            new_study.add_trial(trial)
            added_trials_count += 1
        except Exception as e:
            print(f"警告: 无法添加试验 {trial.number}。原因: {e}")

    print(f"成功向研究 '{args.study_name}' 中添加了 {added_trials_count} 个新试验。")

    # 5. 打印新研究的 Pareto 前沿结果，并更新标签以匹配新的优化目标
    print("\n" + "="*60)
    print(f"    研究 '{args.study_name}' 的帕累托前沿结果 (准确率优化)")
    print("="*60)
    if new_study.best_trials:
        for trial in new_study.best_trials:
            print(f"Trial Number: {trial.number}")
            # trial.values 列表中的值与 study_directions 的顺序一一对应
            print(f"  - Values (MAIS Acc, Head Acc, Chest Acc, Neck Acc):")
            print(f"    - MAIS Acc:   {trial.values[0]:.2f}%")
            print(f"    - Head Acc:   {trial.values[1]:.2f}%")
            print(f"    - Chest Acc:  {trial.values[2]:.2f}%")
            print(f"    - Neck Acc:   {trial.values[3]:.2f}%")
            print(f"  - Params: {json.dumps(trial.params, indent=4)}")
            print("-" * 30)
    else:
        print("未找到最佳试验。")