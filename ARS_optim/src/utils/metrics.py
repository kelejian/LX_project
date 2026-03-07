# src/utils/metrics.py

import numpy as np
from typing import Dict, Any, Optional
from ARS_optim.src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MetricsTracker:
    """
        轻量级评估统计器 (Metrics Tracker)

        角色定位：
        1. 面向“单次评估运行过程”的快速监控，记录每个 case 的损失变化、耗时与参数位移。
        2. 输出简洁聚合统计（如平均改进率、成功率），便于终端实时观察优化是否有效。

        职责边界：
        - 本类不负责生成最终对外分析报表；
        - 详细宏观指标（如分部位风险均值、联合风险下降、真值对比等）由 run_evaluation.py
            的结果汇总逻辑统一生成并落盘。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计数据"""
        self.history = {
            "case_ids": [],
            "initial_loss": [],
            "final_loss": [],
            "improvement_rate": [], # (Init - Final) / Init
            "time_cost": [],
            "param_shift_l2": [],   # 参数调整的欧氏距离
            "steps_taken": []
        }
    
    def update(
        self,
        result: Dict[str, Any],
        case_id: int,
        initial_action: Optional[np.ndarray] = None,
        optimized_action: Optional[np.ndarray] = None
    ):
        """
        更新单次优化结果
        
        Args:
            result: optimizer.optimize() 返回的字典
            case_id: 当前案例 ID
        """
        if 'initial' not in result:
            raise KeyError("result 缺少 'initial' 字段，无法更新指标。")

        init_loss = float(result['initial'].get('loss_mean', 0.0))
        final_loss_batch = result.get('final_loss_batch')
        if final_loss_batch is None:
            final_loss = init_loss
        else:
            final_loss = float(final_loss_batch.mean().item())
        
        # 计算优化率 (防止除零)
        if abs(init_loss) > 1e-6:
            imp_rate = (init_loss - final_loss) / init_loss
        else:
            imp_rate = 0.0
            
        # 参数偏移量 (L2 Distance) 由调用方提供，避免强依赖 result 结构
        if initial_action is not None and optimized_action is not None:
            shift = float(np.linalg.norm(np.asarray(optimized_action) - np.asarray(initial_action)))
        else:
            shift = 0.0
        
        # 记录数据
        self.history["case_ids"].append(case_id)
        self.history["initial_loss"].append(init_loss)
        self.history["final_loss"].append(final_loss)
        self.history["improvement_rate"].append(imp_rate)
        self.history["time_cost"].append(float(result.get('time_cost', 0.0)))
        self.history["param_shift_l2"].append(shift)
        self.history["steps_taken"].append(len(result.get('trajectory', [])))

    def compute_summary(self) -> Dict[str, float]:
        """计算当前所有记录的聚合统计信息"""
        count = len(self.history["case_ids"])
        if count == 0:
            return {}
            
        summary = {
            "total_samples": count,
            "avg_initial_loss": np.mean(self.history["initial_loss"]),
            "avg_final_loss": np.mean(self.history["final_loss"]),
            "avg_improvement_rate": np.mean(self.history["improvement_rate"]),
            "avg_time_ms": np.mean(self.history["time_cost"]) * 1000, # ms
            "avg_param_shift": np.mean(self.history["param_shift_l2"]),
            "avg_steps": np.mean(self.history["steps_taken"]),
            
            # 成功率统计：优化率 > 0 即视为有效优化
            "success_rate": np.mean(np.array(self.history["improvement_rate"]) > 1e-4)
        }
        return summary

    def log_summary(self):
        """打印统计摘要"""
        s = self.compute_summary()
        if not s:
            logger.warning("No metrics to log.")
            return

        logger.info("\n" + "="*40)
        logger.info(" [Evaluation Summary]")
        logger.info("="*40)
        logger.info(f" Total Samples      : {s['total_samples']}")
        logger.info(f" Avg Time Cost      : {s['avg_time_ms']:.2f} ms")
        logger.info(f" Avg Steps          : {s['avg_steps']:.1f}")
        logger.info("-" * 40)
        logger.info(f" Avg Initial Loss   : {s['avg_initial_loss']:.4f}")
        logger.info(f" Avg Final Loss     : {s['avg_final_loss']:.4f}")
        logger.info(f" Avg Improvement    : {s['avg_improvement_rate']*100:.2f}%")
        logger.info(f" Success Rate       : {s['success_rate']*100:.1f}%")
        logger.info(f" Avg Param Shift    : {s['avg_param_shift']:.4f} (L2)")
        logger.info("="*40 + "\n")

    @staticmethod
    def calculate_damage_reduction(initial: float, final: float) -> float:
        """静态辅助函数：计算单项损伤降低率"""
        if initial <= 1e-6: return 0.0
        return (initial - final) / initial