"""结果保存模块 - 参考mini-swe-agent的save_traj实现"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from swe_mas import OUTPUT_DIR
from swe_mas.utils.logger import get_logger

logger = get_logger(__name__)


def save_results(
    instance_id: str,
    problem_statement: str,
    final_state: dict[str, Any],
    model_name: str = "qwen-max",
    *,
    run_id: str = "default",
    output_root: Path | None = None,
) -> tuple[Path, Path]:
    """保存运行结果
    
    Args:
        instance_id: 实例ID
        problem_statement: 问题描述
        final_state: 最终状态
        model_name: 模型名称
        
    Returns:
        (trajectory_path, predictions_path) 轨迹文件路径和预测文件路径
    """
    # 创建实例目录（按 run_id 分组）
    out_root = output_root or OUTPUT_DIR
    run_dir = out_root / run_id
    instance_dir = run_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存完整轨迹（参考mini-swe-agent的trajectory格式）
    trace = final_state.get("recovery_trace", final_state.get("recovery_struct", {}).get("trace", {}))
    trajectory = {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "workflow_state": {
            "analysis": final_state.get("analysis", ""),
            "located_files": final_state.get("located_files", ""),
            "plan": final_state.get("plan", ""),
            "patch": final_state.get("patch", ""),
            "verification": final_state.get("verification", ""),
            "success": final_state.get("success", False),
            "error": final_state.get("error"),
            "fault_type": final_state.get("fault_type"),
            "used_recovery": bool(final_state.get("recovery_summary")),
            "recovery_summary": final_state.get("recovery_summary", ""),
            "recovery_struct": final_state.get("recovery_struct", {}),
            "recovery_trace": trace,
        },
        "all_messages": final_state.get("all_messages", []),
        "all_commands": final_state.get("all_commands", []),
    }
    
    trajectory_path = instance_dir / "trajectory.json"
    with open(trajectory_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    
    logger.info(f"轨迹已保存: {trajectory_path}")

    # 1.1 保存摘要/上下文对比视图
    try:
        summary_text = final_state.get("recovery_summary", "")
        if summary_text:
            (instance_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
        context_text = trace.get("context_text") if isinstance(trace, dict) else None
        if context_text:
            (instance_dir / "context.txt").write_text(context_text, encoding="utf-8")
        if summary_text or context_text:
            compare_md = []
            compare_md.append("# Recovery Context vs Summary")
            if context_text:
                compare_md.append("## Context\n```\n" + context_text + "\n```")
            if summary_text:
                compare_md.append("## Summary\n```\n" + summary_text + "\n```")
            (instance_dir / "compare.md").write_text("\n\n".join(compare_md), encoding="utf-8")
    except Exception as e:
        logger.warning(f"保存摘要/上下文视图失败: {e}")

    # 1.2 保存事件流，便于后处理/grep
    try:
        events_path = instance_dir / "events.jsonl"
        events = []
        ts = datetime.now().isoformat()
        if summary_text:
            events.append({
                "ts": ts,
                "kind": "summary",
                "instance_id": instance_id,
                "run_id": run_id,
                "model": model_name,
                "fault_type": final_state.get("fault_type"),
                "target_role": trace.get("target_role") if isinstance(trace, dict) else None,
                "strategy": trace.get("strategy_id") if isinstance(trace, dict) else None,
                "granularity": trace.get("granularity") if isinstance(trace, dict) else None,
                "context_len": trace.get("context_len") if isinstance(trace, dict) else None,
                "summary_len": len(summary_text),
            })
        events.append({
            "ts": ts,
            "kind": "final",
            "instance_id": instance_id,
            "run_id": run_id,
            "model": model_name,
            "success": final_state.get("success", False),
            "fault_type": final_state.get("fault_type"),
            "recovery_count": final_state.get("recovery_count", 0),
            "elapsed": final_state.get("elapsed"),
            "patch_len": len(final_state.get("patch", "") or ""),
        })
        with open(events_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"保存events.jsonl失败: {e}")
    
    # 2. 保存SWE-bench格式的预测结果
    predictions_file = run_dir / "predictions.json"
    
    # 读取现有预测
    predictions = {}
    if predictions_file.exists():
        with open(predictions_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    
    # 添加/更新当前实例
    predictions[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": final_state.get("patch", ""),
    }
    
    # 保存预测
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    logger.info(f"预测已保存: {predictions_file}")
    
    return trajectory_path, predictions_file


def save_summary(
    total: int,
    successful: int,
    failed: int,
    execution_time: float,
    *,
    run_id: str = "default",
    output_root: Path | None = None,
) -> Path:
    """保存运行摘要
    
    Args:
        total: 总任务数
        successful: 成功数
        failed: 失败数
        execution_time: 执行时间（秒）
        
    Returns:
        摘要文件路径
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_instances": total,
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total if total > 0 else 0,
        "execution_time_seconds": execution_time,
    }
    
    out_root = output_root or OUTPUT_DIR
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"摘要已保存: {summary_path}")
    return summary_path
