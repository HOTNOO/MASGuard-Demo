"""单实例运行脚本"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from swe_mas import OUTPUT_DIR
from swe_mas.models.qwen_model import QwenModel, QwenModelConfig
from swe_mas.utils.env_executor import LocalExecutor
from swe_mas.utils.logger import setup_logger
from swe_mas.workflow.orchestrator import SWEOrchestrator
from swe_mas.run.save import save_results


def run_single_instance(
    instance_id: str,
    problem_statement: str,
    working_directory: str,
    model_name: str = "qwen-max",
    run_id: str = "default",
    output_root: Path | None = None,
    max_iterations_per_agent: int = 10,
) -> dict:
    """运行单个SWE-bench实例
    
    Args:
        instance_id: 实例ID
        problem_statement: 问题描述
        working_directory: 工作目录
        model_name: 模型名称
        
    Returns:
        最终状态字典
    """
    # 设置日志
    out_root = output_root or OUTPUT_DIR
    run_dir = out_root / run_id
    log_file = run_dir / f"{instance_id}.log"
    logger = setup_logger("swe_mas", log_file=log_file)
    
    logger.info(f"=" * 80)
    logger.info(f"开始处理实例: {instance_id}")
    logger.info(f"=" * 80)
    
    start_time = time.time()
    
    try:
        # 初始化模型
        model = QwenModel(config=QwenModelConfig(model_name=model_name))
        logger.info(f"模型初始化完成: {model_name}")
        
        # 初始化执行器
        executor = LocalExecutor(cwd=working_directory)
        logger.info(f"执行器初始化完成，工作目录: {working_directory}")
        
        # 创建编排器
        orchestrator = SWEOrchestrator(
            model=model,
            executor=executor,
            max_iterations_per_agent=max_iterations_per_agent,
        )
        
        # 运行工作流
        final_state = orchestrator.run(
            problem_statement=problem_statement,
            working_directory=working_directory,
        )
        
        # 保存结果
        trajectory_path, predictions_path = save_results(
            instance_id=instance_id,
            problem_statement=problem_statement,
            final_state=final_state,
            model_name=model_name,
            run_id=run_id,
            output_root=output_root,
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"=" * 80)
        logger.info(f"实例 {instance_id} 处理完成")
        logger.info(f"耗时: {elapsed_time:.2f}秒")
        logger.info(f"状态: {'成功' if final_state.get('success') else '失败'}")
        if final_state.get("patch"):
            logger.info(f"补丁长度: {len(final_state['patch'])} 字符")
        logger.info(f"=" * 80)
        
        return final_state
        
    except Exception as e:
        logger.error(f"处理实例 {instance_id} 时发生异常: {str(e)}", exc_info=True)
        raise


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="运行单个SWE-bench实例")
    parser.add_argument("--instance-id", type=str, required=True, help="实例ID")
    parser.add_argument("--problem", type=str, help="问题描述（或使用--problem-file）")
    parser.add_argument("--problem-file", type=Path, help="问题描述文件路径")
    parser.add_argument("--working-dir", type=str, default=".", help="工作目录")
    parser.add_argument("--model", type=str, default="qwen-max", help="模型名称")
    parser.add_argument("--run-id", type=str, default=None, help="运行ID，用于输出分组；默认使用时间戳")
    parser.add_argument("--output-root", type=Path, default=None, help="输出根目录，默认 outputs/")
     # 每个Agent的最大迭代次数（避免长时间卡在定位/复现等阶段）
    parser.add_argument("--max-iters", type=int, default=10, help="每个Agent的最大迭代次数，默认 10 步")
    
    args = parser.parse_args()
    
    # 获取问题描述
    if args.problem:
        problem_statement = args.problem
    elif args.problem_file:
        problem_statement = Path(args.problem_file).read_text(encoding="utf-8")
    else:
        raise ValueError("必须提供--problem或--problem-file参数")
    
    # 运行
    run_id = args.run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")

    run_single_instance(
        instance_id=args.instance_id,
        problem_statement=problem_statement,
        working_directory=args.working_dir,
        model_name=args.model,
        run_id=run_id,
        output_root=args.output_root,
        max_iterations_per_agent=args.max_iters,
    )


if __name__ == "__main__":
    main()
