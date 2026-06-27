"""批量运行脚本"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

from swe_mas import OUTPUT_DIR
from swe_mas.models.qwen_model import QwenModel, QwenModelConfig
from swe_mas.utils.env_executor import LocalExecutor
from swe_mas.utils.logger import setup_logger
from swe_mas.workflow.orchestrator import SWEOrchestrator
from swe_mas.run.save import save_results, save_summary
from swe_mas.utils.repo_utils import prepare_instance_repo


# SWE-bench数据集映射
DATASET_MAPPING = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def process_instance_fixed(
    instance: dict,
    working_directory: str,
    model_name: str,
    logger,
    *,
    max_iters: int | None = None,
    run_id: str = "default",
    output_root: Path | None = None,
) -> tuple[str, bool]:
    """处理单个实例（使用固定工作目录）"""
    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]

    try:
        logger.info(f"开始处理: {instance_id}")

        model = QwenModel(config=QwenModelConfig(model_name=model_name))
        executor = LocalExecutor(cwd=working_directory)
        orchestrator = SWEOrchestrator(
            model=model,
            executor=executor,
        )
        if max_iters is not None:
            orchestrator.max_iterations = max_iters

        final_state = orchestrator.run(
            problem_statement=problem_statement,
            working_directory=working_directory,
        )

        save_results(
            instance_id=instance_id,
            problem_statement=problem_statement,
            final_state=final_state,
            model_name=model_name,
            run_id=run_id,
            output_root=output_root,
        )

        success = final_state.get("success", False) and bool(final_state.get("patch"))
        logger.info(f"完成处理: {instance_id} - {'成功' if success else '失败'}")

        return instance_id, success

    except Exception as e:
        logger.error(f"处理 {instance_id} 时出错: {str(e)}")
        return instance_id, False


def process_instance_dynamic(
    instance: dict,
    work_root: str,
    model_name: str,
    logger,
    *,
    use_ssh: bool = False,
    checkout: bool = True,
    max_iters: int | None = None,
    run_id: str = "default",
    output_root: Path | None = None,
) -> tuple[str, bool]:
    """处理单个实例（自动克隆并按 base_commit checkout）"""
    instance_id = instance.get("instance_id")
    problem_statement = instance.get("problem_statement", "")

    try:
        # 为实例准备独立工作目录
        workdir, ok, err = prepare_instance_repo(
            instance,
            work_root,
            use_ssh=use_ssh,
            checkout=checkout,
        )
        if not ok or not workdir:
            logger.error(f"准备仓库失败: {instance_id} - {err}")
            return instance_id, False

        logger.info(f"开始处理: {instance_id}")

        model = QwenModel(config=QwenModelConfig(model_name=model_name))
        executor = LocalExecutor(cwd=workdir)
        orchestrator = SWEOrchestrator(
            model=model,
            executor=executor,
        )
        if max_iters is not None:
            orchestrator.max_iterations = max_iters

        final_state = orchestrator.run(
            problem_statement=problem_statement,
            working_directory=workdir,
        )

        save_results(
            instance_id=instance_id,
            problem_statement=problem_statement,
            final_state=final_state,
            model_name=model_name,
            run_id=run_id,
            output_root=output_root,
        )

        success = final_state.get("success", False) and bool(final_state.get("patch"))
        logger.info(f"完成处理: {instance_id} - {'成功' if success else '失败'}")

        return instance_id, success

    except Exception as e:
        logger.error(f"处理 {instance_id} 时出错: {str(e)}")
        return instance_id, False


def main():
    """批量处理主函数"""
    parser = argparse.ArgumentParser(description="批量运行SWE-bench实例")
    parser.add_argument("--dataset", type=str, default="lite", choices=list(DATASET_MAPPING.keys()), help="数据集名称")
    parser.add_argument("--split", type=str, default="test", help="数据集分割")
    parser.add_argument("--working-dir", type=str, default=".", help="固定工作目录（若提供则不自动克隆）")
    parser.add_argument("--work-root", type=str, default="/root/code/work", help="自动克隆的根目录（按实例创建子目录）")
    parser.add_argument("--model", type=str, default="qwen-max", help="模型名称")
    parser.add_argument("--workers", type=int, default=1, help="并行worker数量")
    parser.add_argument("--limit", type=int, help="限制处理的实例数量")
    parser.add_argument("--max-iters", type=int, default=10, help="每个Agent的最大迭代次数")
    parser.add_argument("--instances", type=str, help="逗号分隔的实例ID列表，仅处理这些实例")
    parser.add_argument("--instances-file", type=Path, help="包含实例ID的文件（每行一个）")
    parser.add_argument("--use-ssh", action="store_true", help="克隆仓库时使用SSH")
    parser.add_argument("--no-checkout", dest="checkout", action="store_false", help="不按base_commit强制checkout")
    parser.set_defaults(checkout=True)
    parser.add_argument("--run-id", type=str, default=None, help="运行ID，用于输出分组；默认时间戳")
    parser.add_argument("--output-root", type=Path, default=None, help="输出根目录，默认 outputs/")
    
    args = parser.parse_args()
    
    # 设置日志
    out_root = args.output_root or OUTPUT_DIR
    run_id = args.run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = out_root / run_id
    log_file = run_dir / "swe_mas.log"
    logger = setup_logger("swe_mas", log_file=log_file)
    
    logger.info(f"=" * 80)
    logger.info(f"批量处理SWE-bench实例")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"分割: {args.split}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"=" * 80)
    
    # 加载数据集
    dataset_path = DATASET_MAPPING[args.dataset]
    logger.info(f"加载数据集: {dataset_path}")
    dataset = load_dataset(dataset_path, split=args.split)

    # 实例过滤（优先于limit）
    selected_ids: set[str] | None = None
    if args.instances:
        selected_ids = {s.strip() for s in args.instances.split(",") if s.strip()}
    elif args.instances_file and args.instances_file.exists():
        selected_ids = {line.strip() for line in args.instances_file.read_text(encoding="utf-8").splitlines() if line.strip()}

    if selected_ids:
        dataset = dataset.filter(lambda e: e.get("instance_id") in selected_ids)

    # 限制数量
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    logger.info(f"共 {len(dataset)} 个实例待处理")
    
    # 并行处理
    start_time = time.time()
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        if args.working_dir and args.working_dir != ".":
            # 使用固定工作目录（向后兼容旧用法）
            futures = {
                executor.submit(
                    process_instance_fixed,
                    instance,
                    args.working_dir,
                    args.model,
                    logger,
                    max_iters=args.max_iters,
                    run_id=run_id,
                    output_root=args.output_root,
                ): instance["instance_id"]
                for instance in dataset
            }
        else:
            # 自动克隆并checkout（推荐）
            futures = {
                executor.submit(
                    process_instance_dynamic,
                    instance,
                    args.work_root,
                    args.model,
                    logger,
                    use_ssh=args.use_ssh,
                    checkout=args.checkout,
                    max_iters=args.max_iters,
                    run_id=run_id,
                    output_root=args.output_root,
                ): instance["instance_id"]
                for instance in dataset
            }
        
        for future in as_completed(futures):
            instance_id, success = future.result()
            if success:
                successful += 1
            else:
                failed += 1
            
            logger.info(f"进度: {successful + failed}/{len(dataset)} (成功: {successful}, 失败: {failed})")
    
    # 保存摘要
    elapsed_time = time.time() - start_time
    save_summary(
        total=len(dataset),
        successful=successful,
        failed=failed,
        execution_time=elapsed_time,
        run_id=run_id,
        output_root=args.output_root,
    )
    
    logger.info(f"=" * 80)
    logger.info(f"批量处理完成")
    logger.info(f"总计: {len(dataset)}")
    logger.info(f"成功: {successful}")
    logger.info(f"失败: {failed}")
    logger.info(f"成功率: {successful / len(dataset) * 100:.2f}%")
    logger.info(f"总耗时: {elapsed_time:.2f}秒")
    logger.info(f"=" * 80)


if __name__ == "__main__":
    main()
