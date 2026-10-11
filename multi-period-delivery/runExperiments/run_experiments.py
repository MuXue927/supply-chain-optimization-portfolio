#!/usr/bin/env python3
"""
批量运行实验模板脚本 (scripts/run_experiments.py)

功能:
- 接受实例列表、算法变体、seeds、time_limit、并行度、输出目录等参数
- 对每个 (instance, variant, seed) 组合创建独立输出目录并运行 ALNSCode 优化器
- 在每次运行前通过 ALNSCode.alns_config.default_config.update_from_dict 临时设置 ENABLE_PARAM_TUNER / ENABLE_ML_OPERATORS / SEED / MAX_RUNTIME / LOG_DIR
- 每次运行结束写入 run_metadata.json（包含元数据与运行状态），便于后续 prepare_thesis_results.py 汇总

说明:
- 在 Windows 上建议使用 --parallel 以 1 为默认值（Gurobi/第三方库可能对并行有限制）。
- 本脚本假定 ALNSCode.alns_optimizer.ALNSOptimizer 的签名为:
    ALNSOptimizer(input_file_loc, output_file_loc)
  并提供方法 run_optimization(dataset_name) -> bool
- 若项目中具体接口略有不同，请根据本脚本结构做小范围适配。
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import glob
import time
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import multiprocessing as mp

RUN_META_NAME = "run_metadata.json"

# Ensure repository root is importable when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def utc_now_iso_z() -> str:
    """Return a timezone-aware UTC ISO-8601 timestamp ending with 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_seeds(seeds_spec: str) -> List[int]:
    """
    支持形式:
      - "0-9" -> [0..9]
      - "0,2,5" -> [0,2,5]
      - "0" -> [0]
    """
    if "-" in seeds_spec and "," not in seeds_spec:
        a, b = seeds_spec.split("-")
        return list(range(int(a), int(b) + 1))
    parts = [p.strip() for p in seeds_spec.split(",")]
    out = []
    for p in parts:
        if p == "":
            continue
        out.append(int(p))
    return out


def is_valid_instance(path_str: str):
    """
    快速校验实例路径是否看起来像一个可运行的实例。
    返回 (ok: bool, reason: str)

    支持的格式：
    - 具体的 dataset_N 目录（如 datasets/multiple-periods/small/dataset_1）
      必须包含 order.csv 文件

    校验项（best-effort）：
     - 若是文件：标记为invalid
     - 若是dataset_*目录：检查order.csv是否存在；存在则接受
     - 若是其他目录：不接受（容器目录需逐个指定dataset_*）
    """
    p = Path(path_str)
    if p.is_file():
        return False, "path is a file; expected a dataset directory"

    # 只接受 dataset_* 目录格式
    if p.name.lower().startswith("dataset_"):
        orderf = p / "order.csv"
        if orderf.exists():
            return True, "ok (dataset_* directory)"
        else:
            return False, "dataset_* directory but order.csv missing"

    # 不接受其他目录格式（如容器目录small, medium, large）
    # 用户应该明确指定每个dataset_*
    return False, "not a dataset_* directory (use dataset_1, dataset_2, ...)"


def expand_container_to_datasets(container_path: str) -> List[str]:
    """
    将容器目录（small/medium/large）展开为所有包含的 dataset_* 目录。

    示例:
    - 输入: datasets/multiple-periods/small
    - 输出: [
        datasets/multiple-periods/small/dataset_1,
        datasets/multiple-periods/small/dataset_2,
        ...
      ]

    返回展开的dataset列表（已排序）。
    """
    container = Path(container_path)
    if not container.is_dir():
        return []

    datasets = []
    try:
        for sub in sorted(container.iterdir()):
            if sub.is_dir() and sub.name.lower().startswith("dataset_"):
                datasets.append(str(sub))
    except Exception:
        pass

    return datasets


def expand_instances(patterns: List[str]) -> List[str]:
    paths = []
    for p in patterns:
        if "*" in p or "?" in p:
            paths.extend(sorted(glob.glob(p)))
        else:
            # If the path is a directory, treat the directory itself as an instance
            # (do not expand to individual files). This avoids passing config files
            # or unrelated CSVs as instances.
            if os.path.isdir(p):
                paths.append(p)
            else:
                paths.append(p)
    # dedupe and keep order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    # 过滤掉可能不是实例的配置文件（例如 model_config*.json）
    filtered = [p for p in out if "model_config" not in Path(p).name.lower()]

    # 首先尝试验证每个路径
    # 若路径是容器目录（small/medium/large），自动展开到dataset_*子目录
    pre_validated = []
    for path_str in filtered:
        ok, reason = is_valid_instance(path_str)
        if ok:
            pre_validated.append(path_str)
        else:
            # 检查是否是容器目录
            container_datasets = expand_container_to_datasets(path_str)
            if container_datasets:
                print(
                    f"Auto-expanding container: {path_str} -> {len(container_datasets)} datasets",
                    flush=True,
                )
                pre_validated.extend(container_datasets)
            else:
                # 标记为无效候选
                pre_validated.append((path_str, reason))  # 元组表示验证失败

    # 应用最终验证逻辑到所有候选实例
    valid_instances = []
    skipped = []
    for item in pre_validated:
        if isinstance(item, tuple):
            path_str, reason = item
            skipped.append((path_str, reason))
        else:
            path_str = item
            ok, reason = is_valid_instance(path_str)
            if ok:
                valid_instances.append(path_str)
            else:
                skipped.append((path_str, reason))

    # 如果有被跳过的实例，打印日志
    if skipped:
        print("Skipped instances (invalid):", flush=True)
        for path_str, reason in skipped:
            print(f"  - {path_str}  => {reason}", flush=True)

    return valid_instances


def make_variant_flags(variant: str):
    """
    variant -> (enable_tuner, enable_ml)
    """
    v = variant.strip().lower()
    if v in ("alns-b", "basic"):
        return False, False
    if v in ("alns-t", "tuner", "paramtuner"):
        return True, False
    if v in ("alns-ml", "ml", "mlo", "mlops", "mloperators"):
        return False, True
    if v in ("alns-f", "full", "tuner+ml", "full+ml"):
        return True, True
    if v in ("monolithic", "gurobi", "mono"):
        return None, None
    raise ValueError(f"Unknown variant: {variant}")


def run_single(args_tuple):
    """
    Worker 函数，在新的进程/线程中导入 ALNSCode 并执行一次运行。
    args_tuple: (instance_path, variant, seed, time_limit, out_base_dir)
    """
    if len(args_tuple) >= 6:
        instance_path, variant, seed, time_limit, out_base_dir, isolate_worker_log = (
            args_tuple
        )
    else:
        instance_path, variant, seed, time_limit, out_base_dir = args_tuple
        isolate_worker_log = False
    start_ts = time.time()
    t0 = utc_now_iso_z()
    instance_path = str(Path(instance_path).resolve())
    basename = Path(instance_path).stem
    safe_variant = variant.replace(",", "_")
    run_dir = (
        Path(out_base_dir)
        / f"{basename}__{safe_variant}__seed{seed}__t{int(time_limit)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    worker_log_file = run_dir / "worker_console.log"

    # 如果实例是一个数据目录且缺失 model_config_alns.json，尝试从上级 datasets 目录回退复制一份
    try:
        inst_path_obj = Path(instance_path)
        if inst_path_obj.is_dir():
            cfg_name = "model_config_alns.json"
            dst_cfg = inst_path_obj / cfg_name
            if not dst_cfg.exists():
                # 尝试在 parent 级别（如 datasets/multiple-periods）寻找全局配置并复制
                parent_cfg = (
                    Path("datasets") / "multiple-periods" / "model_config_alns.json"
                )
                if parent_cfg.exists():
                    try:
                        import shutil

                        shutil.copy(parent_cfg, dst_cfg)
                    except Exception:
                        # best-effort,继续让后续逻辑处理可能的缺失
                        pass

    except Exception:
        # Ignore any issues in best-effort fallback; main run will record errors if data still missing
        pass

    # Import inside worker to avoid pickling/import issues
    try:
        from ALNSCode.alns_config import default_config as ALNSConfig
        from ALNSCode.alns_optimizer import ALNSOptimizer
    except Exception as e:
        meta = {
            "instance": instance_path,
            "variant": variant,
            "seed": seed,
            "time_limit": time_limit,
            "start_time": t0,
            "status": "import_error",
            "error": str(e),
        }
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
        return meta

    # Compute flags for variant
    try:
        enable_tuner, enable_ml = make_variant_flags(variant)
    except Exception as e:
        meta = {
            "instance": instance_path,
            "variant": variant,
            "seed": seed,
            "time_limit": time_limit,
            "start_time": t0,
            "status": "bad_variant",
            "error": str(e),
        }
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
        return meta

    v = variant.lower()

    # Run optimizer
    ok = False
    err = None
    alns_meta = {}
    mono_meta = {}
    out_ctx = nullcontext()
    log_fh = None
    if isolate_worker_log:
        log_fh = open(worker_log_file, "a", encoding="utf-8", buffering=1)
        out_ctx = nullcontext(log_fh)

    try:
        with out_ctx as _fh:
            std_ctx = redirect_stdout(_fh) if (_fh is not None) else nullcontext()
            err_ctx = redirect_stderr(_fh) if (_fh is not None) else nullcontext()
            with std_ctx, err_ctx:
                # 如果传入的是具体的 dataset_N 目录，则将其 parent 作为输入路径，dataset_name 为该目录名
                inst_obj = Path(instance_path)
                if inst_obj.name.lower().startswith("dataset_") and inst_obj.is_dir():
                    input_loc = str(inst_obj.parent)
                    dataset_name = inst_obj.name
                else:
                    input_loc = instance_path
                    dataset_name = basename

                if v in ("monolithic", "gurobi", "mono"):
                    # Monolithic / Gurobi baseline path
                    from monolithic.optLogicMultiple import run_model_from_paths

                    mono_meta = run_model_from_paths(
                        input_file_loc=input_loc,
                        output_file_loc=str(run_dir),
                        dataset_name=dataset_name,
                        time_limit=int(time_limit),
                        write_outputs=True,
                    )
                    # 合并关键信息到统一元数据格式
                    if isinstance(mono_meta, dict):
                        status = mono_meta.get("status", "unknown")
                        ok = status in (
                            "optimal",
                            "success",
                            "time_limit",
                            "suboptimal",
                        )
                    else:
                        mono_meta = {}
                        ok = False
                else:
                    # Configure ALNS runtime via default_config
                    cfg_updates = {
                        "ENABLE_PARAM_TUNER": bool(enable_tuner),
                        "ENABLE_ML_OPERATORS": bool(enable_ml),
                        "SEED": int(seed),
                        "MAX_RUNTIME": int(time_limit),
                        # 每次运行单独日志目录
                        "LOG_DIR": str(run_dir / "logs"),
                    }
                    ALNSConfig.update_from_dict(cfg_updates)

                    # 输出目录传给 ALNSOptimizer；传入的 input_loc 应为包含 dataset_* 子目录的父路径
                    optimizer = ALNSOptimizer(input_loc, str(run_dir))
                    ok = optimizer.run_optimization(dataset_name)

                    # Collect lightweight ALNS tracking metrics for run_metadata.json
                    # Keep only scalar values to avoid huge metadata and non-serializable objects.
                    try:
                        tracker = getattr(optimizer, "tracker", None)
                        if tracker is not None and hasattr(tracker, "get_statistics"):
                            stats = tracker.get_statistics() or {}
                            total_iterations = int(
                                stats.get("total_iterations", 0) or 0
                            )
                            best_actual_obj = stats.get("best_actual_objective", None)
                            best_search_obj = stats.get("best_objective", None)
                            elapsed_time = stats.get("elapsed_time", None)

                            # Prefer tracker elapsed_time; fall back to this run wall-clock later.
                            iter_per_second = None
                            try:
                                elapsed_val = (
                                    float(elapsed_time)
                                    if elapsed_time is not None
                                    else None
                                )
                                if elapsed_val is not None and elapsed_val > 1e-9:
                                    iter_per_second = (
                                        float(total_iterations) / elapsed_val
                                    )
                            except Exception:
                                iter_per_second = None

                            alns_meta = {
                                "total_iterations": total_iterations,
                                "best_actual_objective": best_actual_obj,
                                "best_search_objective": best_search_obj,
                                "tracker_elapsed_seconds": elapsed_time,
                                "iter_per_second": iter_per_second,
                            }
                    except Exception:
                        # Best-effort only; experiment should not fail due to metadata enrichment.
                        alns_meta = {}

                status = "success" if ok else "failed"
    except Exception as e:
        ok = False
        status = "runtime_error"
        err = str(e)
    finally:
        if log_fh is not None:
            log_fh.close()
    end_ts = time.time()
    meta = {
        "instance": instance_path,
        "variant": variant,
        "seed": seed,
        "time_limit": time_limit,
        "start_time": t0,
        "end_time": utc_now_iso_z(),
        "wall_seconds": end_ts - start_ts,
        "status": status,
    }
    # attach error if present (covers both exception and explicit failure cases)
    if err is not None:
        meta["error"] = err

    # merge monolithic/gurobi specific metadata when available
    if isinstance(mono_meta, dict) and mono_meta:
        for k in (
            "final_obj",
            "best_bound",
            "gap_pct",
            "mip_gap",
            "gurobi_runtime",
            "iterations",
            "node_count",
            "vehicle_count",
            "output_files",
            "warnings",
        ):
            if k in mono_meta:
                meta[k] = mono_meta[k]

    # merge ALNS-specific tracking metadata when available
    if isinstance(alns_meta, dict) and alns_meta:
        for k, v in alns_meta.items():
            meta[k] = v

        # Ensure iter_per_second is always available when total_iterations exists.
        if meta.get("iter_per_second", None) is None:
            try:
                total_iterations = int(meta.get("total_iterations", 0) or 0)
                wall = float(meta.get("wall_seconds", 0.0) or 0.0)
                if wall > 1e-9:
                    meta["iter_per_second"] = float(total_iterations) / wall
            except Exception:
                pass

    # write metadata
    try:
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
    except Exception:
        # best-effort
        pass

    return meta


def main():
    parser = argparse.ArgumentParser(description="Run ALNS experiments (template).")
    parser.add_argument(
        "--instances",
        nargs="+",
        required=True,
        help="实例路径或 glob 模式, 支持多个，例如 data/small/*.json 或 datasets/small/dir",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="ALNS-B,ALNS-F",
        help=(
            "Comma-separated variants. Thesis names: "
            "ALNS-B,ALNS-T,ALNS-ML,ALNS-F; monolithic is also supported."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-4",
        help="随机种子范围或列表，例如 0-9 或 0,2,5",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="每次运行的 wall-clock time limit (秒)，会写入 ALNSConfig.MAX_RUNTIME",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="OutPut-ALNS/experiments",
        help="实验输出根目录",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="并行进程数 (Windows 建议 1 或小于 CPU 核心数)",
    )

    args = parser.parse_args()

    instances = expand_instances(args.instances)
    if not instances:
        print("No instances found after validation. Exiting.")
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build tasks
    tasks = []
    for inst in instances:
        for var in variants:
            for s in seeds:
                tasks.append((inst, var, s, args.time_limit, str(out_dir)))

    print(
        f"Discovered {len(instances)} instances, {len(variants)} variants, {len(seeds)} seeds -> total runs: {len(tasks)}"
    )
    print(f"Output base dir: {out_dir}")
    start = time.time()

    if args.parallel is None or args.parallel <= 1:
        results = []
        for t in tasks:
            print(f"Running: instance={Path(t[0]).name} variant={t[1]} seed={t[2]}")
            res = run_single((*t, False))
            results.append(res)
    else:
        # multiprocessing with spawn (Windows safe)
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=args.parallel) as pool:
            parallel_tasks = [(*t, True) for t in tasks]
            for res in pool.imap_unordered(run_single, parallel_tasks):
                print(json.dumps(res, ensure_ascii=False))
        results = []  # pool already printed metadata

    elapsed = time.time() - start
    summary = {
        "n_instances": len(instances),
        "n_variants": len(variants),
        "n_seeds": len(seeds),
        "total_runs": len(tasks),
        "elapsed_seconds": elapsed,
        "timestamp": utc_now_iso_z(),
    }
    (out_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(
        "Experiment finished. Summary written to:", out_dir / "experiment_summary.json"
    )


if __name__ == "__main__":
    main()
