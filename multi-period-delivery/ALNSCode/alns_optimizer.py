"""
alns_optimizer.py
模块说明（中文）：
- 模块定位：
  本模块为项目中 ALNS（Adaptive Large Neighborhood Search）优化流程的高层控制器与结果后处理组件。
  它封装数据加载、初始解生成、算子注册、ALNS 配置/运行、以及结果导出与可视化等职责，
  是运行启发式求解器并将结果转化为可分析输出的入口。

- 核心职责：
  1. 管理输入/输出路径与数据加载（load_data）
  2. 生成并校验初始解（create_initial_solution）
  3. 构建并注册 destroy / repair 算子（_register_destroy_operators/_register_repair_operators）
  4. 配置 ALNS（setup_alns）、选择器与接受/停止准则，并执行迭代（run_optimization）
  5. 将求解结果写出 CSV、生成统计与可视化图表（_process_results、_generate_reports 等）
  6. 支持可选的参数自动调整（ParamAutoTuner）与基于 ML 的算子选择（MLOperatorSelector）

- 对外接口（主要）：
  - ALNSOptimizer 类：主要方法包括 load_data, create_initial_solution, setup_alns, run_optimization
  - is_file_in_use(file_path)：工具函数，用于检测文件是否被占用（注意：在不同平台上可能不可靠）

- 维护与改进建议（后续可逐步落地）：
  * 将 imports 分组（标准库 / 第三方 / 项目内部）并移除未使用的模块（例如 sys）
  * 将 matplotlib 延迟导入到需要绘图的函数内（减少模块导入开销，便于单元测试）
  * 使用原子写入（写入临时文件并 os.replace）替代直接 append，以避免并发/重复写入问题
  * 将 SegmentedRouletteWheel 的 num_destroy/num_repair 动态计算为实际注册的算子数，避免硬编码不一致
  * 避免访问对象的私有属性（例如 tracker._iteration_count / stop._max_iterations），改为公开接口或增加安全判断
  * 在对 pandas DataFrame 做删除/赋值前检查列存在性并使用 copy() 避免 SettingWithCopy 警告
  * 为文件写入加入重试与超时机制（在 Windows 下 is_file_in_use 并非百分百可靠）
  * 在可能为空的列表/字典访问处增加防护（例如 result.statistics.objectives 可能为空）
  * 将日志分层：保持 logger 用于系统级错误，LogPrinter 用于用户可见输出

注意：下面的实现修改应逐步应用并在每一步后运行单元/烟雾测试以确认行为未改变。
"""

from __future__ import annotations
import os
import csv
import time
import datetime
import logging
import builtins
import io
import pandas as pd
import numpy as np
import numpy.random as rnd
from typing import Any, Dict, Optional, TYPE_CHECKING, cast
from pathlib import Path
from contextlib import contextmanager, redirect_stdout

# Package-internal imports used by the optimizer
from .alnsopt import SolutionState, initial_solution
from .destroy_operators import (
    infeasible_removal,
    # wrappers / factories
    create_random_removal,
    create_shaw_removal,
    create_periodic_shaw_removal,
    create_worst_removal,
    create_surplus_inventory_removal,
    create_path_removal,
    create_inventory_level_destruction,
    create_cross_period_dealer_removal,
    create_period_range_removal,
    wrap_operator_no_args,
)
from .repair_operators import (
    greedy_repair,
    inventory_balance_repair,
    infeasible_repair,
    # wrapper factories for repair operators
    wrap_repair_no_args,
    create_local_search_repair,
    create_smart_batch_repair,
    create_regret_repair,
    create_sequential_repair,
    create_lookahead_repair,
    FleetConsolidationRepairOperator,
)
from .InputDataALNS import DataALNS
from .optutility import LogPrinter
from .alnstrack import ALNSTracker
from .combined_stopping import create_standard_combined_criterion
from .alns_config import default_config as ALNSConfig
# visualization functions are imported lazily in _generate_reports / startup plotting
# to reduce module import overhead (matplotlib/pandas are heavy). They will be imported
# inside those functions when actually needed.


from .param_tuner import ParamAutoTuner
from .ml_operator_selector import MLOperatorSelector
from .ml_destroy_selection import ml_based_destroy
from .ml_repair_selection import ml_based_repair
from .repair_utils import consolidate_route_vehicles
from .accept import DemandConstraintAccept
from .callbacks import register_callbacks


# For type checking only (avoid runtime import when modules are optional)
if TYPE_CHECKING:
    from .param_tuner import ParamAutoTuner
    from .ml_operator_selector import MLOperatorSelector

# Third-party ALNS imports
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import SegmentedRouletteWheel
from alns.Result import Result


@contextmanager
def suppress_noisy_operator_stdout(enabled: bool = True):
    """Suppress high-volume operator prints while preserving key progress logs."""
    if not enabled:
        yield
        return

    noisy_tags = ("[OPLOG]", "[DIAG]", "[OPJSON]")
    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        try:
            msg = " ".join(str(a) for a in args)
        except Exception:
            msg = ""
        if any(tag in msg for tag in noisy_tags):
            return
        return original_print(*args, **kwargs)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print


# Config constants referenced by optimizer (use getattr for defensive access)
SEED = ALNSConfig.SEED
DATASET_TYPE = ALNSConfig.DATASET_TYPE
DATASET_IDX = ALNSConfig.DATASET_IDX

logger = logging.getLogger(__name__)


# Utility: check if file is in use (moved here to avoid circular imports)
def is_file_in_use(file_path):
    """
    检查文件是否被占用（被其他程序打开）
    """
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, "a"):
            pass
        return False
    except PermissionError:
        return True


def safe_state_snapshot(state: Any) -> Any:
    if state is None:
        return None
    try:
        if hasattr(state, "copy") and callable(getattr(state, "copy")):
            return state.copy()
    except Exception:
        pass
    try:
        import copy as _copy

        return _copy.deepcopy(state)
    except Exception:
        return state


class ALNSOptimizer:
    """ALNS优化器主类, 封装了整个优化流程"""

    def __init__(self, input_file_loc: Any = None, output_file_loc: Any = None):
        """
        初始化ALNS优化器

        Parameters:
        -----------
        input_file_loc : str, optional
            输入文件位置, 如果为None则使用默认路径
        output_file_loc : str, optional
            输出文件位置, 如果为None则使用默认路径
        """
        # 设置默认路径
        if input_file_loc is None or output_file_loc is None:
            current_dir = Path(__file__).parent
            par_path = current_dir.parent

            if input_file_loc is None:
                input_file_loc = (
                    par_path / "datasets" / "multiple-periods" / DATASET_TYPE
                )
            if output_file_loc is None:
                output_file_loc = (
                    par_path / "OutPut-ALNS" / "multiple-periods" / DATASET_TYPE
                )

        self.input_file_loc = Path(input_file_loc)
        self.output_file_loc = Path(output_file_loc)

        # 初始化日志打印器
        self.log_printer = LogPrinter(time.time())

        # 初始化数据和状态
        self.data: Optional[DataALNS] = None
        self.best_solution: Optional[SolutionState] = None
        self.best_feasible_state: Optional[SolutionState] = None
        self.best_feasible_obj: float = float("inf")
        self.result: Optional[Result] = None
        self.tracker: Optional[ALNSTracker] = None

        # 初始化参数自动调整器和ML选择器
        self.param_tuner: Optional[ParamAutoTuner] = None
        self.ml_selector: Optional[MLOperatorSelector] = None

        # 随机数生成器
        self.rng = rnd.default_rng(seed=SEED)
        # 记录已注册的算子数量（用于动态设置选择器参数）
        self._num_registered_destroy_ops: int = 0
        self._num_registered_repair_ops: int = 0
        self._iteration: int = 0
        self._max_iterations: int = int(getattr(ALNSConfig, "MAX_ITERATIONS", 0))
        self.progress: float = 0.0
        self.acceptance = None
        self._background_executor: Any = None
        self._ml_retrain_future: Any = None
        self._incumbent_state_snapshot: Optional[SolutionState] = None

    def get_iteration(self) -> int:
        return int(getattr(self, "_iteration", 0))

    def is_param_tuner_enabled(self) -> bool:
        return bool(getattr(ALNSConfig, "ENABLE_PARAM_TUNER", False))

    def is_ml_enabled(self) -> bool:
        return bool(getattr(ALNSConfig, "ENABLE_ML_OPERATORS", False))

    def increment_iteration(self) -> int:
        self._iteration = int(getattr(self, "_iteration", 0)) + 1
        return self._iteration

    def get_max_iterations(self) -> int:
        return int(
            getattr(self, "_max_iterations", getattr(ALNSConfig, "MAX_ITERATIONS", 0))
        )

    def set_max_iterations(self, max_iterations: int):
        self._max_iterations = int(max_iterations)

    def load_data(self, dataset_name: str) -> bool:
        """
        加载数据

        Parameters:
        -----------
        dataset_name : str
            数据集名称

        Returns:
        --------
        bool : 是否加载成功
        """
        try:
            self.log_printer.print("Loading Data...")
            start_time = time.time()

            # 构建完整的输出路径
            full_output_path = self.output_file_loc / dataset_name

            self.data = DataALNS(
                str(self.input_file_loc), str(full_output_path), dataset_name
            )
            self.data.load()

            end_time = time.time()
            self.log_printer.print(
                f"Data loading elapsed time: {end_time - start_time:.2f}s"
            )

            # 清理输出文件
            self._clear_output_files(full_output_path)

            return True

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            self.log_printer.print(f"Error loading data: {e}", color="bold red")
            return False

    def _clear_output_files(self, output_path: Path):
        """清理输出文件"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)

            # 定义需要清理的文件列表
            files_to_clear = {
                "opt_result.csv": [
                    "day",
                    "plant_code",
                    "client_code",
                    "product_code",
                    "vehicle_id",
                    "vehicle_type",
                    "qty",
                ],
                "non_fulfill.csv": ["client_code", "product_code", "volume"],
            }

            # 清理并初始化每个文件
            for file_name, header in files_to_clear.items():
                file_path = output_path / file_name
                with open(file_path, "w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)

        except Exception as e:
            logger.error(f"清理输出文件失败: {e}")

    def create_initial_solution(self) -> Optional[SolutionState]:
        """
        创建初始解

        Returns:
        --------
        SolutionState or None : 初始解, 如果创建失败则返回None
        """
        if self.data is None:
            self.log_printer.print("Error: Data not loaded", color="bold red")
            return None

        try:
            self.log_printer.print("Initializing the solution...")

            # 创建SolutionState的实例
            state = SolutionState(self.data)

            # 初始化参数自动调整器和ML选择器（按运行时配置动态启用）
            use_tuner = self.is_param_tuner_enabled()
            use_ml = self.is_ml_enabled()
            if use_tuner or use_ml:
                self.log_printer.print("Initializing parameter auto-tuner...")
                self.param_tuner = ParamAutoTuner(self.data, self.rng)

                # 设置参数自动调整器引用
                state.set_param_tuner(self.param_tuner)

            if use_ml and self.param_tuner is not None:
                self.log_printer.print("Initializing ML selector...")
                self.ml_selector = MLOperatorSelector(self.param_tuner, self.rng)
                state.set_ml_selector(self.ml_selector)

            # 获取初始解（可选静默内部 print，避免初始解构造阶段刷屏）
            quiet_init_logs = bool(
                getattr(ALNSConfig, "QUIET_INITIAL_SOLUTION_LOGS", True)
            )
            if quiet_init_logs:
                with redirect_stdout(io.StringIO()):
                    init_sol = initial_solution(state, rng=self.rng)
            else:
                init_sol = initial_solution(state, rng=self.rng)

            # 对初始解进行评估
            initial_feasibility, violations = init_sol.validate()

            init_actual_obj = float(init_sol.calculate_cost())
            self.log_printer.print(
                f"Initial Solution Objective (Gurobi-aligned): {init_actual_obj:.2f}"
            )
            self.log_printer.print(
                f"Vehicles Used in Initial Solution: {len(init_sol.vehicles)}"
            )
            self.log_printer.print(f"Initial Solution Cost: {init_actual_obj:.2f}")

            if not initial_feasibility:
                self.log_printer.print(
                    "Warning: The initial solution is infeasible!", color="bold red"
                )
                self._print_violation_details(violations)

                # 根据配置决定是否终止程序
                if getattr(ALNSConfig, "TERMINATE_ON_INFEASIBLE_INITIAL", True):
                    self.log_printer.print(
                        "Terminating due to infeasible initial solution.",
                        color="bold red",
                    )
                    return None
                else:
                    self.log_printer.print(
                        "Continuing with infeasible initial solution.", color="yellow"
                    )

            return init_sol

        except Exception as e:
            logger.error(f"创建初始解失败: {e}")
            self.log_printer.print(
                f"Error creating initial solution: {e}", color="bold red"
            )
            return None

    def _print_violation_details(self, violations: Dict):
        """打印违约详情"""
        if violations.get("negative_inventory"):
            self.log_printer.print("Negative inventory violations:")
            for key, inv in violations["negative_inventory"]:
                self.log_printer.print(
                    f"  Plant: {key[0]}, SKU: {key[1]}, Day: {key[2]}, Inventory: {inv}"
                )

        if violations.get("veh_overload"):
            self.log_printer.print("Vehicle overload violations:")
            for v in violations["veh_overload"]:
                veh = v["veh"]
                self.log_printer.print(
                    f"  Vehicle: Plant {veh.fact_id}, Dealer {veh.dealer_id}, Type {veh.type}, Day {veh.day}, Loaded: {v['loaded']}, Capacity: {v['cap']}"
                )

        if violations.get("unmet_demand"):
            self.log_printer.print("Unmet demand violations:")
            for d in violations["unmet_demand"]:
                self.log_printer.print(
                    f"  Dealer: {d['dealer']}, SKU: {d['sku_id']}, Demand: {d['demand']}, Shipped: {d['shipped']}"
                )

    def setup_alns(self) -> ALNS:
        """
        设置ALNS算法实例

        Returns:
        --------
        ALNS : 配置好的ALNS实例
        """
        try:
            # 创建ALNS的实例
            alns = ALNS(rng=rnd.default_rng(seed=SEED))

            # 注册destroy算子
            self._register_destroy_operators(alns)

            # 注册repair算子
            self._register_repair_operators(alns)

            return alns

        except Exception as e:
            logger.error(f"ALNS设置失败: {e}")
            raise

    def _register_destroy_operators(self, alns: ALNS):
        """注册destroy算子"""
        self.log_printer.print("注册destroy算子...")

        try:
            # 获取基础参数
            destroy_params = ALNSConfig.get_destroy_params()

            # 如果启用了参数自动调整和ML选择器，添加ML增强的破坏算子
            if self.is_ml_enabled() and self.param_tuner and self.ml_selector:
                self.log_printer.print("注册ML增强的破坏算子...")
                ml_destroy_op = self._create_named_partial(
                    ml_based_destroy,
                    name="ml_based_destroy",
                    min_sample_size=getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20),
                )
                alns.add_destroy_operator(ml_destroy_op)
                self._num_registered_destroy_ops = (
                    getattr(self, "_num_registered_destroy_ops", 0) + 1
                )

            # 注册random removal算子（使用 wrapper 工厂以便预设参数）
            random_removal_op = create_random_removal(
                destroy_params["random_removal_degree"]
            )
            alns.add_destroy_operator(random_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # 注册 shaw removal算子（使用 wrapper 工厂）
            shaw_removal_op = create_shaw_removal(destroy_params["shaw_removal_degree"])
            alns.add_destroy_operator(shaw_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # 注册 periodic shaw removal算子（多参数工厂）
            periodic_shaw_removal_op = create_periodic_shaw_removal(
                **destroy_params.get("periodic_shaw_params", {})
            )
            alns.add_destroy_operator(periodic_shaw_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # 其余需要参数的算子使用相应工厂创建并注册（infeasible_removal 不需要额外参数）
            # worst_removal: prefer operator-level default if not provided in DESTROY_DEFAULTS
            worst_defaults = ALNSConfig.get_operator_default("worst_removal").get(
                "params", {}
            )
            worst_removal_op = create_worst_removal(
                destroy_params.get(
                    "worst_removal_degree", worst_defaults.get("degree", 0.25)
                )
            )
            alns.add_destroy_operator(worst_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # surplus_inventory_removal: use operator default fallback
            surplus_defaults = ALNSConfig.get_operator_default(
                "surplus_inventory_removal"
            ).get("params", {})
            surplus_inventory_removal_op = create_surplus_inventory_removal(
                destroy_params.get(
                    "surplus_inventory_removal_degree",
                    surplus_defaults.get("degree", 0.25),
                )
            )
            alns.add_destroy_operator(surplus_inventory_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # path_removal: use operator default fallback
            path_defaults = ALNSConfig.get_operator_default("path_removal").get(
                "params", {}
            )
            path_removal_op = create_path_removal(
                destroy_params.get(
                    "path_removal_degree", path_defaults.get("degree", 0.5)
                )
            )
            alns.add_destroy_operator(path_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # cross_period_dealer_removal: large-jump neighborhood for cross-period reconstruction
            block_defaults = ALNSConfig.get_operator_default(
                "cross_period_dealer_removal"
            ).get("params", {})
            cross_period_dealer_removal_op = create_cross_period_dealer_removal(
                destroy_params.get(
                    "cross_period_dealer_removal_degree",
                    block_defaults.get("degree", 0.45),
                )
            )
            alns.add_destroy_operator(cross_period_dealer_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # period_range_removal: contiguous time-window destruction
            period_range_defaults = ALNSConfig.get_operator_default(
                "period_range_removal"
            ).get("params", {})
            period_range_removal_op = create_period_range_removal(
                destroy_params.get(
                    "period_range_removal_degree",
                    period_range_defaults.get("degree", 0.35),
                ),
                destroy_params.get(
                    "period_range_removal_window_ratio",
                    period_range_defaults.get("window_ratio", 0.30),
                ),
            )
            alns.add_destroy_operator(period_range_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # infeasible_removal 不需要传参，直接包裹注册以保证兼容性
            alns.add_destroy_operator(wrap_operator_no_args(infeasible_removal))
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # inventory_level_destruction: P1 优先级，按经销商库存异常破坏
            inv_defaults = ALNSConfig.get_operator_default(
                "inventory_level_destruction"
            ).get("params", {})
            inventory_level_destruction_op = create_inventory_level_destruction(
                destroy_params.get(
                    "inventory_level_destruction_degree",
                    inv_defaults.get("degree", 0.25),
                )
            )
            alns.add_destroy_operator(inventory_level_destruction_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            self.log_printer.print("所有destroy算子注册完成")

        except Exception as e:
            logger.error(f"注册destroy算子失败: {e}")
            raise

    def _register_repair_operators(self, alns: ALNS):
        """注册repair算子"""
        self.log_printer.print("注册repair算子...")

        try:
            # 如果启用了参数自动调整和ML选择器，添加ML增强的修复算子
            if self.is_ml_enabled() and self.param_tuner and self.ml_selector:
                self.log_printer.print("注册ML增强的修复算子...")
                ml_repair_op = self._create_named_partial(
                    ml_based_repair,
                    name="ml_based_repair",
                    min_sample_size=getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20),
                )
                alns.add_repair_operator(ml_repair_op)
                self._num_registered_repair_ops = (
                    getattr(self, "_num_registered_repair_ops", 0) + 1
                )

            # 获取 repair 参数（通过 centralized accessors）
            repair_params = ALNSConfig.get_repair_params()
            # per-operator fallbacks (read from OPERATOR_DEFAULTS when available)
            local_defaults = ALNSConfig.get_operator_default("local_search_repair").get(
                "params", {}
            )
            smart_defaults = ALNSConfig.get_operator_default("smart_batch_repair").get(
                "params", {}
            )
            regret_defaults = ALNSConfig.get_operator_default(
                "regret_based_repair"
            ).get("params", {})
            lookahead_defaults = ALNSConfig.get_operator_default(
                "lookahead_repair"
            ).get("params", {})

            # 注册 repair 算子，使用 wrapper / factory 以便预设参数并保证 __name__ 可读
            alns.add_repair_operator(wrap_repair_no_args(greedy_repair))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            local_iter = repair_params.get("local_search_repair", {}).get(
                "max_iter", local_defaults.get("max_iter", 10)
            )
            alns.add_repair_operator(create_local_search_repair(local_iter))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            alns.add_repair_operator(wrap_repair_no_args(inventory_balance_repair))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            alns.add_repair_operator(wrap_repair_no_args(infeasible_repair))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            smart_iter = repair_params.get("smart_batch_repair", {}).get(
                "max_iter", smart_defaults.get("max_iter", 10)
            )
            smart_batch_size = repair_params.get("smart_batch_repair", {}).get(
                "batch_size", smart_defaults.get("batch_size", 10)
            )
            smart_timeout = repair_params.get("smart_batch_repair", {}).get(
                "timeout", smart_defaults.get("timeout", None)
            )
            alns.add_repair_operator(
                create_smart_batch_repair(
                    smart_iter, batch_size=smart_batch_size, timeout=smart_timeout
                )
            )
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            regret_k = repair_params.get("regret_based_repair", {}).get(
                "k", regret_defaults.get("k", 2)
            )
            regret_topN = repair_params.get("regret_based_repair", {}).get(
                "topN", regret_defaults.get("topN", 6)
            )
            regret_time = repair_params.get("regret_based_repair", {}).get(
                "time_limit", regret_defaults.get("time_limit", 10.0)
            )
            alns.add_repair_operator(
                wrap_repair_no_args(
                    create_regret_repair(regret_k, regret_topN, regret_time)
                )
            )
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            # 注册专门的车队整合修复算子，用于减少碎片化的车辆分配
            alns.add_repair_operator(FleetConsolidationRepairOperator())
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            # 注册顺序修复算子 (P0 优先级)
            alns.add_repair_operator(create_sequential_repair())
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            # 注册前瞻修复算子（停滞期跳盆地用，默认 h=2）
            lookahead_h = repair_params.get("lookahead_repair", {}).get(
                "lookahead_horizon",
                lookahead_defaults.get("lookahead_horizon", 2),
            )
            alns.add_repair_operator(create_lookahead_repair(lookahead_h))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            self.log_printer.print("所有repair算子注册完成")

        except Exception as e:
            logger.error(f"注册repair算子失败: {e}")
            raise

    def _create_named_partial(self, func, name=None, **kwargs):
        """
        创建带有__name__属性的partial对象, 以兼容ALNS库

        Parameters:
        -----------
        func : callable
            要创建partial的函数
        name : str, optional
            partial对象的名称, 如果不提供则自动生成
        **kwargs : dict
            要固定的关键字参数

        Returns:
        --------
        partial object with __name__ attribute
        """

        class _NamedCallable:
            def __init__(self, wrapped, callable_name: str, fixed_kwargs: Dict):
                self._wrapped = wrapped
                self._fixed_kwargs = dict(fixed_kwargs)
                self.__name__ = callable_name

            def __call__(self, state, rng, **call_kwargs):
                merged = dict(self._fixed_kwargs)
                merged.update(call_kwargs or {})
                return self._wrapped(state, rng, **merged)

        if name is None:
            param_str = "_".join([f"{k}_{v}" for k, v in kwargs.items()])
            name = f"{func.__name__}_{param_str}"

        return _NamedCallable(func, name, kwargs)

    def run_optimization(self, dataset_name: str) -> bool:
        """
        运行完整的优化流程

        Parameters:
        -----------
        dataset_name : str
            数据集名称

        Returns:
        --------
        bool : 是否优化成功
        """
        try:
            self.log_printer.print(f"开始优化流程! / {datetime.datetime.now()}")
            optimization_start_time = time.time()

            # 1. 加载数据
            t0 = time.time()
            self.log_printer.print("Step 1: 加载数据...")
            if not self.load_data(dataset_name):
                return False

            # Optional startup plotting (guarded by config to avoid heavy startup cost).
            # 供应链3D图默认禁用，需同时打开 ENABLE_STARTUP_PLOTTING 与 ENABLE_SUPPLY_CHAIN_3D。
            if getattr(ALNSConfig, "ENABLE_STARTUP_PLOTTING", False) and getattr(
                ALNSConfig, "ENABLE_SUPPLY_CHAIN_3D", False
            ):
                try:
                    # delayed import to avoid importing matplotlib unless requested
                    from .visualization import (
                        plot_supply_chain_network_3d_enhanced as _plot_startup,
                    )

                    if self.data is not None:
                        _plot_startup(
                            self.data, str(self.output_file_loc / dataset_name)
                        )
                except Exception as e:
                    logger.debug(f"Startup plotting skipped: {e}")

            self.log_printer.print(f"Step 1耗时: {time.time() - t0:.2f}s")

            # 2. 创建初始解
            t0 = time.time()
            self.log_printer.print("Step 2: 创建初始解...")
            init_sol = self.create_initial_solution()
            if init_sol is None:
                return False
            self.log_printer.print(f"Step 2耗时: {time.time() - t0:.2f}s")

            # 3. 设置ALNS算法
            t0 = time.time()
            self.log_printer.print("Step 3: 设置ALNS算法...")
            alns = self.setup_alns()
            self.log_printer.print(f"Step 3耗时: {time.time() - t0:.2f}s")

            # 4. 配置算子选择机制
            t0 = time.time()
            self.log_printer.print("Step 4: 配置算子选择机制...")
            # 动态使用已注册的算子数量来配置选择器，避免硬编码
            num_destroy = getattr(self, "_num_registered_destroy_ops", 7) or 7
            num_repair = getattr(self, "_num_registered_repair_ops", 6) or 6
            select = SegmentedRouletteWheel(
                scores=getattr(ALNSConfig, "ROULETTE_SCORES", [1.0]),
                decay=getattr(ALNSConfig, "ROULETTE_DECAY", 0.9),
                seg_length=getattr(ALNSConfig, "ROULETTE_SEG_LENGTH", 100),
                num_destroy=num_destroy,
                num_repair=num_repair,
            )
            self.log_printer.print(f"Step 4耗时: {time.time() - t0:.2f}s")

            # 5. 配置接受准则
            t0 = time.time()
            self.log_printer.print("Step 5: 配置接受准则...")
            sa_accept = SimulatedAnnealing(
                start_temperature=getattr(ALNSConfig, "SA_START_TEMP", 1.0),
                end_temperature=getattr(ALNSConfig, "SA_END_TEMP", 1e-3),
                step=getattr(ALNSConfig, "SA_STEP", 0.99),
                method="exponential",
            )
            self.acceptance = DemandConstraintAccept(sa_accept, self.data)
            self.log_printer.print(f"Step 5耗时: {time.time() - t0:.2f}s")

            # 6. 配置停止准则
            t0 = time.time()
            self.log_printer.print("Step 6: 配置停止准则...")

            # 使用组合停止准则：
            # 小规模数据集: 300次迭代 OR 900秒运行时间
            # 中规模数据集: 600次迭代 OR 1800秒运行时间
            # 大规模数据集: 1000次迭代 OR 3600秒运行时间
            stop = create_standard_combined_criterion(
                max_iterations=getattr(ALNSConfig, "MAX_ITERATIONS", 300),  # 300次迭代
                max_runtime=getattr(ALNSConfig, "MAX_RUNTIME", 900),  # 900秒
                max_no_improvement=None,  # 暂不使用无改进停止条件
            )

            # 将迭代上限初始化为一个更保守的高值，避免过早被 MaxIterations 截断。
            try:
                runtime_budget = float(getattr(ALNSConfig, "MAX_RUNTIME", 900))
                base_iter_cap = int(getattr(ALNSConfig, "MAX_ITERATIONS", 300))
                iter_per_second = float(
                    getattr(ALNSConfig, "DYNAMIC_MAX_ITERATIONS_PER_SECOND", 1000.0)
                )
                safety_factor = float(
                    getattr(ALNSConfig, "DYNAMIC_MAX_ITERATION_SAFETY_FACTOR", 1.5)
                )
                buffer_iters = int(
                    getattr(ALNSConfig, "DYNAMIC_MAX_ITERATION_BUFFER", 200)
                )
                dynamic_floor = int(
                    max(
                        base_iter_cap,
                        runtime_budget
                        * max(iter_per_second, 1.0)
                        * max(safety_factor, 1.0),
                    )
                ) + max(0, buffer_iters)
                if hasattr(stop, "set_max_iterations"):
                    stop.set_max_iterations(dynamic_floor)

                # Keep optimizer/tracker iteration horizon aligned with the effective
                # stopping configuration so progress-based policies remain consistent.
                try:
                    self.set_max_iterations(dynamic_floor)
                except Exception:
                    pass
                try:
                    if self.tracker is not None and hasattr(
                        self.tracker, "set_max_iterations"
                    ):
                        self.tracker.set_max_iterations(dynamic_floor)
                except Exception:
                    pass
            except Exception:
                pass

            self.log_printer.print(
                f"组合停止准则配置: 迭代上限将动态扩容, 运行时间预算{getattr(ALNSConfig, 'MAX_RUNTIME', 900)}秒"
            )
            self.log_printer.print(f"Step 6耗时: {time.time() - t0:.2f}s")

            # 7. 设置追踪器和回调函数
            t0 = time.time()
            self.log_printer.print("Step 7: 设置追踪器和回调函数...")
            self.tracker = ALNSTracker(log_printer=self.log_printer)
            # 新增: 将tracker注入到初始解中, 以便在repair算子中使用
            init_sol.set_tracker(self.tracker)
            try:
                self.tracker.set_initial_reference(
                    initial_search_obj=float(init_sol.objective()),
                    initial_actual_obj=float(init_sol.calculate_cost()),
                )
            except Exception:
                pass

            # 将初始严格可行解注册为可回滚基线，避免早期候选不可行时无基线可回滚。
            try:
                init_feasible, _ = init_sol.validate()
                if init_feasible:
                    self.best_feasible_state = safe_state_snapshot(init_sol)
                    self.best_feasible_obj = float(init_sol.objective())
            except Exception:
                pass

            register_callbacks(alns, self, stop=stop)
            self.log_printer.print(f"Step 7耗时: {time.time() - t0:.2f}s")

            # 8. 运行ALNS算法
            t0 = time.time()
            self.log_printer.print("Step 8: 运行ALNS算法...")
            try:
                quiet_operator_logs = bool(
                    getattr(ALNSConfig, "QUIET_OPERATOR_LOGS", True)
                )
                with suppress_noisy_operator_stdout(enabled=quiet_operator_logs):
                    self.result = alns.iterate(
                        init_sol, select, cast(Any, self.acceptance), stop
                    )
                self.log_printer.print(f"Step 8耗时: {time.time() - t0:.2f}s")
            except Exception as e:
                # 捕获并记录完整回溯以便定位 None 与 float 比较导致的异常
                import traceback

                tb = traceback.format_exc()
                # print to stdout as well as logger in case logger not configured
                try:
                    print("ALNS iterate exception traceback:\\n", tb)
                except Exception:
                    pass
                logger.error("ALNS iterate failed: %s", e)
                logger.debug(tb)
                self.log_printer.print(f"ALNS iterate failed: {e}", color="bold red")
                return False

            optimization_end_time = time.time()
            self.log_printer.print(
                f"优化总耗时: {optimization_end_time - optimization_start_time:.2f}s"
            )

            # 9. 处理结果
            t0 = time.time()
            self.log_printer.print("Step 9: 处理结果...")
            self._process_results()
            self.log_printer.print(f"Step 9耗时: {time.time() - t0:.2f}s")

            # 10. 生成报告和图表
            t0 = time.time()
            self.log_printer.print("Step 10: 生成报告和图表...")
            self._generate_reports()
            self.log_printer.print(f"Step 10耗时: {time.time() - t0:.2f}s")

            self.log_printer.print_title(f"优化完成! {datetime.datetime.now()}")

            return True

        except Exception as e:
            logger.error(f"优化过程失败: {e}")
            self.log_printer.print(f"Optimization failed: {e}", color="bold red")
            return False

    def _process_results(self):
        """处理优化结果"""
        if self.result is None:
            return

        try:
            # 获取最终解
            self.best_solution = safe_state_snapshot(self.result.best_state)

            # 可选：按实际目标函数选取追踪到的最佳可行解作为最终输出口径
            tracker_stats = self.tracker.get_statistics()
            if bool(getattr(ALNSConfig, "PREFER_FINAL_BEST_ACTUAL_SOLUTION", True)):
                candidate_actual_best = tracker_stats.get("best_actual_solution")
                if candidate_actual_best is not None:
                    self.best_solution = safe_state_snapshot(candidate_actual_best)

            # 回调路径维护的“最优完全可行解”优先级最高，避免最终导出退化为非严格可行状态。
            callback_best_feasible = getattr(self, "best_feasible_state", None)
            if callback_best_feasible is not None:
                self.best_solution = safe_state_snapshot(callback_best_feasible)

            # 输出前对最优解做一次轻量并车压缩（若不可行则自动回退）
            if self.best_solution is not None and getattr(
                ALNSConfig, "FINAL_BEST_ENABLE_CONSOLIDATION", True
            ):
                before_cnt = len(self.best_solution.vehicles)
                try:
                    max_pairs = int(
                        getattr(ALNSConfig, "FINAL_BEST_CONSOLIDATION_MAX_PAIRS", 800)
                    )
                except Exception:
                    max_pairs = 800

                snapshot_before = safe_state_snapshot(self.best_solution)
                merged_cnt = 0
                try:
                    merged_cnt = consolidate_route_vehicles(
                        self.best_solution, max_pairs_per_group=max(1, max_pairs)
                    )
                except Exception:
                    merged_cnt = 0

                if merged_cnt > 0:
                    try:
                        feasible_after, _ = self.best_solution.validate()
                    except Exception:
                        feasible_after = False
                    if not feasible_after and snapshot_before is not None:
                        self.best_solution = snapshot_before
                        self.log_printer.print(
                            "Final consolidation produced infeasible state; rolled back.",
                            color="yellow",
                        )
                    else:
                        after_cnt = len(self.best_solution.vehicles)
                        self.log_printer.print(
                            f"Final consolidation merged {merged_cnt} vehicles ({before_cnt} -> {after_cnt})."
                        )

            # 输出统计信息
            self.log_printer.print(
                f"Total iterations tracked: {tracker_stats['total_iterations']}"
            )
            self.log_printer.print(
                f"Best objective found: {tracker_stats['best_objective']:.4f}"
            )
            self.log_printer.print(
                f"Best actual objective found: {tracker_stats.get('best_actual_objective', float('inf')):.4f}"
            )
            self.log_printer.print(
                f"Final Search Gap vs Initial: {tracker_stats['final_gap']:+.2f}%"
            )
            self.log_printer.print(
                f"Final Actual Gap vs Initial: {tracker_stats['final_actual_gap']:+.2f}%"
            )

            # 验证最终解
            final_feasibility, violations = self.best_solution.validate()
            if not final_feasibility:
                self.log_printer.print(
                    "Warning: The final best solution is infeasible!", color="bold red"
                )

            best_search_obj = float(self.best_solution.objective())
            best_actual_obj = float(self.best_solution.calculate_cost())
            self.log_printer.print(
                f"Best Search Objective (ALNS-guiding): {best_search_obj:.6f}"
            )
            self.log_printer.print(
                f"Best Actual Objective (Gurobi-aligned): {best_actual_obj:.2f}"
            )
            self.log_printer.print(
                f"Vehicles Used in Best Solution: {len(self.best_solution.vehicles)}"
            )
            self.log_printer.print(f"Best Solution Cost: {best_actual_obj:.2f}")

            # 更新数据和输出结果
            self._update_demands()
            self._update_inventory()
            self._output_results()
            self._generate_statistics()
            self._write_repair_audit_report()

        except Exception as e:
            logger.error(f"处理结果失败: {e}")
            raise

    def _write_repair_audit_report(self):
        """Write per-repair-operator feasibility audit summary to CSV."""
        try:
            if self.best_solution is None:
                return
            audit = getattr(self.best_solution, "repair_audit", None)
            if not audit:
                return

            rows = []
            for op_name, stats in audit.items():
                attempts = int(stats.get("attempts", 0))
                feasible = int(stats.get("feasible", 0))
                infeasible = int(stats.get("infeasible", 0))
                rolled_back = int(stats.get("rolled_back", 0))
                rows.append(
                    {
                        "operator": op_name,
                        "attempts": attempts,
                        "feasible": feasible,
                        "infeasible": infeasible,
                        "feasible_rate": (float(feasible) / attempts)
                        if attempts > 0
                        else 0.0,
                        "rolled_back": rolled_back,
                        "inserted_total": int(stats.get("inserted_total", 0)),
                        "improvement_total": float(stats.get("improvement_total", 0.0)),
                        "fail_negative_inventory": int(
                            stats.get("fail_negative_inventory", 0)
                        ),
                        "fail_veh_over_load": int(stats.get("fail_veh_over_load", 0)),
                        "fail_plant_inv_exceed": int(
                            stats.get("fail_plant_inv_exceed", 0)
                        ),
                        "fail_unmet_demand": int(stats.get("fail_unmet_demand", 0)),
                    }
                )

            if not rows:
                return

            df = pd.DataFrame(rows).sort_values(
                by=["attempts", "operator"], ascending=[False, True]
            )
            out = Path(self.data.output_file_loc) / "repair_feasibility_audit.csv"
            df.to_csv(out, index=False)
            self.log_printer.print(f"Repair feasibility audit saved: {out}")
        except Exception as e:
            logger.error(f"写入修复算子审计报告失败: {e}")

    def _update_demands(self):
        """更新需求数据"""
        try:
            # 更新需求
            for veh in self.best_solution.vehicles:
                for (sku_id, day), qty in veh.cargo.items():
                    key = (veh.dealer_id, sku_id)
                    if key in self.data.demands:
                        self.data.demands[key] -= qty

            # 获取未满足的需求数据
            non_fulfill_data = []
            for (dealer, sku_id), qty in self.data.demands.items():
                if qty > 0:
                    non_fulfill_data.append(
                        {"client_code": dealer, "product_code": sku_id, "volume": qty}
                    )

            # 保存未满足需求
            if non_fulfill_data:
                df_non_fulfill = pd.DataFrame(non_fulfill_data)
                output_file = Path(self.data.output_file_loc) / "non_fulfill.csv"
                # 检查文件是否被占用
                if is_file_in_use(output_file):
                    self.log_printer.print(
                        f"文件 {output_file} 正在被占用, 请关闭后重试。",
                        color="bold red",
                    )
                    return
                df_non_fulfill.to_csv(output_file, mode="a", header=False, index=False)

            # 更新数据对象中的需求
            self.data.demands = {
                (dealer, sku_id): qty
                for (dealer, sku_id), qty in self.data.demands.items()
                if qty > 0
            }

        except Exception as e:
            logger.error(f"更新需求失败: {e}")
            raise

    def _update_inventory(self):
        """更新库存数据"""
        try:
            sku_inv_left = {}

            # 遍历每个工厂的每种SKU
            for plant_id in self.data.plants:
                for sku_id in self.data.all_skus:
                    # 获取期初库存
                    current_inv = self.data.sku_initial_inv.get((plant_id, sku_id), 0)

                    # 按时间顺序计算每天结束时的库存
                    for day in range(1, self.data.horizons + 1):
                        # 加上当天生产量
                        current_inv += self.data.sku_prod_each_day.get(
                            (plant_id, sku_id, day), 0
                        )

                        # 减去当天配送量
                        shipped_qty = 0
                        for veh in self.best_solution.vehicles:
                            if veh.fact_id == plant_id and (sku_id, day) in veh.cargo:
                                shipped_qty += veh.cargo[(sku_id, day)]
                        current_inv -= shipped_qty

                        # 保存当天结束时的库存
                        sku_inv_left[(plant_id, sku_id, day)] = current_inv

            # 创建并保存库存DataFrame
            df_sku_inv_left = pd.DataFrame(
                [
                    {
                        "plant_code": plant,
                        "product_code": sku_id,
                        "day": day,
                        "volume": qty,
                    }
                    for (plant, sku_id, day), qty in sku_inv_left.items()
                ]
            )
            output_file = Path(self.data.output_file_loc) / "sku_inv_left.csv"
            if is_file_in_use(output_file):
                self.log_printer.print(
                    f"文件 {output_file} 正在被占用, 请关闭后重试。", color="bold red"
                )
                return
            df_sku_inv_left.to_csv(output_file, index=False)

        except Exception as e:
            logger.error(f"更新库存失败: {e}")
            raise

    def _output_results(self):
        """输出结果到文件"""
        try:
            output_file = Path(self.data.output_file_loc) / "opt_result.csv"
            df_result = self._create_result_dataframe()
            if is_file_in_use(output_file):
                self.log_printer.print(
                    f"文件 {output_file} 正在被占用, 请关闭后重试。", color="bold red"
                )
                return
            df_result.to_csv(output_file, mode="a", header=False, index=False)

        except Exception as e:
            logger.error(f"输出结果失败: {e}")
            raise

    def _create_result_dataframe(self) -> pd.DataFrame:
        """创建结果DataFrame"""
        rows = []
        for idx, veh in enumerate(self.best_solution.vehicles):
            for (sku_id, day), qty in veh.cargo.items():
                rows.append(
                    {
                        "day": day,
                        "plant_code": veh.fact_id,
                        "client_code": veh.dealer_id,
                        "product_code": sku_id,
                        "vehicle_id": idx + 1,
                        "vehicle_type": veh.type,
                        "qty": qty,
                    }
                )
        return pd.DataFrame(rows)

    def _generate_statistics(self):
        """生成统计信息"""
        try:
            dataset = self.data.dataset_name
            output_location = Path(self.data.output_file_loc)
            input_location = self.input_file_loc / dataset

            # 读取结果文件
            df_result = pd.read_csv(output_location / "opt_result.csv")
            df_sku_size = pd.read_csv(input_location / "product_size.csv")
            df_all_veh = pd.read_csv(input_location / "vehicle.csv")

            # 合并数据前, 确保主键列类型一致（全部转为str）
            if "product_code" in df_result.columns:
                df_result["product_code"] = df_result["product_code"].astype(str)
            if "product_code" in df_sku_size.columns:
                df_sku_size["product_code"] = df_sku_size["product_code"].astype(str)
            if "vehicle_type" in df_result.columns:
                df_result["vehicle_type"] = df_result["vehicle_type"].astype(str)
            if "vehicle_type" in df_all_veh.columns:
                df_all_veh["vehicle_type"] = df_all_veh["vehicle_type"].astype(str)
            df_result = pd.merge(df_result, df_sku_size, on=["product_code"])
            df_result = pd.merge(df_result, df_all_veh, on=["vehicle_type"])

            # 删除不需要的列
            if "cost_to_use" in df_result.columns:
                del df_result["cost_to_use"]

            # 计算占用体积
            df_result["occupied_size"] = df_result["qty"] * df_result["standard_size"]

            # 保存详细结果
            details_file = output_location / "opt_details.csv"
            if is_file_in_use(details_file):
                self.log_printer.print(
                    f"文件 {details_file} 正在被占用, 请关闭后重试。", color="bold red"
                )
                return
            df_result.to_csv(details_file, index=False)

            # 生成汇总统计
            df_stat1 = df_result.copy()
            if "product_code" in df_stat1.columns:
                df_stat1 = df_stat1.drop(["product_code", "standard_size"], axis=1)

            df_summary = df_stat1.groupby(
                ["day", "plant_code", "client_code", "vehicle_id"]
            ).agg(
                {
                    "vehicle_type": "first",
                    "qty": "sum",
                    "carry_standard_size": "first",
                    "min_standard_size": "first",
                    "occupied_size": "sum",
                }
            )

            # 重置索引
            df_summary = df_summary.reset_index()

            summary_file = output_location / "opt_summary.csv"
            if is_file_in_use(summary_file):
                self.log_printer.print(
                    f"文件 {summary_file} 正在被占用, 请关闭后重试。", color="bold red"
                )
                return
            df_summary.to_csv(summary_file, index=False)

            # 验证容量约束
            self._validate_capacity_constraints(df_summary)

        except Exception as e:
            logger.error(f"生成统计信息失败: {e}")
            raise

    def _validate_capacity_constraints(self, df_summary: pd.DataFrame):
        """验证容量约束"""
        try:
            # 检查是否所有车辆装载的SKU体积满足车辆容量约束
            is_cap_exceed = pd.Series(
                df_summary["occupied_size"] <= df_summary["carry_standard_size"]
            )

            if is_cap_exceed.all():
                self.log_printer.print("✓ 所有车辆的装载体积都满足容量约束")
            else:
                self.log_printer.print(
                    "⚠️ 部分车辆的装载体积超出容量限制", color="yellow"
                )
                df_exceed = df_summary[
                    df_summary["occupied_size"] > df_summary["carry_standard_size"]
                ]

                df_exceed_info = df_exceed.loc[
                    :,
                    [
                        "vehicle_id",
                        "vehicle_type",
                        "carry_standard_size",
                        "occupied_size",
                    ],
                ]
                df_exceed_info["exceeded_volume"] = (
                    df_exceed_info["occupied_size"]
                    - df_exceed_info["carry_standard_size"]
                )

                output_path = Path(self.data.output_file_loc) / "extra_volume.csv"
                if is_file_in_use(output_path):
                    self.log_printer.print(
                        f"文件 {output_path} 正在被占用, 请关闭后重试。",
                        color="bold red",
                    )
                    return
                df_exceed_info.to_csv(output_path, index=False)
                self.log_printer.print(f"超容量信息已保存到: {output_path}")

        except Exception as e:
            logger.error(f"验证容量约束失败: {e}")

    def _generate_reports(self):
        """生成报告和图表"""
        try:
            # 支持通过配置开关在性能分析/CI场景下禁用所有报告与绘图（避免 matplotlib 的文本布局开销）
            if not getattr(ALNSConfig, "ENABLE_REPORTS", True):
                try:
                    self.log_printer.print(
                        "报告生成已禁用 (ENABLE_REPORTS=False)，跳过绘图。"
                    )
                except Exception:
                    pass
                return
            if self.result is None or self.best_solution is None:
                return
            # delayed import of visualization tools to avoid importing matplotlib at module import time
            try:
                from .visualization import (
                    plot_objective_changes,
                    plot_operator_performance,
                    plot_gap_changes,
                )
            except Exception as e:
                logger.debug(f"Visualization tools unavailable: {e}")
                return
            # 创建图像目录
            images_dir = Path(self.data.output_file_loc) / "images"
            images_dir.mkdir(exist_ok=True)
            # 绘制目标函数变化（引导目标 + 实际目标）
            plot_objective_changes(
                self.result,
                str(self.data.output_file_loc),
                tracker=self.tracker,
            )
            # 绘制算子性能（调用可视化模块）
            plot_operator_performance(self.best_solution, self.result)
            # 绘制Gap变化
            plot_gap_changes(self.tracker, str(self.data.output_file_loc))

            # 生成参数自动调整和ML选择器报告
            if self.param_tuner:
                self._generate_param_tuning_report()

        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            self.log_printer.print(
                f"Warning: Failed to generate reports: {e}", color="yellow"
            )

    def _plot_operator_performance(self):
        """绘制算子性能（简化版本）"""
        try:
            import matplotlib.pyplot as plt
            # 这里实现一个简化的算子性能图表
            # 原始版本过于复杂, 简化为基本的柱状图

            destroy_counts = self.result.statistics.destroy_operator_counts
            repair_counts = self.result.statistics.repair_operator_counts

            # 创建简化的性能图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Destroy operators
            if destroy_counts:
                operators = list(destroy_counts.keys())
                totals = [
                    sum(counts[:4]) for counts in destroy_counts.values()
                ]  # 前4个结果的总和
                ax1.bar(operators, totals, color="skyblue")
                ax1.set_title("Destroy Operators Performance")
                ax1.set_ylabel("Total Usage")
                ax1.tick_params(axis="x", rotation=45)

            # Repair operators
            if repair_counts:
                operators = list(repair_counts.keys())
                totals = [sum(counts[:4]) for counts in repair_counts.values()]
                ax2.bar(operators, totals, color="lightcoral")
                ax2.set_title("Repair Operators Performance")
                ax2.set_ylabel("Total Usage")
                ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            file_path = (
                Path(self.data.output_file_loc)
                / "images"
                / "Performance_of_Operators.svg"
            )
            dpi_value = getattr(ALNSConfig, "REPORT_DPI", 150)
            plt.savefig(file_path, dpi=dpi_value, bbox_inches="tight")
            plt.close()

        except Exception as e:
            logger.error(f"绘制算子性能失败: {e}")

    def _generate_param_tuning_report(self):
        """生成参数自动调整和ML选择器的报告"""
        if not self.param_tuner:
            return

        try:
            self.log_printer.print("生成参数自动调整和ML选择器报告...")

            # 创建报告目录
            report_dir = self.output_file_loc / "param_tuning_report"
            report_dir.mkdir(parents=True, exist_ok=True)

            # 生成参数自动调整报告
            param_report_path = report_dir / "param_tuning_report.csv"
            with open(param_report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Operator",
                        "Parameter",
                        "Final Value",
                        "Range",
                        "Usage Count",
                        "Success Rate",
                        "Avg Performance",
                    ]
                )

                for op_name, op_params in self.param_tuner.operators.items():
                    for param_name, param_value in op_params.params.items():
                        if param_name in op_params.ranges:
                            param_range = f"{op_params.ranges[param_name][0]} - {op_params.ranges[param_name][1]}"
                        else:
                            param_range = "N/A"

                        writer.writerow(
                            [
                                op_name,
                                param_name,
                                param_value,
                                param_range,
                                op_params.usage_count,
                                f"{op_params.get_success_rate():.4f}",
                                f"{op_params.get_avg_performance():.4f}",
                            ]
                        )

            self.log_printer.print(f"参数自动调整报告已保存到: {param_report_path}")

            # 如果ML选择器可用，生成ML选择器报告
            if self.ml_selector:
                # 生成破坏算子性能报告
                destroy_report_path = report_dir / "destroy_operators_performance.csv"
                with open(destroy_report_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Operator", "Usage Count", "Average Performance"])

                    # 统计各算子使用情况
                    destroy_op_stats = {}
                    for op_name in self.param_tuner.operators:
                        if "removal" in op_name:
                            destroy_op_stats[op_name] = {"count": 0, "total_perf": 0.0}

                    # 从ML选择器中获取性能数据
                    for i, op_encoding in enumerate(self.ml_selector.destroy_features):
                        if len(op_encoding) > 0:
                            # 最后几个元素是算子的one-hot编码
                            op_idx = np.argmax(
                                op_encoding[-len(self.ml_selector.destroy_op_map) :]
                            )
                            op_name = list(self.ml_selector.destroy_op_map.keys())[
                                op_idx
                            ]
                            perf = self.ml_selector.destroy_performances[i]

                            if op_name in destroy_op_stats:
                                destroy_op_stats[op_name]["count"] += 1
                                destroy_op_stats[op_name]["total_perf"] += perf

                    # 写入报告
                    for op_name, stats in destroy_op_stats.items():
                        avg_perf = (
                            stats["total_perf"] / stats["count"]
                            if stats["count"] > 0
                            else 0
                        )
                        writer.writerow([op_name, stats["count"], f"{avg_perf:.4f}"])

                self.log_printer.print(
                    f"破坏算子性能报告已保存到: {destroy_report_path}"
                )

                # 生成修复算子性能报告
                repair_report_path = report_dir / "repair_operators_performance.csv"
                with open(repair_report_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Operator", "Usage Count", "Average Performance"])

                    # 统计各算子使用情况
                    repair_op_stats = {}
                    for op_name in self.param_tuner.operators:
                        if "repair" in op_name:
                            repair_op_stats[op_name] = {"count": 0, "total_perf": 0.0}

                    # 从ML选择器中获取性能数据
                    for i, op_encoding in enumerate(self.ml_selector.repair_features):
                        if len(op_encoding) > 0:
                            # 最后几个元素是算子的one-hot编码
                            op_idx = np.argmax(
                                op_encoding[-len(self.ml_selector.repair_op_map) :]
                            )
                            op_name = list(self.ml_selector.repair_op_map.keys())[
                                op_idx
                            ]
                            perf = self.ml_selector.repair_performances[i]

                            if op_name in repair_op_stats:
                                repair_op_stats[op_name]["count"] += 1
                                repair_op_stats[op_name]["total_perf"] += perf

                    # 写入报告
                    for op_name, stats in repair_op_stats.items():
                        avg_perf = (
                            stats["total_perf"] / stats["count"]
                            if stats["count"] > 0
                            else 0
                        )
                        writer.writerow([op_name, stats["count"], f"{avg_perf:.4f}"])

                self.log_printer.print(
                    f"修复算子性能报告已保存到: {repair_report_path}"
                )

        except Exception as e:
            logger.error(f"生成参数自动调整和ML选择器报告失败: {e}")
            self.log_printer.print(
                f"生成参数自动调整和ML选择器报告失败: {e}", color="bold red"
            )
