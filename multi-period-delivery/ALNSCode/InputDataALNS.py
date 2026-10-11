"""
模块定位与设计意图:
- 说明: 本模块负责 ALNS 求解器所需的静态输入数据管理, 包括从数据文件读取、标准化与校验, 计算派生映射与摘要指标, 并导出供算法各组件(初始解生成、destroy/repair 算子、库存演化、目标评估與 ML 选择器)使用的统一数据容器 DataALNS。
- 目标: 将 IO 与数据清洗逻辑与算法逻辑严格分离, 提供确定性且类型明确的轻量查询接口, 以减少下游模块重复计算与数据不一致风险。

核心设计要点:
- 结构化流程: 采用阶段化处理流程(参数加载 -> 文件路径构造 -> 批量 DataFrame 加载 -> 标准化转换 -> 计算派生数据 -> 数据完整性校验), 每一步单独封装并易于单元测试。
- 类型与键规范化: 在早期阶段将所有标识类字段(plant, dealer, sku, vehicle_type)统一转换为 str, 以避免后续键不匹配导致的隐蔽错误。
- 派生缓存: 计算密集或重复使用的派生映射(例如 supply_chain)采用实例级缓存, 若需程序化修改底层数据, 请显式清理缓存属性。
- 轻量运行时依赖: 在模块导入阶段避免加载 heavy 依赖或业务逻辑, 例如避免导入会改变 SolutionState 的函数。需要时在函数体内做延迟导入以避免循环依赖與启动开销。

对外接口与使用说明:
- Class DataALNS(input_file_loc, output_file_loc, dataset_name):
  - load(): 执行完整的数据加载與处理流水线, 成功后实例将包含算法所需的所有映射與集合。
  - 主要查询方法: available_skus_in_plant(plant_id), plan_dealer_skus(plant_id), available_skus_to_dealer(plant_id, dealer_id), construct_supply_chain(), get_total_demand_for_sku(sku_id), get_total_production_for_sku(sku_id, day=None), get_total_initial_inventory_for_sku(sku_id), get_sku_supply_demand_balance(sku_id), get_summary_statistics().
  - 生命周期说明: 在访问派生字段前, 必须调用 load() 以保证字段已初始化。

实现细节與注意事项:
- 数据验证: _load_csv_safely 在缺失必需列时会抛异常並记录错误, 设计为快速失败以保证下游一致性。可在更高层捕获以实现更复杂的回退策略。
- DataFrame -> 原生结构: 将 pandas DataFrame 的中间结果转换为纯 Python 类型(dict, set)以降低热路径对 pandas 的依赖, 提高算法运行性能。
- 缓存失效: 对于程序性修改底层表或增量加载场景, 请通过删除或设置 self._supply_chain_cache = None 的方式显式使缓存失效。
- 可扩展性: 若数据模式复杂化, 建议引入 schema 校验库(pydantic 或 jsonschema)並将字段映射配置化。

维护建议:
- 当新增源文件或变更 CSV 列名时, 同步更新 _load_all_dataframes 的 required_columns 與 _process_dataframes 的 title_map/type_mapping。
- 为边界情况添加单元测试, 包括空表、缺失 SKU 大小等场景。
- 在大规模实验中, 若内存成为瓶颈, 考虑将大型表以流式或分块方式处理, 避免一次性载入所有数据。

Import 与性能约定:
- 本文件采用三段式导入顺序: 标准库 -> 第三方 -> 项目内部。
- 重型或有循环依赖风险的导入应延迟到函数体内部。

说明: 已保留原实现逻辑, 仅整理并统一模块级注释與 import 分组, 以便团队成员理解与维护。
"""

# =========================
# Standard library
# =========================
import os
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Union, Optional

# =========================
# Third-party libraries
# =========================
import pandas as pd

# =========================
# Public API
# =========================
__all__ = ["DataALNS"]


@dataclass
class DataALNS:
    """
    DataALNS
    --------
    ALNS 求解主程序统一依赖的数据容器。封装原始数据读取、转换、派生及校验过程。
    """
    input_file_loc: str
    output_file_loc: str
    dataset_name: str  # 指定数据集目录名 (子目录)

    # 规划周期: 默认 1 (加载后若生产计划包含更大 day 将被更新)
    horizons: int = 1
    # 当前起始日 (保留字段，若需滚动规划可扩展)
    day: int = 1

    # ==== 惩罚参数 (从配置文件加载) ====
    param_pun_factor1: float = field(init=False)
    param_pun_factor2: float = field(init=False)
    param_pun_factor3: float = field(init=False)
    param_pun_factor4: float = field(init=False)
    param_pun_objective: float = field(init=False)

    # ==== 基础集合 / 属性 ====
    plants: Set[str] = field(init=False)
    dealers: Set[str] = field(init=False)
    all_skus: Set[str] = field(init=False)
    all_veh_types: Set[str] = field(init=False)
    sku_sizes: Dict[str, float] = field(init=False)

    # 生产：逐日 & 汇总
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)   # (plant, sku, day) -> qty
    sku_prod_total: Dict[Tuple[str, int], int] = field(init=False)           # (sku, day) -> aggregated qty

    # 库存
    sku_initial_inv: Dict[Tuple[str, str], int] = field(init=False)          # (plant, sku) -> init qty
    skus_initial: Dict[str, Set[str]] = field(init=False)                    # plant -> init SKUs
    skus_prod: Dict[str, Set[str]] = field(init=False)                       # plant -> produced SKUs
    skus_plant: Dict[str, Set[str]] = field(init=False)                      # plant -> available SKUs (init ∪ produced)

    # 需求
    demands: Dict[Tuple[str, str], int] = field(init=False)                  # (dealer, sku) -> demand
    skus_dealer: Dict[str, Set[str]] = field(init=False)                     # dealer -> demanded SKUs

    # 历史库存 (递推初值), 仅 day=0
    historical_s_ikt: Dict[Tuple[str, str, int], int] = field(init=False)    # (plant, sku, 0) -> init inv

    # 车辆参数
    veh_type_cost: Dict[str, float] = field(init=False)
    veh_type_cap: Dict[str, float] = field(init=False)
    veh_type_min_load: Dict[str, float] = field(init=False)

    # 库存容量上限
    plant_inv_limit: Dict[str, float] = field(init=False)

    # 原始订单 DataFrame (规范化后存档，统计或调试使用)
    df_order: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------
    # 初始化: 仅设置日志器 (避免重复添加 handler)
    # ------------------------------------------------------------------
    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # 基础文件 / DataFrame 读取工具
    # ------------------------------------------------------------------
    def _validate_file_exists(self, file_path: str, file_description: str) -> bool:
        """验证文件是否存在, (不存在则记录错误并返回 False)"""
        if not os.path.exists(file_path):
            self.logger.error(f"{file_description} 文件不存在: {file_path}")
            return False
        return True

    def _load_csv_safely(
        self,
        file_path: str,
        file_description: str,
        required_cols: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        安全读取 CSV:
          - 存在性检查
          - 读取为 DataFrame
          - 必需列预检查 (若提供)
          - 关键列统一转换为 str
          - 异常处理与日志记录

        参数:
            file_path: 文件路径
            file_description: 日志用描述
            required_cols: 必需的原始列名列表

        返回:
            DataFrame 或 None (失败)
        """
        if not self._validate_file_exists(file_path, file_description):
            return None
        try:
            df = pd.read_csv(file_path, header=0)
            # 早期必需列检查
            if required_cols:
                missing = set(required_cols) - set(df.columns)
                if missing:
                    self.logger.error(
                        f"{file_description} 缺少必需列: {missing} (文件: {file_path})"
                    )
                    raise ValueError(f"{file_description} 缺少必需列: {missing}")
            key_cols = [
                c
                for c in ["product_code", "plant_code", "client_code", "vehicle_type"]
                if c in df.columns
            ]
            for col in key_cols:
                df[col] = df[col].astype(str)
            self.logger.info(f"成功加载 {file_description}: {len(df)} 行数据")
            return df
        except Exception as e:
            self.logger.error(f"加载 {file_description} 失败: {e}")
            return None

    def _reorganize_dataframe(self, df_input: pd.DataFrame, title_map: Dict[str, str]) -> pd.DataFrame:
        """
        重新组织 DataFrame:
          - 重命名列 (title_map: 原名 -> 新名)
          - 标准化列顺序为映射后的列
          - 缺失列抛出异常
        """
        if df_input is None or df_input.empty:
            self.logger.warning("输入 DataFrame 为空, 跳过重组织")
            return df_input

        missing_cols = set(title_map.keys()) - set(df_input.columns)
        if missing_cols:
            self.logger.error(f"DataFrame 缺少必需列: {missing_cols}")
            raise ValueError(f"DataFrame 缺少必需列: {missing_cols}")

        df_result = df_input.rename(columns=title_map)
        ordered_cols = list(title_map.values())
        return df_result.reindex(columns=ordered_cols)

    def _convert_dataframe_types(self, df: pd.DataFrame, type_mapping: Dict[str, type]) -> pd.DataFrame:
        """
        将指定列转换为目标类型:
          - 缺失列安全跳过
          - 数值列使用 to_numeric(errors='coerce') + fillna 防止异常
          - key 列统一为 str
        """
        if df is None or df.empty:
            return df
        df_result = df.copy()
        key_cols = [
            c
            for c in ["product_code", "plant_code", "client_code", "vehicle_type"]
            if c in df_result.columns
        ]
        for col in key_cols:
            df_result[col] = df_result[col].astype(str)
        for col, dtype in type_mapping.items():
            if col not in df_result.columns:
                continue
            try:
                if dtype == str:
                    df_result[col] = df_result[col].astype(str)
                elif dtype == int:
                    df_result[col] = (
                        pd.to_numeric(df_result[col], errors="coerce").fillna(0).astype(int)
                    )
                elif dtype == float:
                    df_result[col] = pd.to_numeric(df_result[col], errors="coerce").fillna(0.0)
            except Exception as e:
                self.logger.warning(f"转换列 {col} 到类型 {dtype} 失败: {e}")
        return df_result

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def load(self):
        """
        加载并构建所有数据接口 (幂等):
          - 若任一步骤失败, 抛出异常以保证外部调用看到一致状态
        """
        try:
            self._load_parameters()
            file_paths = self._construct_file_paths()
            dataframes = self._load_all_dataframes(file_paths)
            self._process_dataframes(dataframes)
            self._compute_derived_data()
            self._validate_data_integrity()
            self.logger.info("数据加载和处理完成")
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise

    # ------------------------------------------------------------------
    # 步骤 1: 参数加载
    # ------------------------------------------------------------------
    def _load_parameters(self):
        """读取 JSON 配置中的惩罚参数, 缺失则抛出异常"""
        config_file = os.path.join(self.input_file_loc, "model_config_alns.json")
        if not self._validate_file_exists(config_file, "配置文件"):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        try:
            with open(config_file, "r", encoding="utf-8") as fp:
                param = json.load(fp)

            self.param_pun_factor1 = float(param.get("pun_factor1", 0))
            self.param_pun_factor2 = float(param.get("pun_factor2", 0))
            self.param_pun_factor3 = float(param.get("pun_factor3", 0))
            self.param_pun_factor4 = float(param.get("pun_factor4", 0))
            self.param_pun_objective = float(param.get("pun_objective", 0))

            self.logger.info("配置参数加载成功")
        except Exception as e:
            self.logger.error(f"加载配置参数失败: {e}")
            raise

    # ------------------------------------------------------------------
    # 步骤 2: 构建文件路径
    # ------------------------------------------------------------------
    def _construct_file_paths(self) -> Dict[str, str]:
        """构建数据集目录内各数据文件路径映射"""
        dataset_path = os.path.join(self.input_file_loc, self.dataset_name)
        return {
            "order": os.path.join(dataset_path, "order.csv"),
            "plant_cap": os.path.join(dataset_path, "plant_size_upper_limit.csv"),
            "sku_size": os.path.join(dataset_path, "product_size.csv"),
            "vehicle": os.path.join(dataset_path, "vehicle.csv"),
            "production": os.path.join(dataset_path, "warehouse_production.csv"),
            "inventory": os.path.join(dataset_path, "warehouse_storage.csv"),
        }

    # ------------------------------------------------------------------
    # 步骤 3: 读取所有 DataFrame
    # ------------------------------------------------------------------
    def _load_all_dataframes(self, file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """批量加载 CSV -> DataFrame 映射, 任意失败则终止"""
        dataframes = {}
        file_desc = {
            "order": "订单需求信息",
            "plant_cap": "生产基地库存容量上限",
            "sku_size": "SKU 体积信息",
            "vehicle": "车辆类型信息",
            "production": "生产计划",
            "inventory": "期初库存",
        }
        required_columns = {
            "order": ["client_code", "product_code", "volume"],
            "plant_cap": ["plant_code", "max_volume"],
            "sku_size": ["product_code", "standard_size"],
            "vehicle": ["vehicle_type", "carry_standard_size", "min_standard_size", "cost_to_use"],
            "production": ["plant_code", "product_code", "produce_date", "volume"],
            "inventory": ["plant_code", "product_code", "volume"],
        }
        for key, path in file_paths.items():
            df = self._load_csv_safely(path, file_desc[key], required_cols=required_columns.get(key))
            if df is None:
                raise FileNotFoundError(f"必需文件 {file_desc[key]} 加载失败")
            dataframes[key] = df
        return dataframes

    # ------------------------------------------------------------------
    # 步骤 4: 规范化 DataFrame
    # ------------------------------------------------------------------
    def _process_dataframes(self, dataframes: Dict[str, pd.DataFrame]):
        """列映射与类型转换, 生成 _df_* 中间结果"""

        # 订单
        title_map = {"client_code": "dealer_id", "product_code": "sku_id", "volume": "order_qty"}
        type_mapping = {"dealer_id": str, "sku_id": str, "order_qty": int}
        df_order = self._reorganize_dataframe(dataframes["order"], title_map)
        self.df_order = self._convert_dataframe_types(df_order, type_mapping)

        # 基地容量
        title_map = {"plant_code": "fact_id", "max_volume": "max_cap"}
        type_mapping = {"fact_id": str, "max_cap": float}
        df_plant_cap = self._reorganize_dataframe(dataframes["plant_cap"], title_map)
        self._df_plant_cap = self._convert_dataframe_types(df_plant_cap, type_mapping)

        # SKU 大小
        title_map = {"product_code": "sku_id", "standard_size": "size"}
        type_mapping = {"sku_id": str, "size": float}
        df_sku_size = self._reorganize_dataframe(dataframes["sku_size"], title_map)
        self._df_sku_size = self._convert_dataframe_types(df_sku_size, type_mapping)

        # 车辆
        title_map = {
            "vehicle_type": "vehicle_type",
            "carry_standard_size": "capacity",
            "min_standard_size": "min_load",
            "cost_to_use": "cost",
        }
        type_mapping = {"vehicle_type": str, "capacity": float, "min_load": float, "cost": float}
        df_all_veh = self._reorganize_dataframe(dataframes["vehicle"], title_map)
        self._df_all_veh = self._convert_dataframe_types(df_all_veh, type_mapping)

        # 生产计划
        title_map = {"plant_code": "fact_id", "product_code": "sku_id", "produce_date": "day", "volume": "prod_qty"}
        type_mapping = {"fact_id": str, "sku_id": str, "day": int, "prod_qty": int}
        df_sku_prod = self._reorganize_dataframe(dataframes["production"], title_map)
        self._df_sku_prod = self._convert_dataframe_types(df_sku_prod, type_mapping)

        # 期初库存
        title_map = {"plant_code": "fact_id", "product_code": "sku_id", "volume": "inv"}
        type_mapping = {"fact_id": str, "sku_id": str, "inv": int}
        df_sku_inv = self._reorganize_dataframe(dataframes["inventory"], title_map)
        self._df_sku_inv = self._convert_dataframe_types(df_sku_inv, type_mapping)

    # ------------------------------------------------------------------
    # 步骤 5: 派生结构计算
    # ------------------------------------------------------------------
    def _compute_derived_data(self):
        """聚合集合 / 映射 / 期初递推结构"""
        # 规划周期
        if len(self._df_sku_prod) > 0:
            self.horizons = int(self._df_sku_prod["day"].max())

        # 基础集合
        self.plants = set(self._df_plant_cap["fact_id"].unique())
        self.dealers = set(self.df_order["dealer_id"].unique())
        self.all_skus = set(self._df_sku_size["sku_id"].unique())
        self.all_veh_types = set(self._df_all_veh["vehicle_type"].unique())

        # SKU 大小
        self.sku_sizes = dict(zip(self._df_sku_size["sku_id"], self._df_sku_size["size"]))

        # 生产
        self.sku_prod_each_day = self._df_sku_prod.set_index(["fact_id", "sku_id", "day"])["prod_qty"].to_dict()
        self.sku_prod_total = self._df_sku_prod.groupby(["sku_id", "day"])["prod_qty"].sum().to_dict()

        # 期初库存
        self.sku_initial_inv = self._df_sku_inv.set_index(["fact_id", "sku_id"])["inv"].to_dict()

        # 历史库存 (仅 day=0)
        self.historical_s_ikt = {}
        for (plant, sku), inv in self.sku_initial_inv.items():
            self.historical_s_ikt[plant, sku, 0] = inv

        # SKU 可用集合映射
        self._compute_sku_mappings()

        # 需求
        self.demands = self.df_order.set_index(["dealer_id", "sku_id"])["order_qty"].to_dict()
        self.skus_dealer = {}
        for (dealer, sku) in self.demands:
            self.skus_dealer.setdefault(dealer, set()).add(sku)

        # 车辆
        self.veh_type_cost = dict(zip(self._df_all_veh["vehicle_type"], self._df_all_veh["cost"]))
        self.veh_type_cap = dict(zip(self._df_all_veh["vehicle_type"], self._df_all_veh["capacity"]))
        self.veh_type_min_load = dict(zip(self._df_all_veh["vehicle_type"], self._df_all_veh["min_load"]))

        # 基地容量上限
        self.plant_inv_limit = dict(zip(self._df_plant_cap["fact_id"], self._df_plant_cap["max_cap"]))

    def _compute_sku_mappings(self):
        """构建期初/生产/综合 SKU 集合映射 (保持原逻辑不变)"""
        # 期初库存 SKU
        self.skus_initial = defaultdict(set)
        for (plant, sku) in self.sku_initial_inv:
            self.skus_initial[plant].add(sku)
        self.skus_initial = dict(self.skus_initial)

        # 生产 SKU
        self.skus_prod = defaultdict(set)
        for (plant, sku, day) in self.sku_prod_each_day:
            self.skus_prod[plant].add(sku)
        self.skus_prod = dict(self.skus_prod)

        # 可供 (期初 ∪ 生产)
        self.skus_plant = {}
        for plant in self.plants:
            initial = self.skus_initial.get(plant, set())
            prod = self.skus_prod.get(plant, set())
            self.skus_plant[plant] = initial.union(prod)

    # ------------------------------------------------------------------
    # 步骤 6: 完整性校验
    # ------------------------------------------------------------------
    def _validate_data_integrity(self):
        """对关键集合/映射进行存在性与交叉一致性检查"""
        errors = []

        # 必要集合
        if not self.plants:
            errors.append("没有找到生产基地数据")
        if not self.dealers:
            errors.append("没有找到经销商数据")
        if not self.all_skus:
            errors.append("没有找到 SKU 数据")
        if not self.all_veh_types:
            errors.append("没有找到车辆类型数据")

        # SKU 大小缺失
        order_skus = {sku for dealer, sku in self.demands.keys()}
        missing_sku_sizes = order_skus - set(self.sku_sizes.keys())
        if missing_sku_sizes:
            errors.append(f"订单中的 SKU 缺少尺寸信息: {missing_sku_sizes}")

        # 基地无可供 SKU (仅警告)
        plants_with_skus = set(self.skus_plant.keys())
        plants_without = self.plants - plants_with_skus
        if plants_without:
            self.logger.warning(f"以下生产基地没有可提供的 SKU: {plants_without}")

        if errors:
            msg = "数据完整性验证失败:\\n" + "\\n".join(f"- {e}" for e in errors)
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("数据完整性验证通过")

    # ------------------------------------------------------------------
    # 供应链与访问帮助函数
    # ------------------------------------------------------------------
    def available_skus_in_plant(self, plant_id: str) -> Set[str]:
        """返回某生产基地可提供的全部 SKU 集 (不存在则空集, 并记录警告)"""
        if plant_id not in self.plants:
            self.logger.warning(f"生产基地 {plant_id} 不存在于数据中")
            return set()
        return self.skus_plant.get(plant_id, set())

    def plan_dealer_skus(self, plant_id: str) -> Set[Tuple[str, str]]:
        """返回该生产基地潜在服务的 (经销商, SKU) 组合 (依据 supply_chain 交集)"""
        if plant_id not in self.plants:
            self.logger.warning(f"生产基地 {plant_id} 不存在于数据中")
            return set()
        supply_chain = self.construct_supply_chain()
        result = set()
        for (plant, dealer), skus in supply_chain.items():
            if plant == plant_id:
                for sku in skus:
                    result.add((dealer, sku))
        return result

    def available_skus_to_dealer(self, plant_id: str, dealer_id: str) -> Set[str]:
        """返回生产基地对指定经销商可供 SKU 集合 (不存在返回空集并警告)"""
        if plant_id not in self.plants:
            self.logger.warning(f"生产基地 {plant_id} 不存在于数据中")
            return set()
        if dealer_id not in self.dealers:
            self.logger.warning(f"经销商 {dealer_id} 不存在于数据中")
            return set()
        supply_chain = self.construct_supply_chain()
        return supply_chain.get((plant_id, dealer_id), set())

    def construct_supply_chain(self) -> Dict[Tuple[str, str], Set[str]]:
        """
        构建供应链关系:
            {(plant, dealer): intersect_skus} 其中 intersect_skus 为 plant 可供且 dealer 需求
        结果缓存以避免重复计算。
        """
        if not hasattr(self, "_supply_chain_cache"):
            supply_chain = defaultdict(set)
            for plant, skus_supply in self.skus_plant.items():
                for dealer, skus_need in self.skus_dealer.items():
                    inter = skus_supply.intersection(skus_need)
                    if inter:
                        supply_chain[plant, dealer] = inter
            self._supply_chain_cache = dict(supply_chain)
        return self._supply_chain_cache

    # ------------------------------------------------------------------
    # 指标汇总 / 供需平衡分析
    # ------------------------------------------------------------------
    def get_total_demand_for_sku(self, sku_id: str) -> int:
        """聚合指定 SKU 的总需求量"""
        total = 0
        for (dealer, sku), demand in self.demands.items():
            if sku == sku_id:
                total += demand
        return total

    def get_total_production_for_sku(self, sku_id: str, day: Optional[int] = None) -> int:
        """返回指定 SKU 在某一日或所有日的总生产量"""
        if day is not None:
            return self.sku_prod_total.get((sku_id, day), 0)
        total = 0
        for (sku, d), qty in self.sku_prod_total.items():
            if sku == sku_id:
                total += qty
        return total

    def get_total_initial_inventory_for_sku(self, sku_id: str) -> int:
        """返回指定 SKU 的期初总库存"""
        total = 0
        for (plant, sku), inv in self.sku_initial_inv.items():
            if sku == sku_id:
                total += inv
        return total

    def get_sku_supply_demand_balance(self, sku_id: str) -> Dict[str, Union[int, float]]:
        """
        供需平衡:
            total_supply = initial + production_all
            balance = supply - demand
            balance_ratio = supply / demand (无需求 -> inf)
        """
        total_demand = self.get_total_demand_for_sku(sku_id)
        total_init = self.get_total_initial_inventory_for_sku(sku_id)
        total_prod = self.get_total_production_for_sku(sku_id)
        total_supply = total_init + total_prod
        balance = total_supply - total_demand
        ratio = total_supply / total_demand if total_demand > 0 else float("inf")
        return {
            "sku_id": sku_id,
            "total_demand": total_demand,
            "total_initial_inventory": total_init,
            "total_production": total_prod,
            "total_supply": total_supply,
            "balance": balance,
            "balance_ratio": ratio,
            "status": "surplus" if balance > 0 else "deficit" if balance < 0 else "balanced",
        }

    def get_summary_statistics(self) -> Dict[str, Union[int, float]]:
        """整体汇总指标 (用于日志或快速 sanity check)"""
        total_demand = sum(self.demands.values())
        total_initial = sum(self.sku_initial_inv.values())
        total_prod = sum(self.sku_prod_total.values())
        total_supply = total_initial + total_prod
        return {
            "num_plants": len(self.plants),
            "num_dealers": len(self.dealers),
            "num_skus": len(self.all_skus),
            "num_vehicle_types": len(self.all_veh_types),
            "planning_horizon": self.horizons,
            "total_demand": total_demand,
            "total_initial_inventory": total_initial,
            "total_production": total_prod,
            "total_supply": total_supply,
            "supply_demand_ratio": total_supply / total_demand if total_demand > 0 else float("inf"),
        }
