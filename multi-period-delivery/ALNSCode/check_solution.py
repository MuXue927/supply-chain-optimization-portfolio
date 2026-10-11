"""
check_solution

模块定位
    对 ALNS 求解产生的输出结果进行系统化一致性与可行性检查（需求满足、基地容量、
    车辆容量、库存流转等），并生成可读的验证摘要，辅助问题定位与结果复核。

设计原则
    - 不修改输入/输出数据，只做只读校验
    - 统一日志输出（logger），尽量避免在库内修改全局 logging 配置
    - 校验方法解耦为独立函数，便于单独调试与扩展

输入/输出
    - 输入：DataALNS（静态数据）与 OutPutData（算法输出）
    - 输出：ValidationResult 列表与总体可行性布尔值
"""

# =========================
# 标准库
# =========================
import os
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List

# =========================
# 项目内部
# =========================
from .InputDataALNS import DataALNS
from .OutputDataALNS import OutPutData
from .alns_config import default_config as ALNSConfig

# 设置日志（建议在主程序统一配置 logging，这里仅获取模块级 logger）
logger = logging.getLogger(__name__)

# 获取当前.py文件所在的目录，这样可以确保路径正确，即使文件被移动到其他位置也不会影响
current_dir = os.path.dirname(__file__)
# 当前.py文件所在目录的上一级目录
par_path = os.path.dirname(current_dir)
input_loc = os.path.join(par_path, 'datasets', 'multiple-periods', ALNSConfig.DATASET_TYPE)
output_loc = os.path.join(par_path, 'OutPut-ALNS', 'multiple-periods', ALNSConfig.DATASET_TYPE)


@dataclass
class ValidationResult:
    """验证结果的数据类"""
    constraint_name: str
    is_satisfied: bool
    violations: List[str]
    total_violations: int
    details: Dict


class SolutionValidator:
    """解决方案验证器类"""
    
    def __init__(self, input_data: DataALNS, output_data: OutPutData):
        self.input_data = input_data
        self.output_data = output_data
        self.validation_results = []
    
    def check_demand_satisfaction(self) -> ValidationResult:
        """检查订单满足情况"""
        logger.info("检查订单满足情况...")
        violations = []
        details = {
            'total_demands': len(self.input_data.demands),
            'satisfied_demands': 0,
            'oversupplied_demands': 0,
            'undersupplied_demands': 0,
            'total_shortage': 0,
            'total_oversupply': 0
        }
        
        # 遍历所有需求
        for (dealer_id, sku_id), demand in self.input_data.demands.items():
            # 计算实际配送量
            actual_shipped = 0
            # 从车辆配送记录中统计配送量
            for day in range(1, self.input_data.horizons + 1):
                for fact_id in self.input_data.plants:
                    actual_shipped += self.output_data.order_fulfill.get((day, fact_id, dealer_id, sku_id), 0)
            
            # 检查需求满足情况
            if actual_shipped < demand:
                shortage = demand - actual_shipped
                details['undersupplied_demands'] += 1
                details['total_shortage'] += shortage
                violations.append(
                    f"经销商 {dealer_id} 的 SKU {sku_id} 需求未满足: "
                    f"需求 {demand}, 实际配送 {actual_shipped}, 缺口 {shortage}"
                )
            elif actual_shipped > demand:
                oversupply = actual_shipped - demand
                details['oversupplied_demands'] += 1
                details['total_oversupply'] += oversupply
                # 超量配送不算违反约束，但记录信息
                logger.info(
                    f"经销商 {dealer_id} 的 SKU {sku_id} 配送超量: "
                    f"需求 {demand}, 实际配送 {actual_shipped}, 超出 {oversupply}"
                )
            else:
                details['satisfied_demands'] += 1
        
        is_satisfied = len(violations) == 0
        return ValidationResult(
            constraint_name="需求满足约束",
            is_satisfied=is_satisfied,
            violations=violations,
            total_violations=len(violations),
            details=details
        )

    
    def check_plant_capacity(self) -> ValidationResult:
        """检查生产基地库存上限约束"""
        logger.info("检查生产基地库存上限约束...")
        violations = []
        details = {
            'total_checks': 0,
            'violations_count': 0,
            'max_violation': 0,
            'total_excess': 0
        }
        
        # 遍历每个工厂和每个时间段
        for plant_id in self.input_data.plants:
            for day in range(1, self.input_data.horizons + 1):
                details['total_checks'] += 1
                total_inventory = 0
                # 计算当天的库存总量
                for sku_id in self.input_data.all_skus:
                    if (plant_id, sku_id, day) in self.output_data.sku_inv_left:
                        total_inventory += self.output_data.sku_inv_left[(plant_id, sku_id, day)]
                
                # 检查是否超过库存上限, 库存上限为存储SKU的数量上限
                capacity_limit = self.input_data.plant_inv_limit[plant_id]
                if total_inventory > capacity_limit:
                    excess = total_inventory - capacity_limit
                    details['violations_count'] += 1
                    details['total_excess'] += excess
                    details['max_violation'] = max(details['max_violation'], excess)
                    violations.append(
                        f"工厂 {plant_id} 在第 {day} 天超出库存上限: "
                        f"上限 {capacity_limit}, 实际 {total_inventory}, 超出 {excess}"
                    )
        
        is_satisfied = len(violations) == 0
        return ValidationResult(
            constraint_name="生产基地库存上限约束",
            is_satisfied=is_satisfied,
            violations=violations,
            total_violations=len(violations),
            details=details
        )
    
    def check_vehicle_capacity(self) -> ValidationResult:
        """检查车辆容量约束"""
        logger.info("检查车辆容量约束...")
        violations = []
        details = {
            'total_vehicles': len(self.output_data.vehicle_load),
            'capacity_violations': 0,
            'min_load_warnings': 0,
            'max_capacity_violation': 0,
            'total_excess_capacity': 0
        }
        
        # 遍历每个配送任务
        for (day, fact_id, dealer_id, vehicle_id, vehicle_type), load in self.output_data.vehicle_load.items():
            # 检查是否超过车辆容量上限
            capacity_limit = self.input_data.veh_type_cap[vehicle_type]
            if load > capacity_limit:
                excess = load - capacity_limit
                details['capacity_violations'] += 1
                details['total_excess_capacity'] += excess
                details['max_capacity_violation'] = max(details['max_capacity_violation'], excess)
                violations.append(
                    f"车辆 {vehicle_id} (类型: {vehicle_type}) 在第 {day} 天超出容量上限: "
                    f"上限 {capacity_limit}, 实际装载 {load}, 超出 {excess}"
                )
            
            # 检查是否满足最小起运量要求（这是软约束，不影响可行性）
            min_load_req = self.input_data.veh_type_min_load[vehicle_type]
            if load < min_load_req:
                details['min_load_warnings'] += 1
                logger.info(
                    f"车辆 {vehicle_id} (类型: {vehicle_type}) 在第 {day} 天未达到最小起运量: "
                    f"最小起运量 {min_load_req}, 实际装载 {load}"
                )
        
        is_satisfied = len(violations) == 0
        return ValidationResult(
            constraint_name="车辆容量约束",
            is_satisfied=is_satisfied,
            violations=violations,
            total_violations=len(violations),
            details=details
        )
    
    def check_inventory_flow(self) -> ValidationResult:
        """检查库存流转约束"""
        logger.info("检查库存流转约束...")
        violations = []
        details = {
            'total_flow_checks': 0,
            'negative_inventory_violations': 0,
            'record_inconsistencies': 0,
            'min_negative_inventory': 0,
            'max_inconsistency': 0
        }
        
        # 记录比较允许的误差阈值（可由集中配置覆盖）
        record_eps = getattr(ALNSConfig, "RECORD_INCONSISTENCY_EPS", 1e-6)
        
        # 遍历每个工厂的每种SKU
        for plant_id in self.input_data.plants:
            for sku_id in self.input_data.all_skus:
                # 获取期初库存
                current_inv = self.input_data.sku_initial_inv.get((plant_id, sku_id), 0)
                
                # 按时间顺序检查库存流转
                for day in range(1, self.input_data.horizons + 1):
                    details['total_flow_checks'] += 1
                    
                    # 加上当天生产量
                    current_inv += self.input_data.sku_prod_each_day.get((plant_id, sku_id, day), 0)
                    
                    # 减去当天配送量
                    for dealer_id in self.input_data.dealers:
                        current_inv -= self.output_data.order_fulfill.get((day, plant_id, dealer_id, sku_id), 0)
                    
                    # 检查库存是否为负
                    if current_inv < 0:
                        details['negative_inventory_violations'] += 1
                        details['min_negative_inventory'] = min(details['min_negative_inventory'], current_inv)
                        violations.append(
                            f"工厂 {plant_id} 的 SKU {sku_id} 在第 {day} 天出现负库存: {current_inv}"
                        )
                    
                    # 检查与记录的剩余库存是否一致
                    recorded_inv = self.output_data.sku_inv_left.get((plant_id, sku_id, day), 0)
                    inconsistency = abs(current_inv - recorded_inv)
                    if inconsistency > record_eps:  # 使用可配置的小误差范围进行比较
                        details['record_inconsistencies'] += 1
                        details['max_inconsistency'] = max(details['max_inconsistency'], inconsistency)
                        violations.append(
                            f"工厂 {plant_id} 的 SKU {sku_id} 在第 {day} 天库存记录不一致: "
                            f"计算值 {current_inv}, 记录值 {recorded_inv}, 差异 {inconsistency}"
                        )
        
        is_satisfied = len(violations) == 0
        return ValidationResult(
            constraint_name="库存流转约束",
            is_satisfied=is_satisfied,
            violations=violations,
            total_violations=len(violations),
            details=details
        )
    
    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """执行所有约束检查"""
        logger.info("开始解决方案验证...")
        
        # 执行所有约束检查
        validation_methods = [
            self.check_demand_satisfaction,
            self.check_plant_capacity,
            self.check_vehicle_capacity,
            self.check_inventory_flow
        ]
        
        all_results = []
        overall_feasible = True
        
        for validation_method in validation_methods:
            try:
                result = validation_method()
                all_results.append(result)
                if not result.is_satisfied:
                    overall_feasible = False
            except Exception as e:
                logger.error(f"验证方法 {validation_method.__name__} 失败: {str(e)}")
                # 创建错误结果
                error_result = ValidationResult(
                    constraint_name=validation_method.__name__,
                    is_satisfied=False,
                    violations=[f"验证过程出错: {str(e)}"],
                    total_violations=1,
                    details={}
                )
                all_results.append(error_result)
                overall_feasible = False
        
        self.validation_results = all_results
        return overall_feasible, all_results
    
    def print_validation_summary(self):
        """打印验证结果摘要"""
        print("\n" + "="*60)
        print("解决方案验证结果摘要")
        print("="*60)
        
        total_violations = 0
        for result in self.validation_results:
            status = "✅ 通过" if result.is_satisfied else "❌ 违反"
            print(f"{result.constraint_name}: {status}")
            if not result.is_satisfied:
                print(f"  违反数量: {result.total_violations}")
                total_violations += result.total_violations
                # 显示前3个违反详情
                for i, violation in enumerate(result.violations[:3]):
                    print(f"    {i+1}. {violation}")
                if len(result.violations) > 3:
                    print(f"    ... 还有 {len(result.violations) - 3} 个违反")
            print()
        
        print(f"总违反数量: {total_violations}")
        overall_status = "可行" if total_violations == 0 else "不可行"
        print(f"解决方案状态: {overall_status}")
        print("="*60)


def check_solution(input_loc: str, output_loc: str) -> bool:
    """
    主检查函数
    参数:
        input_loc   输入数据根目录（datasets/...）
        output_loc  输出数据根目录（OutPut-ALNS/...）
    返回:
        bool        是否通过全部校验
    """
    try:
        medium_path = f'dataset_{ALNSConfig.DATASET_IDX}'
        dataset_output_loc = os.path.join(output_loc, medium_path)
        
        # 加载输入数据
        logger.info("加载输入数据...")
        input_data = DataALNS(input_loc, dataset_output_loc, medium_path)
        input_data.load()
        
        # 加载输出数据
        logger.info("加载输出数据...")
        output_parent_dir = os.path.dirname(dataset_output_loc)
        output_data = OutPutData(output_parent_dir, medium_path)
        output_data.load()
        
        # 打印数据统计
        print(f"\n数据统计:")
        print(f"输入数据 - 工厂: {len(input_data.plants)}, 经销商: {len(input_data.dealers)}, SKU: {len(input_data.all_skus)}")
        output_stats = output_data.get_summary_stats()
        print(f"输出数据统计: {output_stats}")
        
        # 创建验证器并执行验证
        validator = SolutionValidator(input_data, output_data)
        overall_feasible, validation_results = validator.validate_all()
        
        # 打印详细结果
        validator.print_validation_summary()
        
        return overall_feasible
        
    except Exception as e:
        logger.error(f"解决方案检查失败: {str(e)}")
        print(f"\n检查过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    check_solution(input_loc, output_loc)
