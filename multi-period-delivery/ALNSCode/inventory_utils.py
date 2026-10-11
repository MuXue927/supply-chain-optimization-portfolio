"""
模块: inventory_utils.py
职责:
- 提供与库存相关的常用聚合工具函数，供 destroy/repair/selection 等算子复用。
- 目标是将常见的 O(V*S) 聚合操作提前计算为 O(V + D*S) 或 O(P*S) 形式，
  从而减少在多次算子调用中重复遍历车辆列表的开销。

风格与设计考量（与 alnsopt.py 注释风格保持一致）:
- 以只读快照方式读取 state 的 s_ikt 与 vehicles，调用方在修改 state 后应负责重新计算或重建缓存。
- 函数尽量返回简单、可序列化的聚合结构（dict / defaultdict），方便日志、断言与单元测试。
- 注重说明函数的复杂度、使用前提与不变式，便于维护者在引入新算子时正确使用。

注意:
- 本模块不直接修改 state.s_ikt 或 state.vehicles；若需要批量修改库存请使用 alnsopt.batch_update_inventory
  或在修改后显式调用 compute_inventory / mark_objective_dirty 来保持一致性。
"""
from collections import defaultdict


def precompute_plant_day_inventory(state):
    """
    聚合 state.s_ikt 到 (plant, day) 维度并返回总库存映射。

    返回:
      defaultdict(int): keys 为 (plant, day)，value 为该工厂在该日的库存总量

    说明与使用约定:
      - 该函数基于传入状态的当前 s_ikt 快照进行聚合，属于只读操作。
      - 若调用方随后对 state.s_ikt 进行了修改（例如通过 veh_loading / batch_update_inventory），
        应在使用本聚合结果前重新调用本函数以获得最新视图。
      - 典型用途包括: 校验工厂日库存上限、计算过量库存、用于启发式拆车/合并策略等。

    复杂度:
      - O(N) 其中 N 为 state.s_ikt 中记录的索引数量 (≈ P * S * H)
    """
    plant_day_inventory = defaultdict(int)
    for (p, sku, d), inv in state.s_ikt.items():
        # 将每个 (plant, sku, day) 的库存累加到 (plant, day)
        plant_day_inventory[(p, d)] += inv
    return plant_day_inventory


def precompute_dealer_shipments(state):
    """
    预计算与经销商相关的发运/需求聚合，返回三类聚合结果：

      - shipments_by_dealer_sku_day: dict((dealer, sku, day) -> qty)
          * 表示每个经销商在每个 SKU 和每个出发日的发运量
      - shipments_by_dealer_total: dict(dealer -> total_shipped_qty)
          * 表示每个经销商在当前解中的累计发运总量（所有 SKU 与所有日合计）
      - dealer_total_demand: dict(dealer -> total_demand)
          * 表示每个经销商基于 input data 的总需求量（用于比较 shipped vs demand）

    目的与动机:
      - 在许多移除算子（如 worst_removal / surplus_removal 等）中，需要频繁按经销商聚合
        车辆发运量与需求。通过预计算将重复 O(V*S) 的遍历降为一次性 O(V + D*S) 聚合，
        显著降低算子运行时的常数开销。

    使用注意:
      - 返回的数据结构使用 defaultdict(int)，便于直接 += 操作与缺省为 0 的访问。
      - 该函数不修改 state；若在聚合后车辆集合发生变化，请重新计算。

    复杂度:
      - O(V * L) 其中 V 为车辆数，L 为每车上装载的货物条目平均数；随后计算 dealer demand 为 O(D*S)
    """
    shipments_by_dealer_sku_day = defaultdict(int)
    shipments_by_dealer_total = defaultdict(int)
    for veh in state.vehicles:
        dealer = veh.dealer_id
        for (sku, day), qty in veh.cargo.items():
            # 逐条累加每辆车的 cargo 到按 (dealer, sku, day) 的聚合中
            shipments_by_dealer_sku_day[(dealer, sku, day)] += qty
            shipments_by_dealer_total[dealer] += qty

    dealer_total_demand = defaultdict(int)
    for (dealer, sku_id), dem in state.data.demands.items():
        # 将输入数据中的需求按经销商聚合
        dealer_total_demand[dealer] += dem

    return shipments_by_dealer_sku_day, shipments_by_dealer_total, dealer_total_demand
