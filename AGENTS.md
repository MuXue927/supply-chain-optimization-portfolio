# AGENTS.md — 本仓库的项目级工作规则

> 适用于在这个仓库里做任何修改、测量或排障的 agent。
> 依据：`local-only/提示词-Codex-2026-10-17.md` §4、`docs/engineering_lessons.md` §3.8/§4.4、
> 以及 2026-10-23 的 `refactor-protocol` 对齐验证。
> 更细的原文以 `local-only/` 与 `docs/engineering_lessons.md` 为准。

## 0. 开工前先读（预计 30 分钟，读完再动手）

1. `local-only/交接书-Codex-2026-10-16.md`（仓库地图、进度、踩过的坑）
2. `local-only/修复主线重估-2026-10-16.md`（**唯一进度入口**，顶部有最新基线与当前优先级）
3. 本轮相关的最近 1–3 轮记录（在 `local-only/修复记录-P1-2026-10-16.md` 末尾）

## 1. 范围纪律（这些行为会浪费大量 token，直接禁止）

- 不做全仓扫描；仓库地图已在交接书 §1。
- 不重做 `local-only/审核-*.md` 的三轮审核，其结论已逐条处置。
- 不重测已知数字；只有"复查某个具体数字"才允许重测。
- 不碰 `single-period-delivery/`、任何 `monolithic/`、任何 `cplex_api_*/`、任何 Gurobi 相关代码。
- 不 mock Gurobi（本机无 license）。
- 不改 `local-only/` 之外的文档，除非任务明确要求。
- `local-only/` 已在 `.gitignore`，不推送。

## 2. 证据规矩（用数据说话）

- 声称"某处是这样"必须给出**可复现命令 + 命中行**。
- **运行时事实必须实测**：调用次数、占比、哪条分支真的执行——静态读代码看不出来。
- **先测再改**；探索性轮次先给结论，等人类说"可以改"再动生产代码。
- 结论被自己推翻时，把过程写进记录；别怕证伪，怕的是不证。

## 3. 性能测量 6 条纪律（`docs/engineering_lessons.md` §3.8）

1. profile 的收益 ≠ 端到端收益；profile 只用于定位。
2. **同进程交替 A/B**——本机同代码可得 74/90/117/135/164/171 轮，比较两次独立运行什么都证明不了。
3. 先算「占比 × 提速比」再动手。
4. 绝不拿解质量换速度：目标值变化是**正确性信号**，不是噪声。
5. 把"必然失败"的判断提到昂贵操作之前，但必须**先在全部可达用例上证明等价**（曾比 5,570 个、0 分歧）。
6. **插桩先验证再采信**：先 `ast.parse` + 断言注入数 > 0，再相信数字。

## 4. 编码陷阱（完整 13 条见交接书 §6.3）

- 补丁要打在**类**上，`monkeypatch.setattr(instance, ...)` 对 `self.method()` 无效。
- 模块用 `from ... import X` 时，patch **使用方模块**。
- 缓存键含 ndarray 时**逐组件显式比较**（`tuple.__eq__` 会报 truth value is ambiguous）。
- **日期必须读时钟**，不能从文件名推定（§4.4；曾因此把 8 份文档日期写错一天）。

## 5. 目录与命令

- 探针脚本 → `local-only/probes/`；产出 → `local-only/probe-output/`；`local-only/` 顶层只放 `.md`。
- 新增探针后跑一次 `python local-only/probes/_fix_probe_paths.py`（幂等，期望 `repaired: 0` / `0 broken`）。
- 探针**必须从仓库根目录**启动。
- 解释器固定用 `.\.venv\Scripts\python.exe`。

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=multi_period_delivery `
  --cov=single_period_delivery --cov-branch --cov-report=term --cov-fail-under=56
.\.venv\Scripts\python.exe -m ruff check multi-period-delivery single-period-delivery
```

## 6. 改动流程（仅在允许改代码时）

改代码 → 全量 pytest → ruff 两包 → 覆盖率不降 → 端到端跑一次 ALNS（`ok=True`）→
更新 `local-only/` 记录 → 提交（**英文单行祈使句**）→ 推送。

## 7. 结构类改动（重构 / 拆分 / 提取 / 去重 / 迁移）

命中上述意图时，加载 `refactor-protocol` skill，并按下列**三条适配**执行（与 skill 原文不同的地方以本节为准）：

- **一步 = 一个可单独 revert 的单元**，不是一次 commit。commit 仍按"一轮一个 M 编号"，
  但**一个 commit 内不得混合多种变换类型**（改名 / 提取 / 移动 / 换依赖）。
- **diff 规模阈值（400 行 / 10 文件）是提示，不是闸门**；超阈值时要拆的是"变换类型"，
  而不是按文件数硬拆。
- **性能判据以 §3 的 6 条纪律为准**，skill 里"性能回退 >20% 视为回退"在本仓库**不适用**。
- 顺手发现的问题：**报告，不修**；自己造成的孤儿（不再使用的 import/变量）自己清理。

## 8. 当前基线快照（会过期，以 `修复主线重估` 顶部为准）

```text
2026-10-25 12:34   HEAD = M45 提交   632 passed / 42 deselected
覆盖率 65.77%（门禁 56%）   ruff 干净   盲 except Exception 113（原 249）
M45 收窄盲 except Exception：helper 的显式抛出面是闭合的五元组，落成
alns/errors.STATE_OPERATION_ERRORS；另删除 15 处纯日志守卫与 7 处纯 setattr 守卫。
保留的 113 个是回调/算子隔离、范围外模块、惰性 import/可选依赖、I/O、deepcopy。
下一阶段：tuner / ML / full 三种模式的效果与正确性验证。
M44（12:01，只读）建模口径已确认：工厂指派不可调；发货日自由且**已是现状**
（订单文件无 day 列，demand=周期总量，cargo 日只等于车辆发车日）。
装载下界：一车一店 1345 / 同厂同日混装 1176 / 同厂跨日混装 1164；
实际 1464 距一车一店下界仅 119 辆，A−B=169 辆（12.6%）。
唯一大自由度是「一车多店」，需模型层面定夺；模型不变时转 249 个盲 handler。
M43 结论：search 内部调参已收口。收尾跨期整合可改进移动 0（cap 不绑定）；
1464 辆 vs 最大车型下界 1345（+119，8.1%），罚款仅 39.72，多目标拆分上界
1/1161（0.09%）⇒ 车队已到逐日约束下的打包极限。
真缺陷：ROULETTE_SEG_LENGTH=80 大于整场轮数（49–80），算子权重从不更新；
已加 ROULETTE_PROGRESS_SEGMENTS（按 elapsed/MAX_RUNTIME，默认 0）。
下一优先级：模型决策面（工厂指派、发货日自由度）需作者定夺；
其次 249 个盲 handler、恒 no-op 算子、预检缓存与 ML 选择器。

本地跑 pytest 时不要并发：pytest.ini 固定 --basetemp=.pytest_tmp，
第二个会话会删掉第一个的目录树并在 Windows 上留下 WinError 5 的残留。
需要并跑就传唯一的 --basetemp=tmp\pytest-<task>。
```

## 9. 失败也要记录

被回退的尝试、负面结果、踩到的坑，同样写进 `local-only/` 记录。
本项目的经验主要来自"先做错、再写下来"。
