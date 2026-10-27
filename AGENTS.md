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
  --cov=single_period_delivery --cov-branch --cov-report=term --cov-fail-under=65
.\.venv\Scripts\python.exe -m ruff check multi-period-delivery single-period-delivery
```

**覆盖率门禁常设规则**：每轮收尾核对"CI 实测基线 − 门禁 ≤ 1pp"。
超出即**当轮**抬门（新门禁 = CI 实测 − 0.5pp，向下取 0.5 的整数倍），
三处同步：`ci.yml` 的 `--cov-fail-under`、`ci.yml` 注释块、本节的命令。

- 基准取 **CI** 实测值，不取本地：CI 跳过 4 个 CPLEX 测试，比本地 Windows 低约 2pp，
  用本地值会让门禁重新变松。
- 为什么必须 ≤1pp：门禁离基线太远就失去信号——2026-10-27 之前的 56 比当时实测
  低 11pp，删掉整个测试文件 CI 也不会红。

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
2026-10-27   HEAD = M70 docs commit (code 1d78a27; gate 242a1b3)
737 passed / 42 deselected
覆盖率 68.14% 本地 / 65.66% CI（**门禁 65.0**，M70 抬门）   ruff 干净   盲 except Exception 110（原 249）
M68 surplus 同实例归因：`DISABLED_DESTROY_OPERATORS` 默认 `()`，禁用 = 不注册；
small 12 轮默认路径仍为 41.52 / e0377bfd…，原始 medium 360 轮仍为 800.06 / 50bee477…。
15 种子 × 360 轮：禁用 − 启用 = **+0.326，CI [−2.224,+2.876]**，6 胜/8 负/1 平；
启用臂 surplus 354/354 调用/触发，禁用臂 0/0。⇒ 第三分支终局：完整生态位内无净收益，
surplus 调查收口，重启需新的模型语义。
M69 full 臂（bandit ML + tuner，原始 medium，360 轮 × 15 种子）：
锚 ml seed42 = 784.11 / 85bbaa90…，与 M62 重建存档逐位一致。
full 15/15 有效；full−ml = +6.821，CI [−0.166,+13.807]，5 胜/10 负；
full−basic = +6.365，CI [−2.149,+14.880]。CI 跨零 ⇒ 第三分支：
无 `full > ML` 排序证据，basic 已在结构天花板。
深盆地 {99,17,31,123} full−ml +16.82（4/4 变差），浅盆地 +3.18（5 胜/6 负）。
**新缺陷**：同一 full seed42 在 360 轮/固定哈希下重跑得 782.67 与 799.63；
`adaptive_degree` 仍按 elapsed/MAX_RUNTIME，即使 PROGRESS_BASIS=iteration，
iteration=180 时 100s/250s 分别给 0.2234/0.1835 degree。⇒ full 精确罚分不可复现，
需要另立确定性 tuner 修复后再做 full 终审。15 个正式种子 tuner 实际调整均为 0。
M70 修复（**时钟→决策泄漏清零**）：sweep 11 个读取点，1 处坐实（adaptive_degree）、
1 处协议下休眠（warm-up 门：bandit 不构造 selector，实测 trials=0/stop_reason=null）；
`adaptive_degree` 与 `warmup_budget_exhausted` 改按 PROGRESS_BASIS dispatch，
runtime 分支逐位不变（basic/ml/full 120 轮可控时钟：终解/指纹/degree 序列全同）。
确定性：iteration 口径 full 双跑 + full/ml 50ms 延迟注入全部逐位相同；
ml 锚 784.11 / 85bbaa90… 与 M62 存档一致。
M70 重测终审（full 15/15 有效）：full 794.149，full−ml = +6.379，
95% CI [+0.524,+12.235]，4 胜/11 负 ⇒ **分支 (a)：四臂表定稿，full 行是稳定罚分**；
full−basic +5.924 (CI [+0.013,+11.835])；深盆地 +15.66 (4/4 变差)，浅盆地 +3.01 (CI 跨零)；
tuner 基线调整仍 0。
M66 regret N1（可信缓存基线）落地：objective(trust_inventory_cache=True) 公式不变、只跳过
validate→compute_inventory 前置段。等价：0 分歧（2,651+407 次）、末解与指纹逐位相同（120/360 轮 + large）。
速度：objective 6.769 → 1.265 s（−81%）；端到端 medium 120 轮 1.040×、360 轮 1.035×；
**large 0.983×（无收益）**。原因：regret 内层 while 是墙钟预算，省下的时间变成 +9% 候选评估。
⇒ M64 的"13% 墙钟"上限在有时间预算的内层循环里不成立；要兑现须改固定 pass 预算（待批准）。
M67.0 regret 预算澄清（只读）：`time_limit` 是每次调用 10s，不是全局；
50ms objective 人工延迟在 120/360 轮使 delta 调用 1,769,666→894,556 / 2,220,241→1,171,623，
但末解 804.60/800.06 与指纹逐位相同；同一 trust-off 60 轮重跑 pass 数从 276/263/261
漂到 281/266/259，工作量统计可漂、解未漂。⇒ M67-b 不按确定性修复立项。
M67 紧容量变体（存档 CSV 派生，仅改 plant_size_upper_limit.csv）：
`dataset_13_tight_k{110,125,150,200}`、`dataset_3_tight_k{110,125,150,200}`；
k=1.10 可触发 surplus，k=1.50/rate=0.01 为干净控制（0 触发）。
`INVENTORY_HOLDING_COST` 默认 0，默认目标 43.52 逐值不变；fast/delta 评分同步计入
发货使 day..H 库存下降的持有节省。2×2（15 种子 × 360 轮）主判据
`k110/0.01 − k150/0.01` = **+0.278，CI [−2.349,+2.906]，7 胜/8 负**；
只算车辆成本 −0.438（CI 跨零），库存量-天 +71.6（CI 跨零）⇒ surplus 可触发但无净收益，
保留注册 + 标注。范围 medium/360 轮，不外推 large。
M65 数据紧度审计（只读，零改动）：
库存峰值**下界**（放松车队、最早发货）medium 0.1347–0.1445 / large 0.1645–0.1819，
且**等于期初库存**（库存从不累积）；容量 ≈ 5.5–7.4×期初，与 surplus 的 80% 触发差 4.4–5.9 倍
⇒ 该算子在**任何可行解**上都不可能触发（诊断 a+b 成立、c 排除）。
目标函数（ALNS 与 monolithic 同构）只有"车型成本 + 欠载惩罚"，**无库存持有成本**
⇒ 即便数据改紧，移除库存也不改善目标（"可触发但零收益"第二死法）。
对照：车辆载重 p50 0.98、车队仅高于下界 8.6% ⇒ **车队紧、仓库松，差 5–7 倍**。
**可复现性缺口**：文档种子 20260408 重生成 medium 得到不同结构（2 vs 3 种车型）⇒
生成器已与存档数据漂移，**CSV 是唯一可信源**；全量重生成（选项2）会作废所有基线。
处置：保留注册 + 标注；M67 补生态位、M68 做同实例归因，结论均为无净收益。
M64.0 抬门禁 56 → 64（CI 实测 64.56 − 0.5，向下取 0.5 倍数）；旧门禁低 8.6pp，已无信号。
M64 regret 立项（只读）：memo 仍在位且 0 分歧，但精确仿真已 0–2 次/场 ⇒ memo 收益≈0；
当前 regret 6.70 s/次、占墙钟 69%（medium 60 轮）：内联枚举 ≈55%、objective() 18.8%、delta 19.4%；
large 内联 47.5%、可用量查表 679k 次、objective() 仅 4.9%。
**23.6 的方向 2（收紧精确仿真）与方向 3（减 clone）双双失效**（已 0 次仿真、每次调用 1 次 clone）。
口径：固定迭代数下提速收益 = 0；墙钟口径下 regret 减半仅放大 1.4–1.5×（medium），
而逃出窗口需 1.8–2.8× ⇒ **regret 提速是吞吐杠杆，不是质量杠杆**。
下一步待批准 M65 = 每 pass 的 prev_obj 改为增量基线（等价模式，轨迹不变）。
恒空算子机制已查明：surplus 需日利用率 ≥80%（实测 0.14/0.18）、infeasible 需负库存（实测 0）
⇒ 保留 + 标注，不删注册表。
M62 公平终判（medium/dataset_13、360 轮固定、固定 PYTHONHASHSEED、15 种子配对、45/45 有效）：
逃出率（终解≤800）basic 13/15、ml 14/15、ml-trigger 13/15；Fisher ml vs basic p=1.0000（不显著）。
终解均值 788.23/787.77/785.92；逐种子 ml−basic −0.46，CI [−5.07,+4.16]，8 胜 7 负。
"多出的一个逃出"是 s42 救回 + s3 推出（s42 basic 800.06 仅差 0.06），净 +1。
⇒ 判据 A 终局收档：ML 线整体关闭，默认 basic。tuner 维持默认关闭，full ≡ ml。
**窗口教训**：120 轮下 ml−basic 为 +6.71（CI 不含零，"ML 有害"），360 轮下为 −0.46（持平）；
basic 真实逃出在 221–342 轮 ⇒ 120 轮的"ML 有害"是窗口效度问题造成的假象，已作废。
适用范围写死：medium × 360 轮 × bandit 决策器；不外推 large/其他预算/supervised。
M63 后备清单只读盘点完成（未改代码）：
① `surplus_inventory_removal` 与 `infeasible_removal` **100% 空操作**（medium 3 种子 + large 复核）；
   `cross_period_consolidation_repair` 80–93%、`fleet_consolidation_repair` 43–86% 空。
   四项合计吃掉约 17%（7.5–22%）的算子调用，但只占墙钟 1.3% ⇒ 价值在"还回轮盘抽取"，须配对验证。
② **跨期预检缓存否决**：合计占墙钟 0.10%/0.19%/0.07%（medium 三种子）、0.88%（large）。
③ **新发现（最高优先级）**：`regret_based_repair` 占 medium 墙钟 **56.9%**（67.1s/117.8s），
   单次 **5.6 s**，全场仅 12 次调用 ⇒ 每次极贵而非调得太频繁。建议下一个编号做其结构成本分析。
测量口径：签名 1.61 ms/次（medium 498 车），包装器每调用两次 ⇒ 便宜算子耗时高估约 3.2 ms/次，
no-op 率不受影响。
M61 止损判据首轮执行（medium、120 轮固定、固定 PYTHONHASHSEED、15 种子配对）：
① 只读探针：120 轮均值 796.18（3/6 逃出 800）；360 轮均值 788.07（5/6 逃出），
   逃出发生在第 221–342 轮 ⇒ **804.60 不是硬吸引域，是预算不足的落点**。
② 判据 B 触发：tuner−basic 均值 +3.95，95% CI [−1.77, +9.66]（5 赢 9 输）
   ⇒ tuner 默认关闭，full 实测等同 ml。
③ 判据 A 触发且更差：ml−basic 均值 +6.71，95% CI [+1.08, +12.35]（4 赢 10 输）
   ⇒ 该协议下 ML 线收档，默认退回 basic。
④ 边界：A 只覆盖 120 轮；若要外推到 240–360 轮需新协议 + 批准。
协议要求（新结论一律适用）：≥15 种子、固定迭代数、固定 PYTHONHASHSEED、配对比较。
M60 两项机制都实现、都无净收益：
① AcceptanceController（bandit 控退火温度倍率，按 epoch 决策）实测 4/4 变差
   （+3.23/+4.28/+10.60/+19.50），ML_ACCEPT_ENABLED 默认 False。
② tuner 修正（探索只在前 12 次使用、基线最多调整 12 次）：同 4 种子 +0.57 → −5.02，
   扩到 9 种子 +3.04（sd 15.6，4 好 4 坏）。
共同规律：变差的种子正是 basic 找到深盆地（777/783/786）的那几个 ——
任何偏离默认轨迹的改动都会把它们推出去；改动只在"浅盆地"种子上有效。
M59 查出并修复两个真实 bug（ML 由 +9.18 改善到 +1.74）：
① bandit 曾与搜索共用 RNG ⇒ 每次抽样都把搜索随机流后移，"同种子对照"失效；
② 倾斜钩子走 RouletteWheel.__call__ 而非 SegmentedRouletteWheel.__call__，
   跳过分段学分 ⇒ 轮盘权重永远初始均匀。此前 ml 总停在 805.09，basic 能到 787/778。
修复后 s99 的 full 806.55 → 784.96；关倾斜时 ml 与 basic 逐位相同（796.20）。
n-step 信用（窗口 20/衰减 0.9）+ 窄倾斜（0.75–1.33）+ 后期门 0.5：
ml−basic n=9 均值 +1.74（t=0.53，3/9 更好）——持平，不是优势。
仍未达成 full > ML > tuner > basic；tuner 是 full 的主要拖累。
默认已切到 bandit 路径（ML_SELECTOR_KIND="bandit"）。
M58 分层 Thompson bandit（alns/bandit_selector.py）+ 轮盘乘性修正：
bandit 模式不再付监督式的双份开销（147s → 76s，basic 90s）。
判据（medium/120轮/5种子/确定性）：bandit ml−basic +8.28（t=1.33，1/5）、
full−basic +9.71（t=1.71，0/5）；监督式同协议 +9.18/+8.49。两套决策器一样差。
差异集中在 s123/s99：basic 787.23/777.82，两种 ML 停 805 附近 ⇒ 奖励近视，不是估计器问题。
ML_SELECTOR_KIND 默认仍为 supervised；ML_BANDIT_MIN_RUNTIME_SECONDS=90 使短预算退化为 basic。
M57 消除哈希顺序依赖（alns/stable_order.py，3 处集合序进入搜索）：
同种子下 PYTHONHASHSEED=0/1 三模式终解逐位相同（802.09/804.70/804.32）。
从此跨进程对照才有意义。M49–M56 的跨进程消融数字需按此打折。
M57 质量门迟滞 ML_QUALITY_EXIT_MISSES=2：机制生效（接管次数 84→74），
但终解完全不变 —— 迟滞=2 与 =0 配对差一字不差（+9.18/+8.49/−0.69）。对解中性。
确定性代码上 ML 更差：ml−basic +9.18（t=1.92，1/5 更好）、full−basic +8.49、full−ml −0.69。
观测：5 个种子里 3 次落在 804.60，目标函数有强吸引域，均值比较功效低。
下一步方向（待定夺）：ML 决策器改为分层 Thompson 采样 bandit，ML 只做 roulette 乘性修正。
M56 固定迭代数成对试验（medium/120 轮/5 种子×2 哈希种子）：
ml−basic +4.88（t=+2.10，3/10 更好）、full−basic +2.66（t=+0.97）、full−ml −2.21（7/10）。
结论：ML 无正收益且略差；"full 优于 ml"稳定，但两者都没超过 basic。
**复现性缺陷（重要）**：未固定 PYTHONHASHSEED 时同种子同配置两次运行不同
（basic 791.74 vs 795.43），根因是字符串哈希随机化改变集合遍历顺序。
固定后逐位相同。⇒ M49–M55 的跨进程消融结论要打折；同进程内对照仍有效。
实验开关（默认关闭）：PROGRESS_BASIS="iteration"、HARD_ITERATION_BUDGET=True。
分类目标复核：状态特征 9/9 优于算子身份（均值 +0.14），但门槛抖动
（79 次拟合通过 58 次，AUC 0.168–0.944）。下一步候选：质量门迟滞（待批准）。
M55 修复侧目标改为二分类（是否改进）：质量门 = 五块连续时间块的 pooled AUC≥0.55。
修复侧通过质量门 0/6 → 8/10；repair_model_selections == repair_selections（每轮模型决策）。
消融（medium/45s，5 种子）：basic 804.15、ml 801.17（−2.98, t=−1.20）、full 803.59（−0.56, t=−0.51）。
机制修好、效果不显著。降维两项被否决：去参数槽使 AUC 0.72→0.55 / 0.78→0.34；
compact8 状态子集无一致改善 —— 维度不是约束，目标才是。
注意：同种子同配置重跑差 0.30~5.76 分，与要测的效应同量级；下轮应改固定迭代数成对试验。
M53 特征并入算子参数（仅 tuner 侧生效）+ 收益标签 MAD 裁剪（先排除 ±1e6 钳位行）：
medium/45s、3 种子同会话：basic 803.69、ml 804.03、full 803.87；
配对差 +0.34/+0.19/−0.16 —— ML 的边际贡献仍为 0，参数混叠不是 full 落后的主因。
修复模型 R² 从 −431/−253.7 收敛到 −0.05~−26，但 6 次运行 0 次通过 R²≥0；
质量门拒绝是正确行为，真正的问题是修复侧收益在现有特征下不可预测。
M52 给 ML warm-up 加时间上限（ML_WARMUP_BUDGET_FRACTION=0.15）：
medium/45s 只占 8-9%、上限不触发（cost 与改前逐位相同 803.86/804.60）；
large/45s 无上限时 14 轮迭代全在 warm-up、正常搜索 0 轮，加上限后
22 轮（+57%）、终解 2040.32 不变。新缺陷：large 短预算下样本只到 29/30，
低于门槛 40，ml 仍零接管；候选方向见 `修复主线重估` §四。
M51 模型质量门：接管要求时间序留出集 R²≥0（实测破坏 +0.38 放行、修复 −8.5e8 拒绝）。
tuner 的 ranges 全部移除 degree，预算转向 regret k/topN、lookahead_horizon、batch_size。
消融（medium/45s，3 种子，两轮交替）：basic 798.74、tuner 802.33、ml 794.88、
full 803.23；配对差 +3.59/−3.86/+4.49。同种子比 M50：ml 804.84→794.88，
相对 basic 由 +4.3 转为 −3.9，但 ±11.8 波动、不显著。
待讨论：修复侧目标改分类（是否改进，AUC/准确率），或接受"ML 只调破坏侧"。
M50 让 ML 接管真正替换轮盘（NoOpAwareSegmentedRouletteWheel 的 takeover 钩子，
is_ready 无副作用与会消耗冷却的 should_use_ml 分离）。107 轮里 79 轮接管、
模型决策 106 次、随机回退 0；ml 81.9 轮 vs basic 72.8 轮，开销不再拖慢。
消融（medium/45s，5 种子，两轮交替）：basic 802.32、tuner 804.92、ml 804.64、
full 804.34；配对差 +2.60/+2.32/+2.02（|t|≤1.33）。full < ml < tuner 已达成，
但仍略逊 basic。新线索：种子跨度 basic 11.01 vs ml 2.90 / tuner 1.16，
接管压低了搜索多样性。待批准：模型质量门（留出集 R²）、top-k 多样性保留。
tuner 接线正确、仅慢 4%，但 7/11 破坏算子只调 degree（M43 已证无效）。
M49 按预算重标定 ML 门禁：样本门槛 80→40、小实例 120→60、首次拟合 25→12、
重训间隔 80→25、冷却 30→5；并隔离 warm-up 采样对选择反馈的污染。
效果：warm-up 占比 37%→15%，模型选择 2→20（最多 38），冷却拒绝 51→0。
重做消融（medium/60s，5 种子，两轮交替顺序）：basic 792.70 < ml 796.80
< tuner 801.88 < full 805.74；配对差 +4.10/+9.18/+13.04，full 的 t=2.00。
结论：三种增强模式都比 basic 差，full 最差，与设计预期相反；交互项 −0.24。
下一个设计问题：ML 接管目前只是"多一个候选算子"，最多影响 1/11 迭代，
是否让它真正替换选择器需作者定夺。
M48 优化 ML 每次选择的开销：批预测（19 次 transform+predict → 1 次，11.7×）
+ 一次决策内特征复用（键 iteration/id/车辆数/revision）。固定开销
12.1 ms → 约 4.6 ms（−62%）；medium/60s 成本 804.32 → 795.64。
5 种子（medium/90s）消融：full 798.99 < basic 801.65 < tuner 802.68 < ml 803.93，
方向符合"full 应优于单开"，但 t=0.33/1.15/−0.59 均不显著，交互项 −5.96。
限制：ML 每场只接管 1–4 次，被样本门槛与冷却挡住；要得到可用的消融结论，
需先按预算重标定 ML_TAKEOVER_MIN_SAMPLE_SIZE / ML_RETRAIN_INTERVAL /
ML_FALLBACK_COOLDOWN_ITERS。
M47 按作者定夺让 ML 模式独立：ENABLE_ML_OPERATORS 不再隐式构造 tuner，
ML 与 full 是两种不同配置。同时修掉三个让 ML 从未生效的缺陷：
  (a) 四处把 tuner 当 ML 前置条件（含算子注册与 warmup）；
  (b) should_use_ml 要求模型已存在，而训练只在接管后调用（循环依赖）；
  (c) 冷却期内的回退样本继续累计"无改进"，冷却被无限重挂。
修复后 medium/90s 出现 3 轮训练、45 次模型选择、0 次随机回退。
消融首测（medium/90s/3 种子）：tuner +16.26、ML +12.06、full +18.84（均更差），
交互项 −9.48；样本不足，下一步要 5–10 种子配对检验。
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

性能/消融实验必须固定 PYTHONHASHSEED。搜索会遍历字符串键集合，哈希随机化让
同种子同配置两次运行走出不同轨迹（实测 basic 791.74 vs 795.43）。跨进程对照
不固定时测到的是哈希顺序噪声；同一进程内多模式对照不受影响。
local-only/probes/_m56_fixed_iteration_ab.py 自带守卫，新探针请照做。
```

## 9. 失败也要记录

被回退的尝试、负面结果、踩到的坑，同样写进 `local-only/` 记录。
本项目的经验主要来自"先做错、再写下来"。
