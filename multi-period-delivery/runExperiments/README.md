# runExperiments

概述
- 本目录包含用于批量运行 ALNS 实验、聚合单次运行结果并绘图 / 统计检验的脚本。目标是可复现地比较 ALNS 各变体与 Gurobi monolithic 的性能与解质量，并输出统一格式以便后续统计分析。

目录脚本说明
1. run_experiments.py
- 作用：对指定实例集合批量运行 ALNS 多个变体（variants），支持多随机种子并发运行、每次运行写入 per-run 元数据与结果。
- 主要输入参数（CLI）：
  - `--instances <path>`：实例路径（文件或目录；通常指 `datasets/multiple-periods/` 下的 `dataset_N` 目录或文件）
  - `--variants <list>`：逗号分隔的变体名，例如 `basic,tuner,ml,full`
  - `--seeds <list>`：逗号分隔随机种子，例如 `1,2,3`
  - `--max-runtime`：单次运行最大运行时间（秒）
  - `--out-dir`：单次运行输出根目录（默认 `OutPut-ALNS/experiments`）
- 约定输出（每次 run 会生成一个唯一目录）：
  - `run_metadata.json`：包含实例名、variant、seed、命令行、开始/结束时间等元信息
  - `result.json`：包含最终目标值、运行时间、迭代次数、best_solution（若配置）、其他指标
  - `log.txt`：运行日志（stdout/stderr 合并）
  - 可选：`gurobi_best_bound.json`（若在该 run 中调用 monolithic 求解并保存 BestBound）
- 使用示例：
```bash
# 运行 basic 和 full 两个变体，3 个种子，时间上限 600s
python runExperiments/run_experiments.py --instances datasets/multiple-periods/ --variants basic,full --seeds 1,2,3 --max-runtime 600 --out-dir OutPut-ALNS/experiments_demo
```

2. aggregator.py
- 作用：遍历 `run_experiments.py` 的输出目录，合并每次 run 的 `result.json` 与 `run_metadata.json`，生成聚合表格用于统计与绘图。
- 主要输入参数：
  - `--runs-dir <path>`：实验输出根目录（与 `run_experiments` 写入的 `out-dir` 对应）
  - `--out-csv <path>`：聚合结果 CSV 保存路径
- 生成文件（示例列）：
  - `aggregated_results.csv`：每行为一次 run，列包括 `run_id, instance, variant, seed, objective, runtime_s, iterations, best_bound (nullable), gap_percent (nullable), run_ok, notes`
  - `aggregated_stats.csv`：针对每组（instance, variant）计算的基本统计量（mean, std, median, min, max）
- 注意：aggregator 期望每个 run 的 `result.json` 中包含统一字段名（见脚本顶部注释）。若修改 per-run 字段，请同步更新 aggregator。

3. plot_and_stats.py
- 作用：基于 `aggregated_results.csv` 绘制箱线图、Dolan–Moré 性能剖面，并做成对统计检验（配对 Wilcoxon signed-rank）与多重比较校正（Holm / Bonferroni）。
- 主要输入参数：
  - `--aggregated-csv <path>`
  - `--variants <list>`：用于比较的 variant 子集（参考 `run_experiments` 中使用的名字）
  - `--output-dir <path>`：图 / 检验结果保存目录
- 输出（示例）：
  - `boxplot_<variants>.png`
  - `dolan_more_profile_<variants>.png`
  - `stats_wilcoxon.csv`（成对检验 p 值与校正后 p 值）
  - `summary_stats.json`
- 统计说明：
  - 使用配对 Wilcoxon 检验评估同一实例上两个变体的解质量差异，随后使用 Holm 方法做多重比较校正。
  - 建议在报告中同时给出效应方向（哪个方法更优）与 p 值校正结果。

评价指标与 Gap 计算
- 主要评价：
  - 运行时间（runtime，秒）
  - 迭代次数（iterations）
  - 最终目标值（objective，ALNS 输出）
  - 与 Gurobi monolithic 的 BestBound 对比的 Gap（适用于小/中规模，monolithic 能得到有效 bound）
- Gap 计算（目标为最小化）：若存在 Gurobi 提供的 BestBound，按下式计算相对 Gap（%）：
$$
Gap = \frac{f_{ALNS} - BestBound}{|BestBound|} \times 100\%
$$
- 说明：
  - 若 $BestBound=0$，请改用绝对差或在报告中注明特殊处理。
  - 若 BestBound 来自 Gurobi 的最优解（optimal），Gap 表征相对误差；若为下界则表征相对差距。

输出格式规范（建议）
- `aggregated_results.csv` 建议列（至少包含）：
  - `run_id, instance, variant, seed, objective, runtime_s, iterations, best_bound, gap_percent, run_ok, notes`
- `result.json`（per-run）应包含上述字段的原始值，便于追溯与逐 run 检查。

依赖
- Python >= 3.8
- 必需包（示例安装命令）：
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn tqdm
```
- 并行运行依赖：脚本内部采用 `multiprocessing`（标准库），可按需安装 `joblib`。

实验建议与样例设计
- 小/中规模实例：
  - 与 monolithic（Gurobi）对比：先用 monolithic 求解以获得 BestBound（或最优解），再用 ALNS 各变体跑多次（不同 seed）。
  - 时间预算建议：例如 `600s`（或与 monolithic 求解时间相同）。
  - 重复次数：对每个 instance-variant 推荐至少 5 次 seed。
- 大规模实例：
  - 仅比较 ALNS 变体（basic / full / tuner / ml 等），重点考察扩展性、收敛速率与质量提升。
- 度量与统计：
  - 使用 per-instance 平均 objective 做总体比较，并用 Wilcoxon 做配对检验（paired by instance）。
  - 使用 Dolan–Moré 性能剖面评估在给定时间预算下达到解质量的能力。

变体定义（脚本约定）
- `basic`：禁用参数自动调优（ParamAutoTuner）、禁用 ML 算子；使用默认操作器与参数
- `tuner`：启用参数自动调优，但禁用 ML 算子
- `ml`：启用 ML 算子，但禁用参数自动调优
- `full`：启用参数自动调优与 ML 算子（完整版本）

工程与调试建议
- 保留 `scripts/smoke_initial_solution.py` 在 `scripts/`（不要移动）以便快速 smoke test 初始解生成。
- 确保每次 run 的 `result.json` 包含 `run_ok` 或类似标志，便于 aggregator 过滤失败 run。
- 若要增加新的 variant，请在 `run_experiments.py` 中增加映射到 `ALNSCode` 的配置覆盖字典，并在 README 中同步记录。

示例工作流程（推荐）
1. 使用 monolithic（Gurobi）在小/中规模实例上运行，保存 BestBound。
2. 使用 `run_experiments.py` 对所有 variants 与 seeds 进行批量实验。
3. 使用 `aggregator.py` 聚合结果生成 `aggregated_results.csv`。
4. 使用 `plot_and_stats.py` 绘图并做统计检验，生成结果图与统计表。
5. 将结果汇总到报告或 notebook 中，附上 Dolan–Moré 与 Wilcoxon 结果。

注意事项
- 统一时间预算、随机种子与实例集，以确保对比公平。
- 若 monolithic 在某些实例上超时并没有有效下界，`gap` 字段会为 `null`，应在分析中单独说明或剔除。
- 确保 `run_experiments` 的输出命名规则与 `aggregator` 预期匹配，避免聚合错误。

维护与联系方式
- 脚本作者 / 维护者：项目仓库维护者（参见仓库 README 或最近的 Git 提交记录）。
