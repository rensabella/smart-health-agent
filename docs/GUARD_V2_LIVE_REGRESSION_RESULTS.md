# Runtime Guard v2：已授权 live regression 结果

执行时间：**2026-10-23 16:26:17–16:27:36，Asia/Shanghai**。

本次三个任务仍然是 **1/3 通过**。Guard v2 的预算准入在真实运行中阻止了一次未获预算的模型请求，但没有带来更多任务完成。报告任务在 Writer 阶段失败，尚未进入 Safety 审核，因此本次不能验证 reviewer completion 或重复审核问题是否已经解决。

共 **23 次真实 HTTP 请求，全部 HTTP 200**，均返回可解析的决策和完整 usage。输入 **63,882**、输出 **3,357**，合计 **67,239 tokens**；按冻结的原 pilot 价格估算为 **CNY 0.231930**。这是固定价格估算，不是账单金额。

本报告记录实际结果。[执行前计划](GUARD_V2_LIVE_REGRESSION_PLAN.md) 保持原始字节及审批绑定哈希，其“待授权”状态是执行前记录。原始 pilot、冻结计划、原始评分和失败记录均未改写。本次没有修改生产代码、增加功能、删除失败任务、放宽评分、追加付费调用或上传 GitHub。

## 1. 授权与实验一致性

用户在完整计划后明确回复“授权”。新的 `authorization.json` 绑定模型、任务、预算、配置、代码/资产清单、实验脚本及计划哈希；启动器创建独占 start marker 后仅执行一次。退出码 **1** 表示存在任务合同失败，整个实验及证据写入已完成，不表示需要自动重跑。

| 项目 | 实际配置 |
|---|---|
| Provider / profile | `openai_compatible` / `dashscope` |
| Endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Model | `qwen3.8-27b` |
| 任务顺序 | `score_definition`、`csv_analysis`、`grounded_signal_report` |
| 重复 / 并发 | 每项 1 次 / 1 |
| generation | temperature 0、seed null、completion cap 2,048、non-thinking |
| 实际 per-run token 上限 | **65,536**；配置的 200,000 不会放大 Harness 上限 |
| 全局上限 | 64 次 admitted requests；1,000,000 charged/reserved tokens；估算 CNY 20 |
| 原价格估算 | 输入 CNY 3 / M，输出 CNY 12 / M；未宣称重新验证了当前资费 |
| 输入 / 资产 | 与原 pilot 相同的任务文本、fixture seed 10042、合成 CSV、LightGBM bundle、E5 索引、检索阈值 0.78 |
| 检查结果 | 173 个冻结源码/测试/配置文件、10 个资产文件及冻结脚本/计划均匹配；原始 15 个证据文件未变化 |

完整配置和哈希见冻结计划、`source_manifest.json` 及 `verification.json`。批准前的本地完整检查为 **662 passed、7 warnings**，Ruff 和依赖检查通过。执行前再次进行本地 preflight，未发现漂移，未重新安装依赖或训练模型。

新结果目录：

```text
/Users/rensibo/Smart Health/smart_health_agent/evaluation/live_runs/20261023T160007+0800-guard-v2-qwen38-regression
```

目录时间戳对应计划准备时间，实际调用时间见本报告首行及 receipt；没有复用原始 pilot 目录。

## 2. 原始 pilot 与新运行对比

保留原评测器的判定，再应用事先准备的独立合同核验。两者在本次通过/失败结果上相同。**有效终态和控制不变量通过，不等于任务合同通过。**

| Task | 原始 pilot | Guard v2 本次 | 本次精确合同失败 | 本次发布 |
|---|---|---|---|---|
| `score_definition` | Fail；升级到 report 后 `blocked/safety_unavailable` | **Fail**；没有升级，但错误要求上传 CSV，`awaiting_input` | `expected_status`；期望 completed 的 score definition | 无 |
| `csv_analysis` | Pass；completed，数值/来源/发布校验通过 | **Pass**；completed，同样通过数值/来源/发布校验 | 无 | 结构化 analysis |
| `grounded_signal_report` | Fail；一次成功 Safety review 后重复请求审核，最终 `blocked/token_budget_exceeded`，未发布 | **Fail**；两次 Writer 参数 schema 拒绝后，在下一次模型请求前 `blocked/token_budget_exceeded` | `expected_status`、`required_skills`、`required_artifacts` | 无；连 draft/review 都未生成 |

| Task | 模型 HTTP 次数：旧 → 新 | 输入 / 输出 tokens：旧 → 新 | 任务 latency：旧 → 新 | 估算 CNY：旧 → 新 |
|---|---:|---|---|---|
| `score_definition` | 1 → 1 | 1,233 / 28 → 1,234 / 27 | 970.972 → 1,054.513 ms | 0.004035 → 0.004026 |
| `csv_analysis` | 8 → 8 | 18,308 / 510 → 19,266 / 552 | 11,828.449 → 17,058.412 ms | 0.061044 → 0.064422 |
| `grounded_signal_report` | 18 → 14 | 63,585 / 3,013 → 43,382 / 2,778 | 65,312.018 → 59,420.666 ms | 0.226911 → 0.163482 |
| 合计 | **27 → 23** | **83,126 / 3,551 → 63,882 / 3,357** | 实验墙钟 79.964 → 79.125 s，含本地准备等开销 | **0.291990 → 0.231930** |

次数、tokens 和费用减少发生在不同的失败轨迹上；新报告任务更早结束，不能据此宣称推理效率提升。CSV 的本次数值结果仍是 `anomaly`、10 个窗口、`risk_score=0.9983188764750849`；这是合成模型分数，不是临床风险概率。CSV 数值校验采用原标准：分数 `rel_tol=abs_tol=1e-9`，标签与窗口数精确匹配。

## 3. 实际 Skill 执行、拒绝与终态

### score_definition

Run ID：`6a7ec4f9678846c4981cb4684b5982fd`。

- 初始和最终 route 都是 simple / score_definition，`workflow_upgrades=0`。
- event **12** 是 `request_input`；event **13** 为 `run.awaiting_input`。返回的运行提示要求上传 CSV，但任务只是解释 risk score，不需要上传文件。
- 实际 Skill 执行 **0**，delegation **0**，runtime action rejection **0**，provider retry **0**；没有 publication 或 final artifact。
- 一次调用准入、一次完整结算；终态后没有继续工作。

分类：**模型选择了错误但当前政策允许的控制动作**。旧的升级行为本次未观察到，不能说它被 runtime 拦截过一次，也不能说简单任务的 over-planning 已解决。任务在合法终态提前停止，仍不满足用户目标。

### csv_analysis

Run ID：`32d848aea213434f9f35c24a1e1ec525`。

五次实际执行均成功：

```text
csv_analysis → signal_processing → feature_extraction
→ anomaly_prediction → feature_explanation
```

一次 Planner delegation。Data Analysis 在 event **48** 完成，Planner 在 **49** 收到子任务结果，随后继续执行；这表明本次在线执行没有把子任务完成误当作父任务完成。event **55** 通过 publication 检查，**56** 为 `run.completed`。所有 8 次模型请求准入和结算一致，无 action/schema 拒绝、无重试、无未知用量。

分类：**原有成功路径保持通过**。本次没有进行进程崩溃/重启，因此不把这个结果当成 recovery 的 live 验证。

### grounded_signal_report

Run ID：`8765acdb4a424c22a588f3d1b348e11c`。

实际进入 Skill body 的 **7 次**执行全部成功：五个分析 Skill，以及两次 `rag_retrieval`。两个 evidence artifact 保留在 manifests 中。报告写作两次调用被 registry 在输入校验时拒绝，**没有执行 `report_generation.execute`**，也没有产出 draft。未执行 Safety，未产生 review、final report 或 report chunk。

| 证据 | 事件 / call ID | 含义 |
|---|---|---|
| Data 子任务完成 / 返回 | runtime events 48 / 49 | 父任务继续执行 |
| RAG 子任务完成 / 返回 | runtime events 71 / 72 | 父任务继续执行 |
| 第一次 Writer 拒绝 | runtime event 83；registry `skill.rejected`，call `86bc38c132b040299f8a7dc3c5ff95ea` | `invalid_input`；输入不符合 Skill schema；无 output refs |
| 第二次 Writer 拒绝 | runtime event 89；registry `skill.rejected`，call `4fd233e5d6bd4bca97a4c6dd449f9ee1` | 同类 `invalid_input`；无 output refs |
| 下一次请求未获准入 | runtime event **90** `budget.rejected` | 此时 admitted/settled 都为 14，没有第 15 次 provider dispatch |
| 决策失败 / 任务终止 | events 91 / 92 | `report_writer` 的下一次 decision attempt 失败；父任务 blocked |

这里必须区分三种计数：9 个模型 Skill proposal / 9 次交给 Registry 的尝试，**7 次 actual Skill body execution + 2 次 pre-execution schema rejection**。原评测器的 `physical_tool_calls=9` 包含 `skill.rejected` 终结事件，并不表示执行了九次业务函数；runtime `tool_calls=7` 与七个 `skill.started` 一致。本文使用真实 start/completion 事件解释执行次数，未修改原统计字段。

没有 `agent.action_rejected` 事件：两次拒绝来自 Registry schema validation，一次来自预算准入。不能因此将“rejected actions”汇总成零。14 次 provider response 都符合通用 DecisionReply 协议，但其中两次的 Skill 参数不符合该 Skill 的输入 schema；HTTP 200 或可解析 JSON 不代表端到端成功。

分类：**Writer 参数合同失败，未成功修复，随后被保守预算控制终止**。已有日志只保留 `invalid_input`，不含具体 Pydantic 字段路径或原始参数；无法确定是哪个字段、长度或类型出错，不能猜测为某个具体 schema 缺陷。两次失败具有相同错误代码，无法据此断言参数完全相同。

## 4. 逐请求预算结算核验

独立核验脚本逐项比较了 provider attempt、`budget.admitted`、`budget.settled`、最终 run snapshot 和全局 ledger，未使用“测试通过”替代真实运行核验。完整的 **23 条逐请求记录**在 `token_settlements.json`，包含 input byte bound、output cap、reservation、实际输入/输出、结算、释放额度、event ID、latency 和估算费用。

| Task | 准入 / 结算 / HTTP | 累计 reservation | Run 内最高计入预算量（含在途预留） | 最终已结算 | 剩余额度 | 未知 / 未结算 |
|---|---:|---:|---:|---:|---:|---:|
| score_definition | 1 / 1 / 1 | 7,332 | 7,332 | 1,261 | 64,275 | 0 / 0 |
| csv_analysis | 8 / 8 / 8 | 88,638 | 28,152 | 19,818 | 45,718 | 0 / 0 |
| grounded_signal_report | 14 / 14 / 14 | 185,106 | 61,207 | 46,160 | 19,376 | 0 / 0 |

累计 reservation 是先后各请求预留额的和，不是同时占用，也不是实际收费 token 总数。每次完整响应都释放未使用预留，保留实际使用量。三项最终 `usage_complete=true`、`bound_exceeded=false`；全局 admitted/settled/HTTP 均为 **23**，in-flight、retained unknown、unknown attempts 均为 **0**。

报告任务在 **46,160 / 65,536** 已用、剩余 **19,376** 时拒绝下一次准入。`decision_count=15`，但 `llm_requests=14`。event 90 之前的最后一次 HTTP 已在 `08:27:36.478705Z` 完成；拒绝发生在 `08:27:36.489155Z`。拒绝后没有 `agent.thinking`、新的 admitted event、provider attempt、Skill dispatch 或 publication。这提供了本次“拒绝准入没有发出请求”的直接计数和时间序列证据。

拒绝请求本身的 byte length / reservation demand 没有被持久化，所以只报告已观察到的余额与拒绝事件，**不伪造精确申请量**。从所审计的准入代码可以判断，完整 payload byte bound 加 2,048 输出预留不能装入余额。它不是“已经实际用了 65,536 tokens”，也不是 provider 的 context-length 错误。

原始 report 在第 18 次响应后才发现用量达到 66,598，超过 65,536 上限。本次没有该 overrun，准入控制生效；但是为了不超预算而终止未完成工作，属于 **blocked without completion**，不是任务成功。

本次没有 provider error、timeout、取消、transient retry 或未知 usage；这些路径仍只有本地测试证据。两个 Skill schema 错误发生在已成功计量的 provider response 之后，没有丢失本次 token usage。原先发现的“adapter 拒绝响应时丢失 usage”限制在此运行中未触发。

## 5. 原失败模式是否消失

| 原失败模式 | 本次观察 | 能得出的结论 |
|---|---|---|
| score definition 被升级为 report | 无升级，也无被拒绝的升级尝试；却出现不必要的 upload request | 旧升级行为在这个样本里未观察到；简单任务依然失败，不足以宣称 over-planning 已消除 |
| Safety 重复审核同一 draft、未正常 finish | Writer 未生成 draft；Safety 从未开始 | **未覆盖**，不能将零次重复审核当作修复验证 |
| 收到响应后才发现超过 per-run token 上限 | 第 15 次决策在 provider 前被拒绝，已用 46,160，未超支 | **控制生效，但任务被阻断而未完成** |
| 子任务完成被误判为父任务完成 | Data/RAG 的 child-completion 后父任务继续运行 | 正常执行中的层级完成行为有证据；未进行 crash recovery live 实验 |
| 终态后继续执行 | 三个终态之后均无工作事件 | 本次三条路径符合终态静止要求，不推广到未覆盖路径 |

独立 supplemental checker 的通过/失败与主评分一致：Fail / Pass / Fail。报告任务的 reviewer 生命周期及一次审核断言未满足，是因为必需阶段缺失，**不是观察到了重复审核**。`revision_content_change_verified=null` 在这里表示没有 draft 可核验，并非发现了内容不变的新版本。

**本次任务通过数没有增加。** 三项任务各一次，不支持统计显著性、模型总体能力、成本优化或延迟改善的结论。没有为了获得更好的成绩重跑失败任务，剩余请求/费用额度也没有用于追加实验。

## 6. 失败分类后的改进建议（未实施）

1. **优先检查 score_definition 的控制动作约束。** 已观察到确定不依赖 CSV 的任务仍可 `request_input(missing_upload)`。下一次代码变更应先加入这个具体轨迹的离线回归，核对允许的 input reasons、模型可见的完成条件，再决定如何收窄 action policy。不能用增加预算解决这个问题。
2. **先补足 Writer schema 错误的最小诊断证据。** 当前只知道两次 `invalid_input`，不知道哪个字段失败。若实施后续改动，优先考虑脱敏的字段路径/错误类型及可供模型修正的有限反馈，而不是保存原始生理数据、完整 prompt 或猜测修改 schema。先用本地 fixture 证明修复机制有效，保留严格校验。
3. **再评估长报告的预算可完成性。** Writer 的两个请求各预留约 21k，实际输入约 5.9k；保守 byte admission 与重复失败共同消耗了可继续执行的空间。可在离线准确重建的请求上分析上下文和估算误差，再讨论压缩或预算估计方式。此次不扩大 65,536 ceiling、不换模型、不弱化发布条件。

Safety reviewer 的真实完成验证需要后续实验先到达该阶段。本轮授权的一次实验已结束；任何新的 paid run 都应有新的范围、目录和明确授权。

## 7. 文件、复核与复现

新结果目录保留：

```text
summary.json                   原 evaluator 汇总，未改写
results.jsonl                  三项完整结果和受限 trace，含两个失败任务
report.md                      原 evaluator 生成的报告，未改写
REPORT.zh-CN.md                 本次中文诊断报告
run_config.yaml                原实验配置的精确副本
approved_plan.md               已授权计划的冻结副本
authorization_receipt.json      本轮真实授权，不复用旧 receipt
started.json                   独占执行记录
completion_receipt.json         退出码 1 与完成时间、原始证据保护检查
source_manifest.json           preflight 指纹、任务合同、有效预算
comparison.json                冻结 supplemental checker 的原始输出
token_settlements.json          23 条逐请求 reservation/settlement 明细
verification.json              独立对账及原始证据校验
evidence_manifest.json          新证据文件的 SHA-256 清单
```

授权及分析工作目录：

```text
/Users/rensibo/Smart Health/.live_eval_work/guard_v2_regression_20261023T160007+0800
```

实际只运行过一次的命令如下；start marker 和已存在的结果目录会拒绝重复启动。此处是执行记录，不是追加调用授权：

```bash
cd '/Users/rensibo/Smart Health/smart_health_agent'
WORK='/Users/rensibo/Smart Health/.live_eval_work/guard_v2_regression_20261023T160007+0800'
umask 077
set -C
.venv/bin/python "$WORK/run_once.py" \
  --live --authorization "$WORK/authorization.json" \
  > "$WORK/live-console.log" 2>&1
```

只读对比可用新分析输出目录复核，不触发付费请求：

```bash
cd '/Users/rensibo/Smart Health/smart_health_agent'
WORK='/Users/rensibo/Smart Health/.live_eval_work/guard_v2_regression_20261023T160007+0800'
CHECK_DIR="$(mktemp -d "$WORK/postrun-recheck.XXXXXX")"
.venv/bin/python "$WORK/contract_audit.py" \
  --new '/Users/rensibo/Smart Health/smart_health_agent/evaluation/live_runs/20261023T160007+0800-guard-v2-qwen38-regression' \
  --output "$CHECK_DIR/comparison.json"
```

原始 pilot 仍在 `evaluation/live_runs/20261015-bailian-qwen38-initial`。新的 verification 同时确认原始 15 个文件未变，且新 `results.jsonl` 与 `summary.json.records` 完全对应。结果、Key、配置和账本均未上传任何外部存储；授权范围内的 provider 调用只使用既有受限决策输入。
