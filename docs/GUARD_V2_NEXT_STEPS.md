# Runtime Guard v2：本地缺陷修复与下一轮受控回归

日期：2026-10-27，Asia/Shanghai。**只完成本地修复和实验准备，下一轮 live run 未执行、未授权。** 最新真实结果仍为 2026-10-23 的 1/3；本文不替代或重写历史计划、评分和 trace。

## 1. 证据与修复边界

复核了实际工作区代码、`GUARD_V2_LIVE_REGRESSION_RESULTS.md`、冻结计划、23 条 provider attempt/settlement 及独立合同检查器。Git HEAD 仍为 `50020af4764d177bf631ae5c27ee3029db6e415f`；原有 Guard v2 工作区改动予以保留。本次以最新 live 的 `source_manifest.json` 为源码对照，未重写 ML、CSV pipeline、RAG 检索或报告发布业务。

保护清单覆盖原 pilot、最新 live 目录、两个历史实验工作目录和两份冻结文档，共 68 个文件。修改前后逐文件 SHA-256 核验。历史授权记录未复用，失败任务未删除，原始 grades 未重新写入。

## 2. 根因、局部修复与未知项

| 问题 | 已证实的本地原因 | 本次改动 | 尚未证明 |
|---|---|---|---|
| score_definition 请求 CSV | `RequestInputAction` 只校验枚举；非终结决策默认暴露该工具；Runtime 接受后直接转 awaiting_input | 共享 action policy 推导 `allowed_input_reasons`，模型工具枚举与 Runtime 独立检查一致；明确无 CSV 的 model-card finish；即使嵌入调用方省略 Router，score objective 也无 Skill/委派权限 | 托管模型会否直接正确 finish；未进行新的 live 验证 |
| Writer 两次 invalid_input | Registry 丢弃了 ValidationError 的结构；观察只有错误码；部分自定义校验要求没有出现在 JSON Schema 描述中 | 严格 schema 不变；增加脱敏结构化诊断、模型可见约束描述、最多一次 schema 修正 | **不知道历史两次失败的具体字段**；本地测试中的错误是合成代表案例，不能认作历史错误复现到字段层面 |
| 正式评分合同缺口 | 旧 `grade_task` 不强制 reviewer completion/Planner 返回与同 draft 唯一审核，独立 checker 未进入正式评分 | 新合同 `agent-contract-v2.1`，检查任务完成、score 行为、独立审核生命周期、当前版本与发布顺序 | 确定性引用/哈希检查仍不能证明语义支持、医学正确性或 draft 内容确实改变 |
| 拒绝准入不可诊断 | 旧 budget.rejected 只有账本快照，无待发请求需求量 | 新预算事件记录准确 payload 字节数及估算输入界、输出 cap、总需求、可用量和缺口；不保存 payload | 历史被拒绝请求无法精确重建；未来报告能否在 65,536 内完成仍未知 |

### 输入需求策略

- `score_definition` 直接使用既有模型卡生成 schema-valid 结果，经 Harness 验证后完成；不要求 CSV、不升级、不派发 Skill。
- `missing_upload` 仅在非 score / 非已发布结果读取任务且实际没有上传时允许。
- `invalid_data` 仅在 CSV/信号处理真实报告了数据质量错误、相应有效产物仍缺失时允许。模型写错 Skill 参数不是用户数据缺失。
- `clarify_task` 保留给 Router 确认的任务不明确，或当前检索实际缺乏证据。Router 的无模型补充输入路径保持原样。
- 自定义 backend 即使忽略工具清单，Runtime 仍会拒绝不必要输入请求，返回 `input_not_required` 观察，不进入 awaiting_input。

### 校验反馈与修正上限

Registry 返回并记录 `validation_error`，最多 4 个 issue。每个 issue 包含：错误类别、schema 所属字段路径、预期类型/约束、实际值的类型。额外的未知键被替换为 `<unknown>`；异常消息、异常 context、输入值、原始 prompt 和 Key 均不进入诊断。自定义 validator 的错误通常只能定位到 `value_error` 和声明字段/根对象，不能伪装成更精确的原因。

Writer 首次 schema 拒绝后，Runtime 的 observation 提供 `remaining_corrections=1, retryable=true`。第二次 schema 拒绝后为 `0/false`，任务以 `blocked/skill_validation_exhausted` 结束，不生成草稿、审核或发布。`MAX_WRITER_SCHEMA_CORRECTIONS=1` 是整个 run 的额外修正上限，成功调用或重新委派不会重置。每次模型决策仍消耗原有 decision/request/token 预算；余额不足时可能连这一次修正也无法准入。

Registry 的 `retryable` 表示参数错误本身可修正；Runtime observation/journal 的同名字段进一步反映剩余修正额度。输入校验拒绝发生在 Skill body 之前，不消耗实际 Skill 执行次数；不能将它计成一次已执行业务函数。其他 Skill 保持原来的有界 ReAct 错误处理。

## 3. 新评分合同的明确变化

新结果的 `grades.contract_version` 和 summary 的 `grading_contract_version` 为 **`agent-contract-v2.1`**。任务文件、原始预期标签、数值容差和全局预算均未更改。

1. 要求产出结果的任务必须 completed。安全 blocked 是有效控制结果，但不是完成用户任务。显式负向控制任务（如缺文件、执行预算耗尽）仍按自身合同评估；它们不能冒充报告成功。
2. Score task 保持 simple、零实际 Skill/委派、零 workflow upgrade；观察到不必要的 `request_input` 决策也判为行为合同失败，即使随后 guard 阻止并完成。未解析 provider response 的具体动作仍可能不可见，不能凭错误码猜测。
3. Report 必须有实际成功的独立 Safety Skill、`safety.reviewed`、review artifact、reviewer `agent.completed`，以及匹配同一 delegation step 的 Planner `agent.delegation_completed`。
4. 同一 draft reference 不能再次派发审核。新的 draft reference 可以重新审核；这不是语义内容去重。
5. 发布前必须返回当前有效 review；analysis/evidence/draft/review 必须是当前产物，相关版本、hash、policy 和已发布 safety 字段一致，且通过既有 provenance、schema、citation、数值与 publication 顺序检查。

正式 grader 的加强会影响新旧评分可比性。历史成绩仍使用其保存的原合同；本次只对新离线记录和刻意损坏的测试副本应用新规则，没有回写历史评分。

## 4. Token 账本分析

全部以下数值来自 **10 月 23 日已经保存的真实结果**，不是新请求、模拟 token 或重新推断的 prompt。

| Task | 请求数 | 输入字节界累计 | reservation 累计 | 实际输入 / 输出 | 最终 charged | 释放预留累计 | 字节界 / 实际输入 |
|---|---:|---:|---:|---:|---:|---:|---:|
| score_definition | 1 | 5,284 | 7,332 | 1,234 / 27 | 1,261 | 6,071 | 4.282× |
| csv_analysis | 8 | 72,254 | 88,638 | 19,266 / 552 | 19,818 | 68,820 | 3.750× |
| grounded_signal_report | 14 | 156,434 | 185,106 | 43,382 / 2,778 | 46,160 | 138,946 | 3.606× |

准确 prepared payload 的**长度**可由已准入请求 `reservation - 2,048` 得到，因为所审计 adapter 使用相同不可变 bytes 预留和发送。长度可核验不代表能还原完整请求；JSON/UTF-8 字节数量也不等于服务商 tokenizer 的输入 token。

报告任务两次 Writer：

| 请求 | 输入字节 / 输出 cap | reservation | 实际输入 / 输出 | 释放 | 结算后 charged |
|---|---:|---:|---:|---:|---:|
| 13 | 19,591 / 2,048 | 21,639 | 5,896 / 844 | 14,899 | 39,365 |
| 14 | 19,794 / 2,048 | 21,842 | 5,954 / 841 | 15,047 | 46,160 |

累计 reservation 是先后周转总量，不是同时占用或计费总量。报告输入估算累计比实际输入多 113,052，约高 260.6%。两次无效 Writer 决策实际计费 token 共 13,535；它们的未用预留已释放，没有把这部分差额永久扣留。

下一次被拒绝时余额 19,376，原记录缺少它的完整决策状态、委派任务文本、Writer 参数及 payload 长度，**无法精确离线重建需求或结论性证明修复后的预算可完成性**。Writer 前实际 charged 为 32,625；即使假设第一次写作以原请求用量成功，之后还需要 Writer 返回、Planner 委派、Safety 调用/返回和 Planner 发布决策，不能凭剩余额度宣布可完成。新的反馈和 prompt 也改变未来 payload。

本次不调整 65,536 ceiling、byte admission 公式、2,048 输出 cap、token 结算或未知用量保留规则。新事件 `admission_demand` 将补齐下一轮拒绝证据；实际发出请求的 bytes 与记录长度在 MockTransport 测试中逐字节核对，拒绝后的 HTTP 调用数不增加。

## 5. 本地验证

完整结果和限制见 [本地评估报告](GUARD_V2_LOCAL_EVALUATION.md)。最终 **699 passed、7 warnings**；Ruff lint、261 个文件格式检查及 `pip check` 通过。20 项 packaged evaluation dataset 严格验证通过。实际本地 LightGBM/E5 资产的三任务 scripted integration 均通过新合同、网络请求为零；这是工程集成检查，不是 LLM 成功率或 token 可完成性的证据。

新增 37 个参数展开测试覆盖输入需求、真实 report schema 与隐私、单次修正及委派不重置、伪造审核/发布记录拒绝、准入需求持久化。现有 CSV、API、frontend、取消、超时、重试、恢复与预算测试保留。

## 6. 下一轮冻结配置（未授权）

Provider `openai_compatible`，DashScope 北京 endpoint `https://dashscope.aliyuncs.com/compatible-mode/v1`，profile `dashscope`，model **`qwen3.8-27b`**。同一合成 fixture seed **10042**；同一 LightGBM、E5 和知识索引，retrieval threshold **0.78**。

| 顺序 / Task ID | 精确 user request | 预期 |
|---|---|---|
| 1 / score_definition | `解释 risk score` | 无上传、simple、有效 model-card 结果，无多余输入请求 |
| 2 / csv_analysis | `Analyze this synthetic physiological CSV and return the structured analysis.` | 五个数据 Skill、正确数值和完整 provenance |
| 3 / grounded_signal_report | `Analyze this synthetic CSV and generate a health report. Use indexed EDA and wearable sensor documentation as educational context, cite sources, and avoid diagnosis.` | 完整报告及独立审核返回/当前版本发布；失败保留 |

配置逐字节复用原 `run_config.yaml`：temperature 0、seed null、non-thinking、completion cap 2,048；每任务一次、并发 1、无 fault injection/额外 probe/LLM judge/完整任务重跑。全局上限 **64 requests、1,000,000 charged/reserved tokens、估算 CNY 20**。估价沿用输入 CNY 3/M、输出 CNY 12/M，不宣称已验证当前资费。配置 per_task_token_budget=200000 仍被 Harness 收紧到 **65,536**。

Simple：4 decisions/LLM attempts、30s。Complex：32 decisions/LLM attempts、18 实际 Skills、180s；decision timeout 30s，最多 8 次委派，Data/RAG/Writer/Safety Skill cap 8/4/3/3；保留一次 Safety Skill 余量。额外的 Writer schema 修正上限如上，不扩大这些额度。

本轮与 10 月 23 日差异仅为：输入需求策略及模型视图、Writer 约束描述/诊断/修正上限、新 canonical grading、预算需求元数据，以及新的运行标识和目录。176 个源码/测试/配置文件、10 个资产的指纹见新 `preflight.json`；176 个文件中有 24 个相对最新 live 新增或改变。原任务 hash、fixture、资产和 Harness 限制不变。

准备工作目录：

```text
/Users/rensibo/Smart Health/.live_eval_work/guard_v2_local_fixes_20261027T111638+0800
```

保留给未来一次运行的输出目录（目前不存在；时间戳是准备时间，实际运行时间另记）：

```text
/Users/rensibo/Smart Health/smart_health_agent/evaluation/live_runs/20261027T111638+0800-guard-v2-local-fixes-regression
```

`authorization.pending.json` 为 **approved=false**，绑定本计划、配置、preflight 和脚本 hash。没有 `authorization.json` 或 start marker。原有两轮授权均不覆盖本轮。只有收到新的明确授权后，才记录真实批准文本/时间及完全相同的 experiment 对象；不要自动把 pending 改成 true。

### 无付费调用的可重复命令

```bash
cd '/Users/rensibo/Smart Health/smart_health_agent'
WORK='/Users/rensibo/Smart Health/.live_eval_work/guard_v2_local_fixes_20261027T111638+0800'
make check PYTHON=.venv/bin/python
.venv/bin/python "$WORK/preflight.py"
CHECK_DIR="$(mktemp -d "$WORK/recheck.XXXXXX")"
.venv/bin/python -m smarthealth.evaluation.live_llm.runner \
  --dry-run --config "$WORK/config.yaml" \
  --task score_definition --task csv_analysis --task grounded_signal_report \
  --output "$CHECK_DIR/dry-run"
.venv/bin/python "$WORK/offline_checks.py" --output "$CHECK_DIR/offline-integration"
.venv/bin/python "$WORK/budget_analysis.py" --output "$CHECK_DIR/budget_analysis.json"
```

所有检查产生新文件，拒绝覆盖既有输出。preflight 只做本地验证；dry-run 不加载 Key、模型或 provider。offline_checks 使用真实本地资产和明确标注的 scripted backend，禁止构造 provider。

### 仅在未来明确授权后执行一次的命令

以下是计划，**本次未执行**。只有新授权匹配冻结计划后才创建 `authorization.json`：

```bash
cd '/Users/rensibo/Smart Health/smart_health_agent'
WORK='/Users/rensibo/Smart Health/.live_eval_work/guard_v2_local_fixes_20261027T111638+0800'
umask 077
set -C
.venv/bin/python "$WORK/run_once.py" \
  --live --authorization "$WORK/authorization.json" \
  > "$WORK/live-console.log" 2>&1
```

启动器检查新授权、全部冻结 hash、历史证据和输出目录，之后才读取私有 Key；固定原模型/endpoint/本地资产，创建独占 start marker。失败退出也不得删除 marker 或自动重跑。配置或源码变化需要重新审查并生成新计划，不应绕过指纹检查。结果仅保存本地，不上传 GitHub。

## 7. 下一轮分析规则与剩余不确定性

保留三项任务和全部失败。逐项报告原始新合同失败、实际 Skill start/completion 与 pre-execution rejection、终态/发布、HTTP 次数、逐请求预留/需求/释放/已知及未知 usage、latency 和固定价格估算。明确区分模型不再提出坏动作、Guard 阻止坏动作、或任务未到达原失败阶段。

已有本地修复足以准备下一轮**诊断性**受控运行；没有证据证明新 live 成功率已提升。Safety 的真实 reviewer completion、Writer 能否利用反馈、现有预算能否支持完整报告仍待新的明确授权实验。既有无效 provider reply 丢失 usage 可见性、按 artifact reference 而非语义内容防重复、无医学语义真值 benchmark 等局限仍在；本次没有顺带扩展这些机制。
