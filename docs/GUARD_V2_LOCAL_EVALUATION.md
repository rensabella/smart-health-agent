# Runtime Guard v2 本地修复验证报告

日期：2026-10-27。**本次没有 paid LLM 请求，没有新的 live 分数。**

## 1. 三类证据分别解读

| 证据 | 实际结果 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 最新真实 live，2026-10-23，已冻结 | **1/3**；23 HTTP；67,239 tokens；固定价格估算 CNY 0.231930 | 旧代码当时的真实运行结果 | 本次修改后的模型行为 |
| 本次完整自动化测试 | **699 passed，7 warnings，32.61s** | 输入政策、错误修正边界、严格 schema、隐私、发布合同、预算及已有功能的本地回归 | 真实 LLM 任务成功率、医学正确性 |
| 本次真实本地 ML/RAG + scripted decisions | 三个 infrastructure cases 全部通过；**0 provider 网络请求** | 当前 assets、Runtime、Skills、canonical grader 的端到端接线与合同 | LLM 会选择同样动作、Safety 语义判断可靠、实际 token 预算足够 |

本地 scripted 检查不可与历史 live 的 1/3 合并计算，也不是“新 live 3/3”。新 live 状态：**not executed / awaiting fresh explicit authorization**。

## 2. 自动化检查

最终命令：

```bash
cd '/Users/rensibo/Smart Health/smart_health_agent'
make check PYTHON=.venv/bin/python
```

- Pytest：699 passed；相对之前 662 项新增 37 个参数展开的行为测试。
- Ruff lint：通过。
- Ruff format --check：261 个文件通过。
- pip check：`No broken requirements found`。
- 7 条警告来自既有 Starlette/httpx、AnyIO、Streamlit/NumPy 弃用提示；未隐藏警告、改变依赖或调整测试期望来掩盖缺陷。
- Packaged benchmark：20 项任务严格验证通过；三项选择、任务 hash、合成 fixture 和独立参考预测保持一致。

关键新回归：

| 文件 | 行为证据 |
|---|---|
| `tests/test_guard_v2_input_requirements.py` | 无文件 score 完成；拒绝不必要的三种 input reason；真实缺文件、坏 CSV、模糊目标保持可请求输入；参数错误不能转嫁为 CSV 问题；阻止升级/Skill；省略 Router 也无工具授权 |
| `tests/test_report_validation_feedback.py` | 真实 GenerationInput 的缺字段、类型、长度、自定义文本校验、revision 配对及额外键拒绝；无副作用；不泄露值/Key/未知字段名；有效修正真正执行 ReportSkill；一次修正上限、再次失败安全停止、委派不重置 |
| `tests/test_agent_grading_contract.py` | 对明确标注的本地记录做 mutation：多余输入、升级、Skill、缺 Safety、缺 reviewer completion/Planner return、错误 delegation ID、重复审核、提前发布、陈旧 draft、错误 hash 和缺 artifact 都不能通过 |
| `tests/test_runtime_guard_v2.py` 新用例与断言 | 准入元数据与 MockTransport 实际 bytes 一致；拒绝需求/余额/缺口持久化；拒绝后无 HTTP/Skill/发布；保留原结算及未知 usage 测试 |

`tests/test_agent_backends.py` 与 `tests/test_agent_routing.py` 的工具清单期望仅调整为不再给“不缺输入”的状态暴露 request_input；新测试同时保证真实缺输入的工具仍可用。严格 GenerationInput 校验规则未放宽。

## 3. 本地资产集成

`offline_checks.py` 通过 production `execute_task`，显式设置 `mode=infrastructure_test` 和 `ScriptedDecisionBackend`；provider 构造函数被本地断言阻断。使用原来的 LightGBM bundle、E5 ONNX、知识索引、0.78 检索阈值和 seed 10042 合成输入。

| Task | 本地合同 | Skill body 执行次数 | 最终结果 |
|---|---|---:|---|
| score_definition | 通过 | 0 | 模型卡解释；无 CSV；completed |
| csv_analysis | 通过 | 5 | anomaly，10 windows，risk_score **0.9983188764750849** |
| grounded_signal_report | 通过 | 8 | 当前 draft 的独立 review 返回 Planner 后发布 |

这里 Safety 的语义 flags 由固定脚本给出，不是模型理解或医疗审查。Scripted backend 不执行 live token admission，所以本地完成不证明预算可完成性。预算控制另由 MockTransport/账本回归验证。

## 4. 评分合同与历史证据

新合同版本为 `agent-contract-v2.1`，详见 [根因与下一轮计划](GUARD_V2_NEXT_STEPS.md)。特别是 score 任务出现可观察的不必要 request_input 决策即违反行为合同；报告必须完成整个审核返回/当前版本发布链。历史 grades 保持原始字节，未用新 grader 覆盖它们。

历史文件保护包含原 pilot、最新 regression、两个历史准备目录及其报告计划，共 **68 个文件**。最终 preservation 检查确认未修改；本轮本地结果和未来待授权配置使用独立目录。

## 5. 保存的本地证据与复现

工作目录：

```text
/Users/rensibo/Smart Health/.live_eval_work/guard_v2_local_fixes_20261027T111638+0800
```

| 文件 | 内容 |
|---|---|
| `final_checks.log` | 最终完整 pytest / lint / format / dependency 输出 |
| `preservation_before.json`、`verification.json` | 历史文件 hash 与最终复核 |
| `preflight.json` | 176 个 source/test/config、10 个资产、20 项数据集、配置/任务/fixture/模型指纹 |
| `offline-integration/summary.json`、`report.md` | 明确标注 infrastructure_test 的三个本地记录 |
| `budget_analysis.json` | 保存的 23 次真实调用的只读账本对账；未重构未保存的请求 |
| `dry-run/` | 未执行的三项任务、null grades、零准入请求 |
| `authorization.pending.json` | approved=false 的下一轮范围；不构成授权 |
| `authorization_gate_check.json` | 未授权启动在读取 Key/调用评测器之前被拒绝的本地检查 |

可重复的安全离线命令、未来 live 的精确命令与预算在 [GUARD_V2_NEXT_STEPS.md](GUARD_V2_NEXT_STEPS.md)。所有重跑的本地检查也使用新子目录，避免覆盖此次证据。

## 6. 验证结论的范围

本地已验证：输入动作约束与直接解释完成；安全且有上限的校验反馈；正式评分合同；拒绝准入的无 payload 诊断。真实 Writer 错误字段仍未知，修改后模型是否能修正、是否能走完 Safety 生命周期和是否能在 65,536 tokens 内完成报告均未进行 live 验证。当前可准备新的明确授权诊断运行；不应据此声称生产可用、任务成功率提升或成本优化。
