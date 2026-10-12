# Step 5 验证记录

本阶段完成 Agent Router 及其与已有 ReAct、API、SSE 的集成。未实施 Memory、YAML Harness、RAG、Report Writer 或 Safety Verification Agent。

## 环境与检查

在 macOS arm64、Python 3.12.14 和现有项目虚拟环境运行，未新增 Python 依赖，未重新训练或更改默认模型包。

```bash
.venv/bin/ruff check app skills tests examples train.py generate_demo_data.py
.venv/bin/ruff format --check app skills tests examples train.py generate_demo_data.py
MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache" .venv/bin/python -m pytest -o addopts='' -q
.venv/bin/python -m pip check
```

结果：**197 passed，2 个上游弃用提示**；Ruff、格式检查与依赖一致性检查通过。弃用提示来自 Starlette TestClient 的 httpx 使用方式与 AnyIO BlockingPortal 别名，不影响本次用例通过，也未为消除提示升级依赖。

覆盖规则分类、有效入口角色、模型卡结果、真实 RF Skill 推理、图表产物、权限、数据版本、来源过期、取消、预算、升级、模拟 provider 协议以及 journal 恢复。路由测试包括中英文与 NFKC 归一化、完整 simple 语句识别、混合医疗请求、客户端 objective/entry 提示、含糊请求和缺少实际上传。

关键集成断言包括：

- Simple 无 Skill/委派可用；强行调用被拒绝并消耗决策额度。
- Complex 由 Planner 开始；一次升级沿用 run_id、已用预算和全局 deadline，第二次升级被拒绝。
- Health report、个人异常原因请求不会通过 score_definition 提示降级发布；当前能力缺失时零模型决策阻止执行。
- 既有结果只能从同 owner/session/input_version 的已完成任务复用，不能跨输入、过期或重新刷新 TTL。
- 明确要求已有图表时，原任务没有图表不能悄悄返回无图表结果。
- 等待决策期间来源过期，以及取消后后端仍返回动作，都不能发布结果。
- 路由与升级事件在持久化前通过严格 schema/状态检查；重启恢复最新有效路由，兼容旧 journal，不恢复私密输入或报告。

## 真实模型示例

```bash
MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache" \
  .venv/bin/python -m examples.run_routing --output-dir docs/router_example
```

使用现有 LightGBM v1.0、300 行合成 CSV、明确标记的 ScriptedDecisionBackend。所有场景的外部 LLM 请求为 0。

| 场景 | 路由 | 终态 | 决策次数 | Skill 次数 |
|---|---|---|---:|---:|
| 分数定义 | simple | completed | 1 | 0 |
| CSV 分析与图表 | complex | completed | 9 | 6 |
| 展示已有结果与图表 | simple | completed | 1 | 0 |
| 个人风险原因 | complex | blocked | 0 | 0 |
| 缺少上传 | complex | awaiting_input | 0 | 0 |
| 意图不明确 | complex | awaiting_input | 0 | 0 |

实际结构化输出见 [runs.json](router_example/runs.json)，执行事件见 [trace.json](router_example/trace.json)。当前模型分数描述合成异常，不是临床患病概率。脚本示例不证明真实 LLM 的路由理解、规划或健康解释质量。

## HTTP 与 SSE

使用 `examples.agent_http_fixture` 在 127.0.0.1:8765 启动单进程测试服务，运行 `python -m examples.check_agent_http`。复验结果见 [http_smoke.json](router_example/http_smoke.json)。检查真实 socket 断开不会取消后台任务、Last-Event-ID 连续回放、route.selected 进入事件流、幂等提交、完整 6 次 Skill 分析和带认证的 PNG 下载。测试服务随后关闭。

SSE 是执行进度事件，不是报告首 token 流。系统吞吐、真实 LLM 任务完成率、检索与引用质量、医学语义安全尚未评估，留在用户批准的后续阶段。
