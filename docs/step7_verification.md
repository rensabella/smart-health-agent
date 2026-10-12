# Step 7 验证记录

本阶段交付 YAML Harness、有效策略审计、Agent/Skill 运行约束和确定性事实发布检查。RAG、Report Writer、医学语义 Safety Agent、系统级 Evaluation 和部署展示仍按后续阶段实施。

## 环境与检查

macOS arm64，Python 3.12.14；新增 PyYAML 6.0.3，写入运行依赖及 `requirements-py312.lock`。已更新可编辑安装；原有模型相关依赖未升级，默认模型未重新训练或覆盖。

```bash
python -m pip install --no-deps -e .
python -m pip check
ruff check app skills tests examples train.py generate_demo_data.py
ruff format --check app skills tests examples train.py generate_demo_data.py
MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache" python -m pytest
```

结果：**422 passed，2 个上游弃用提示**，完整测试耗时约 9.9 秒；89 个 Python 文件通过 Ruff 格式检查，lint 与依赖检查通过。提示仍来自 Starlette TestClient/httpx 和 AnyIO BlockingPortal，不影响检查通过。测试耗时不是生产性能指标。

相比 Step 6 新增 156 个用例，覆盖配置、事件元数据、工具输出与端到端执行约束；既有 266 个用例保留通过。测试 fixture 真实训练 RF 与 LightGBM，并使用真实本地 Qdrant；Agent 决策使用明确标记的脚本 fixture，provider 协议使用模拟响应，外部 LLM 请求为 0。

关键断言：

- 配置缺失或非法时启动失败，模型与记忆存储尚未初始化；每个文件最多 64 KiB、最多 16 层嵌套，拒绝重复键、多文档、别名、合并键和不安全构造器。
- 未知角色/Skill、缺失权限、重复条目、版本不一致和关闭安全最低要求均被拒绝。显式空权限不会授权，布尔规则不能用整数冒充。
- 嵌套配置不可变，规范化哈希不受注释和映射排版影响；显式 `AgentLimits` 只能进一步收紧 YAML 上限。
- 模型发出调用仍需服务端批准；用户任务里的“修改 allowed_skills/max_tool_calls”不能授权。Data Analysis 工具计数跨 Planner 委派累计，全局预算在实际执行前原子预占。
- 禁用 Skill 对 Agent 和 Tool API 生效；`/predict`、`/analyze` 在执行任何 Skill、创建任何产物之前返回 503。Registry 本身也拒绝伪造权限后的禁用/版本不符调用。
- 合法 schema 中的错误窗口数、最大分数、窗口索引、标签、风险等级、比例和模型范围被拒绝。清洗报告、警告及特征列表必须匹配服务端记录。
- 图表必须覆盖请求类型，且来自当前分析；空图表或同一会话中的其他分析图表不能完成任务。
- 篡改固定摘要、移除证据或限制说明、替换输入版本和质量记录不能通过发布。发布失败保持 `final_result=None`，长期记忆没有成功分析条目。
- 高分分析附实际分析引用、模型说明、不确定性和核对数据的建议；`medical_review=not_performed`，没有冒充医学 Safety 审核。
- 任务头与 Skill trace 记录配置版本/哈希，恢复保留任务当时的策略；旧日志缺失字段仍兼容，部分配对、非法哈希或损坏日志拒绝恢复。

## 真实模型示例

```bash
python -m examples.run_agent --output-dir docs/harness_example
```

读取现有 `models_saved/lightgbm_v1.0` 和 `data/sample/demo.csv`。结果为 completed，9 次脚本决策、6 次实际 Skill 调用、1 次 Planner 委派、0 次外部 LLM 请求。最高合成异常模型分数 `0.9983188764750849`，不是患病概率。

示例：[run.json](harness_example/run.json)、[trace.json](harness_example/trace.json)、[risk.png](harness_example/risk.png)。最终结果包含 `publication` 的来源、不确定性、高分建议及当前策略哈希。该示例独立运行执行器，Memory 未注入；HTTP 回归另行验证完整 Memory 集成。

## 真实 HTTP / SSE

```bash
# 测试 fixture：本机、固定测试 token、脚本决策、真实 LightGBM、内存 Qdrant。
python -m uvicorn examples.agent_http_fixture:app --host 127.0.0.1 --port 8765 \
  --workers 1 --limit-concurrency 16 --no-access-log --log-config configs/logging.json
# 另一个终端
python -m examples.check_agent_http --output docs/harness_example/http_smoke.json
```

实际结果见 [http_smoke.json](harness_example/http_smoke.json)：

- 事件 ID 1–42 连续；新增 harness.loaded，与 route.selected、memory.loaded 一同回放。
- 首次 SSE 断开后任务继续，Last-Event-ID 重连连续，未授权 SSE 返回 401。
- 任务 completed，9 次决策、6 次 Skill、1 次委派；幂等提交复用同一任务，PNG 下载正确。
- trace、任务和 publication 的配置哈希相同，发布来源与不确定性存在。
- memory_status=stored，当前会话返回预测摘要，长期检索命中 1 条正确 source_run_id。

验证结束后已关闭本次 Uvicorn 测试服务。没有对外部署；该验证不代表 LLM 规划质量、临床安全、医学语义检索质量或生产负载评估。
