# Step 6 验证记录

交付范围为两层 Memory、Agent 上下文集成和 Memory API；未实施完整 YAML Harness、Redis 分布式会话、RAG 或医疗报告审核。

## 环境与完整检查

macOS arm64，Python 3.12.14，qdrant-client 1.19.0。新增 Qdrant 客户端及其传递依赖并记录到 requirements-py312.lock；已有模型相关依赖未升级，默认模型未重新训练或覆盖。

```bash
python -m pip install --no-deps -e .
python -m pip check
ruff check app skills tests examples train.py generate_demo_data.py
ruff format --check app skills tests examples train.py generate_demo_data.py
MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache" python -m pytest -o addopts='' -q
```

结果：**266 passed，2 个上游弃用提示**；安装、依赖检查、Ruff 和格式检查通过。提示仍来自 Starlette TestClient/httpx 和 AnyIO BlockingPortal，未为消除提示升级模型依赖。

新增的 69 个测试覆盖：本地短期会话生命周期；真实 Qdrant 持久化、过滤、过期、容量、幂等与 payload 校验；MemoryManager 和 Agent/API 集成。旧的 197 个数据、Registry、ReAct、Router 和 SSE 用例全部保留通过。

关键断言：

- 最近 5 个完整交互与任意数量工具事件分别管理，读取不延长空闲 TTL，新输入清除旧的“当前结果”。
- Qdrant 持久化 payload 中没有原始 query、CSV、逐窗数组、局部测量值或 artifact 引用；只接受摘要字段白名单。
- owner 过滤覆盖检索、列表、偏好与删除，读取再次校验 provenance 与 expiry；真实持久数据库重启后恢复摘要与偏好。
- 新分析才写长期风险记录，复用已有结果、失败、被阻止或取消不新增分析摘要；明确重新运行分析可以生成新记录。
- 历史上下文不能满足缺少上传的请求或伪造当前分析；LLM 只能通过当前 available_refs 和发布检查完成任务。
- 删除使旧 lease 失效；在途决策之后不会重建记忆，取消 native write 的等待者也不能造成删除后迟到写回。
- 记忆写入失败时，已验证分析保持 completed，memory_status 明确标为 unavailable；pending 收尾不能被任务容量淘汰。
- 趋势按时间读取并按模型指纹/评分协议分组，不能用语义 top-k 拼成时间序列。
- Memory API 校验认证、Pydantic schema、请求体上限、显式偏好、失效引用和删除行为。

## 真实模型与数据库示例

```bash
MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache" \
  python -m examples.run_memory --output-dir docs/memory_example
```

实际结果：2 条新分析摘要、3 个短期交互、2 条摘要进入后续分数解释的上下文；重启后恢复 2 条长期摘要和偏好，短期会话不恢复，跨 owner 查询为空，删除验证通过。运行使用现有 LightGBM、300 行合成 CSV、真实 Qdrant 本地持久化和明确标记的脚本决策，外部 LLM 请求为 0。

两个分析使用同一合成样例，目的是验证生命周期，不构成用户真实风险趋势。示例数据库置于独立临时目录并清理，输出仅为白名单摘要与验证信息。详见 [memory.json](memory_example/memory.json)。

## 真实 HTTP 与 SSE

```bash
# 本机测试 fixture，内存 Qdrant，固定测试 token；不要作为正式服务入口。
uvicorn examples.agent_http_fixture:app --host 127.0.0.1 --port 8765 \
  --workers 1 --limit-concurrency 16 --no-access-log --log-config configs/logging.json
# 另一个终端
python -m examples.check_agent_http --output docs/memory_example/http_smoke.json
```

结果为 41 条连续事件，包含 route.selected 和 memory.loaded；断线后任务继续，Last-Event-ID 回放连续，幂等提交与 PNG 下载通过。任务执行 9 次脚本决策、6 次 Skill、1 次委派；memory_status=stored，Memory API 返回当前预测摘要，检索命中 1 条正确 source_run_id。详见 [http_smoke.json](memory_example/http_smoke.json)。测试服务已关闭。

该验证不测量真实 LLM 规划质量、医学语义检索准确率、临床安全、分布式一致性或生产负载。上述评估与部署工作保留到后续已约定阶段。
