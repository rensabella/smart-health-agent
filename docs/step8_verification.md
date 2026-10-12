# Step 8 验证记录

本阶段完成本地语义检索、三个新增 Skill、五角色报告流程、逐条主张审核、有界修订，以及审核后的 SSE 正文。验证于 2026-09-08 至 2026-09-09 在 macOS arm64、Python 3.12.14 执行。

## 自动化检查

阶段收尾时 **468 passed，2 个上游弃用提示**，完整 pytest 运行约 22.17 秒。Ruff lint/格式检查通过，`pip check` 无依赖冲突；新增 ONNX Runtime 1.29.0、tokenizers 0.23.2 及其依赖写入版本快照。既有默认 LightGBM/RF 模型包没有重训或覆盖。

```bash
MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache" python -m pytest
ruff check app skills scripts tests examples train.py generate_demo_data.py
ruff format --check app skills scripts tests examples train.py generate_demo_data.py
python -m pip check
```

相比 Step 7 新增 46 项测试，包含：

- 来源/内容哈希、分块上限、实际 Qdrant 持久化与重开，拒绝损坏向量、载荷和模型身份不匹配。
- 空召回、查询上限、无效向量与缓存文件，不允许缺失检索时假成功。
- 草稿与分析/证据/策略指纹绑定，未知引用、不完整审核、跨 owner 访问和旧版本审核被拒绝。
- Writer 不能调用审核 Skill，Planner 不能提前发布；明显诊断、用药和因果断言即使传入正向语义判断也被拒绝。
- 审核失败后修订并重新审核；初稿加两次修订仍失败时阻止发布，长期 Memory 不写入成功记录。
- 审核期间取消和工具预算预留，验证只在 `publication.checked` 后出现正文分块，trace 不包含草稿或正文。

快速测试使用显式测试 embedding 和脚本决策；真实 E5 验证独立进行。现有上游 Starlette/httpx、AnyIO 弃用提示仍保留。测试总数会随后续阶段增加，本页保留本阶段记录。

## 真实检索与报告

官方模型下载地址在当前网络连接超时，本次通过镜像分段续传同一固定 revision 文件，合并后严格核对官方大小与 SHA-256；两份文件均一致。没有替换模型、降低完整性校验或使用测试向量器生成运行结果。固定文件信息见 [RAG 文档](rag_and_reports.md)。应用正常构建脚本仍使用官方固定地址。

```bash
python -m scripts.build_knowledge_index \
  --model-dir data/models/multilingual-e5-small --output data/knowledge/v1
MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache" \
  python -m examples.run_report --output docs/report_example
```

八份语料生成 **8 个 chunk、384 维 E5 向量**，知识版本 `1.0`。实际查询“EDA 皮电与活动背景”首位命中 `empatica_eda_context`，余弦分数约 **0.89654**；详见 [检索结果](report_example/retrieval.json)。其后也出现较宽泛的背景材料，不能据这个单次命中宣称检索质量达标，召回标注与阈值评估在 Step 9 进行。

五角色示例结果：completed，**18 次脚本决策、9 次真实 Skill、4 次委派、0 次外部 LLM 请求**。LightGBM 最高合成异常分数 `0.9983188764750849`；正文保留模型限制、真实来源与固定高分建议。Memory 仅写入分析摘要。查看 [完整结果](report_example/run.json)、[正文](report_example/report.txt) 和 [脱敏 trace](report_example/trace.json)。

`review_mode=scripted_test` 明确表示该次审核使用脚本 fixture；角色权限和发布门禁已实际执行，但不能据此证明 LLM 能准确判断证据支持关系。ONNX Runtime 遥测在加载时关闭；本机首次运行曾因默认用户缓存目录不可写产生缓存提示，未影响模型输出。

## 真实 HTTP/SSE

```bash
export HEALTH_RAG_MODE=local
export HEALTH_RAG_MODEL_DIR="$PWD/data/models/multilingual-e5-small"
export HEALTH_RAG_INDEX_DIR="$PWD/data/knowledge/v1"
export MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache"
python -m uvicorn examples.report_http_fixture:app \
  --host 127.0.0.1 --port 8765 --workers 1 --no-access-log \
  --log-config configs/logging.json
# 另一个终端：
python -m examples.check_report_http --output docs/report_example/http_smoke.json
```

该 fixture 固定使用本机测试 token、脚本角色、真实 E5/Qdrant/LightGBM；不是生产入口。实际 [HTTP 结果](report_example/http_smoke.json) 验证：

- 五个角色均有审计事件；事件 ID 1–77 连续，首次断开后任务仍为 running，随后 completed。
- 未授权订阅返回 401；按 `Last-Event-ID` 重连连续，报告分块再次回放内容相同。
- 报告正文在发布检查后出现，本样例正文少于 400 字符，因此为 1 个分块；哈希与最终正文一致。
- trace 没有正文/草稿文本，幂等提交复用同一任务，1 张请求的风险图下载为真实 PNG。
- 报告策略哈希、run 哈希与 publication 哈希一致，Memory 状态 stored。

验证服务已关闭，没有对外部署。这里测的是功能连通与约束行为，不是生产吞吐量、外部 LLM 质量或临床安全指标。
