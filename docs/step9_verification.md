# Step 9 验证记录

2026-09-09，macOS arm64、Python 3.12.14。本阶段交付 Agent/RAG/Safety/System Evaluation、Streamlit HTTP/SSE 客户端、Docker Compose、CI 配置及运行文档。底层 Tool API 和训练模型包继续复用，默认后端仍不会自动调用外部 LLM。

## 自动化回归

本机最终完整运行：**499 passed，7 warnings，14.93 秒**。相比 Step 8 增加 31 项测试。Ruff lint 通过，格式检查确认 126 个 Python 文件已格式化，`pip check` 无依赖冲突。

```bash
MPLCONFIGDIR="$PWD/data/generated/matplotlib-cache" python -m pytest
ruff check app skills scripts evaluation frontend tests examples train.py generate_demo_data.py
ruff format --check app skills scripts evaluation frontend tests examples train.py generate_demo_data.py
python -m pip check
```

新增测试覆盖评测分母与 N/A、来源/标签完整性、标签不进入决策后端、live 请求共享预算、操作失败不冒充正确安全拒绝；前端固定 API 地址、响应大小限制、SSE 顺序与缺口、发布门禁、正文哈希、结果过期、凭据清除和页面报告展示；另有 Docker 配置静态约束和 provider 委派工具协议回归。快速测试使用明确的脚本决策及测试向量器，仍实际训练 baseline fixture、运行 Qdrant 和领域服务；真实 E5 的评测独立执行。

保留 7 条上游弃用提示：Starlette/TestClient 与 AnyIO 各一条，Streamlit 使用 NumPy generic timedelta 产生五条。没有隐藏提示或为此升级已训练模型依赖。UI extra 安装 Streamlit 1.63.0，`requirements-py312.lock` 同步为本机 78 个包的版本快照；它不是跨平台、带 wheel 哈希的锁文件。

## 真实模型与系统评测

运行入口为 `python -m evaluation.run_eval`。实际使用已训练 LightGBM、固定 revision 的 multilingual-e5-small ONNX、8 份受控语料的真实 Qdrant 索引，以及 Runtime/Registry/Memory/报告服务。角色决策明确为 `scripted`，外部 provider 请求数为 **0**。

[完整 JSON](../evaluation/evaluation_report.json) 记录逐场景结果、配置/模型/语料/索引身份和计时；[Markdown 报告](../evaluation/evaluation_report.md) 由同一结果生成。[用例说明](evaluation_cases.md) 记录标签依据，当前独立人工复核状态为 pending。

| 指标 | 结果 |
|---|---:|
| 系统场景满足契约 | 14/14 |
| 预期完成任务满足契约 | 8/8 |
| 路由类别和目标匹配 | 14/14 |
| Skill 成功执行 | 44/45 |
| 平均 Skill 调用次数 | 3.214/场景 |
| 检索命中 | 8/8，top-k=3，阈值 0.78 |
| 已发布 claim-source 对得到支持 | 3/4 |
| 安全病例判断符合标签 | 7/8 |
| 不应发布却发布 / 全部发布 | 1/4 |
| 不应发布却发布 / 全部负例 | 1/5 |
| 场景延迟 mean / p95 | 75.36 / 465.50 ms |
| 首个审核后报告块平均延迟 | 102.44 ms，4 个报告 |
| 顺序本地吞吐量 | 10.624 场景/秒，并发 1 |
| 模型首 token 延迟 | not_measured |

一次 Skill 失败来自故意无效的 CSV；任务随后正确要求补充输入，但该调用仍计为失败。时间指标在模型已经加载后测量，不包含启动或上传准备；吞吐量包含场景初始化和清理，不代表 HTTP 服务生产容量。首进度事件、首审核后报告块没有替代模型首 token 指标。

保留失败病例 `unsupported_sleep_equivalence`：主张“睡眠质量等同于睡眠时长”不受其引用支持，乐观的脚本审核器仍提交正向语义判断，确定性文本守卫也没有阻止它。没有为这个句子添加专门规则，也没有删改负例来使指标通过。它证明当前工程约束不能替代语义判断；本次没有证明外部 LLM 能纠正这个失败。

`--fail-on-regression` 会将未达预期的病例转成退出状态 1，当前基线包含上述失败。完整报告中的动态主张尚无独立语义标签，相关质量字段为 null；3/4 的引用精度来自独立单主张病例，不被推广成完整报告的语义准确率。详见 [指标定义](evaluation.md)。

## 前端与 HTTP/SSE

真实本机 Streamlit 浏览器检查已验证首页、API 连接、CSV 文件上传，以及“解释 risk score”简单任务的完成与结果展示；上传限制显示为 5 MB。该服务连接显式脚本决策的报告 fixture 和真实本地领域模型。

完整报告正文的页面展示由 Streamlit AppTest 验证：使用实际合成报告样例快照，检查正文以普通文本显示，并显示脚本审核标记。另一项回归验证运行中的任务不会因为暂时没有最终结果而清空未来报告分块。纯 HTTP/SSE 客户端测试另外覆盖正文哈希、发布次序、重连缺口和过期结果。这些检查没有调用外部 LLM。

真实 Uvicorn 的完整报告 HTTP/SSE 冒烟记录见 [Step 8 结果](report_example/http_smoke.json)：77 条连续事件、断开后后台继续、按事件 ID 重放、幂等提交、未授权订阅 401、1 张真实 PNG、审核后正文与快照哈希一致。Step 9 使用实际前端 HTTP 客户端另创建了报告任务，但**浏览器最后一次“连接任务”点击被自动审批系统因额度限制拒绝，未执行**。因此没有把这次浏览器操作写成完整报告端到端验收；没有绕过该拒绝。

收尾时已正常停止本次启动的 Uvicorn 与 Streamlit 验证服务，两者均返回退出状态 0；没有对外部署或保留后台测试服务。

## 集成修复

- provider 委派工具原先只列出 Data Analysis，现从 Harness 权限向决策请求传递可委派角色，工具 schema 只公开当前允许的角色；空列表不公开 delegate 工具。RAG、Writer、Safety 可以通过同一适配协议被选择，仍受服务端权限约束。
- 前端先区分正在运行与已完成结果，再处理正文不可用；不再把执行中的任务误当成报告已过期。
- Docker 中模型、Memory、trace 路径显式位于受控卷，避免非 root、只读根文件系统下写入源码目录；Streamlit 配置及 5 MiB 上传限制纳入镜像和 Compose。

## 部署与剩余验证边界

已提供 [Dockerfile](../Dockerfile)、[Compose](../docker-compose.yml)、[GitHub Actions](../.github/workflows/checks.yml) 和 [部署文档](deployment.md)。静态测试检查单 worker、数据卷、秘密隔离、资源约束等配置关系。当前环境没有 Docker 可执行程序，所以没有实际构建镜像或运行容器；远程 CI 也未执行，没有初始化或推送 Git 仓库。

真实 LLM 的规划/审核效果、独立标签复核、更大的检索难负例集、多用户身份、HTTP 并发压测和分布式任务恢复仍待后续工作。本阶段完成的是可运行的本地工程实现与可复现基线，没有把脚本质量、静态容器检查或单进程计时当成生产验证。
