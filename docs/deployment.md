**Docker 部署与资产准备**

[Dockerfile](../Dockerfile) 使用 Python 3.12 slim、Linux `libgomp1` 和 UID/GID `10001`。同一镜像运行 FastAPI、Streamlit 及一次性准备命令，运行服务均使用非 root 用户。[Compose 配置](../docker-compose.yml) 将 API 固定为一个 Uvicorn worker，默认使用报告 Harness、真实 LightGBM、E5 ONNX 和本地 Qdrant。

2026-10-12，[GitHub Actions](https://github.com/rensabella/smart-health-agent/actions/runs/38269692504) 已在 `ubuntu-latest` 对源码提交 `e97b5e0` 成功构建 Docker 镜像。本地开发环境没有 Docker 可执行程序；**容器内资产准备、服务启动及运行行为尚未验证**。该 CI 的 `container-build` 任务只构建镜像，Python 测试与 Compose 静态检查也不能作为容器运行验证。

从仓库根目录执行以下命令，需要 Docker Engine/Desktop 和 Docker Compose v2。先在当前 shell 注入自己的 API bearer token；例如生成新的随机值：

```bash
export HEALTH_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose config --quiet
docker compose build api
```

也可以从密码管理器或部署秘密存储注入已有 token。Compose 中没有默认 token，缺少变量时配置检查就会失败。完整的 `docker compose config` 会展开环境变量；使用 `--quiet` 做校验，避免把秘密写入输出。Streamlit 侧栏需要输入同一个 token；前端不从共享环境读取 API token，也不挂载后端的数据卷。

构建只安装 Python 依赖和系统 OpenMP 库，不训练模型、不下载 E5、不调用外部 LLM。[.dockerignore](../.dockerignore) 使用源码白名单，排除本机虚拟环境、模型二进制、下载缓存、训练数据、记忆、日志及秘密文件。镜像仅包含已审核的合成 `data/sample/demo.csv`，供页面下载示例。macOS 的 `requirements-py312.lock` 不作为 Linux 锁文件使用；当前镜像从 `pyproject.toml` 解析 Linux 依赖。

第一次运行时，按顺序准备资产：

```bash
docker compose --profile prepare run --rm prepare-data
docker compose --profile prepare run --rm train-models
docker compose --profile prepare run --rm prepare-rag
docker compose up -d api frontend
docker compose ps
```

`prepare-data` 在专用卷中生成合成训练集和示例数据；`train-models` 用当前镜像的依赖训练 RandomForest、LightGBM，并写出校验清单。API 默认加载 `lightgbm_v1.0`。在镜像内训练能够避免宿主机模型与 Linux 运行环境的库版本不一致；模型加载器会严格检查依赖版本，不应直接把 macOS 模型目录当成已经兼容的容器资产。

`prepare-rag` 显式下载固定 revision 的 `intfloat/multilingual-e5-small` ONNX 和 tokenizer，校验预置长度与 SHA-256，再为仓库内的受控语料建立本地 Qdrant 索引。模型资产约 487 MB，需预留下载、索引和运行内存空间。下载必须能访问 Hugging Face 及其签名 CDN；失败时命令返回错误，不会用其他模型或随机向量替代。已有模型文件只在哈希完全匹配时复用。应用启动本身不会下载模型、网页或索引。

准备服务属于 `prepare` profile，不会随普通 `docker compose up` 自动执行；显式运行目标服务即可触发它们。不要使用 `docker compose --profile prepare up` 将生成、训练与索引并行启动，这些命令需要上述顺序。[Docker Compose profiles 说明](https://docs.docker.com/compose/how-tos/profiles/)

训练模型包目录和知识索引目录已经存在时，脚本会拒绝覆盖。正常重启只需 `docker compose up -d`。更新模型、依赖、语料或索引时，应停止 API，在新目录或新卷准备新的完整版本，再修改 Compose 的资产路径并重建或重启；不能同时让索引构建器和 API 打开同一 Qdrant 目录。重复执行准备命令不是一个原地升级流程。

准备完成后，可以访问：

| 地址 | 用途 |
|---|---|
| `http://127.0.0.1:8501` | Streamlit，侧栏输入 bearer token 后上传合成 CSV |
| `http://127.0.0.1:8000/health` | 进程存活检查 |
| `http://127.0.0.1:8000/ready` | 当前模型加载状态 |
| `http://127.0.0.1:8000/docs` | FastAPI 接口说明 |

默认 `HEALTH_AGENT_BACKEND=disabled`，因此可以使用底层数据分析 API，但创建 Agent 任务会明确返回后端未配置。需要真实 Agent 决策时，配置兼容的 tool-calling 服务，再重建 API 容器：

```bash
export HEALTH_AGENT_BACKEND=openai_compatible
export HEALTH_LLM_BASE_URL=https://your-provider.example/v1
export HEALTH_LLM_MODEL=your-tool-calling-model
export HEALTH_LLM_API_KEY=your-provider-secret
docker compose up -d --force-recreate api frontend
```

这些地址与秘密是部署者提供的配置，示例值不可直接使用。启动不会主动发送 LLM 请求；用户提交 Agent 任务后才会调用已配置的后端。本部署步骤没有调用任何外部 LLM。

持久化路径相互隔离：

| 命名卷 | 容器路径 | API 挂载方式与用途 |
|---|---|---|
| `demo_data` | `/var/lib/smarthealth/datasets` | 只供准备服务使用；包含合成训练数据 |
| `model_bundles` | `/var/lib/smarthealth/models` | 只读；已训练模型与元数据 |
| `e5_cache` | `/var/lib/smarthealth/embeddings` | 只读；固定 E5 模型资产 |
| `knowledge_index` | `/var/lib/smarthealth/knowledge` | 可写；本地 Qdrant 锁文件与索引存储 |
| `long_term_memory` | `/var/lib/smarthealth/memory` | 可写；受控历史摘要与偏好 |
| `agent_traces` | `/var/lib/smarthealth/traces` | 可写；脱敏 Agent 事件 |

镜像预先创建归 UID/GID `10001` 所有的卷目录，并放置初始化标记。新命名卷通过 Docker 默认的内容复制机制获得这些目录内容。不要为这些挂载启用 `nocopy`，也不要直接换成归 root 所有的空宿主机目录。[Docker 卷初始化说明](https://docs.docker.com/engine/storage/volumes/)

已有卷或手动导入的文件不会自动重新分配所有权；它们需要归同一个 UID/GID 所有。应用还会收紧知识索引、记忆与日志目录的文件权限，并拒绝不支持的符号链接。模型缓存可只读挂载，但知识索引即使只用于检索也必须可写：当前实现会调整权限，本地 Qdrant 还需要打开锁文件。这里的“知识内容不可变”不等同于可以使用只读文件系统挂载。

不要增加 `--workers`，也不要执行 `--scale api=2` 来共享这些卷。当前 Agent 任务、短期会话、工具额度、暂存上传及 SSE 状态由单个 API 进程持有；本地 Qdrant 文件路径也不能由多个进程同时打开。未来扩容需要独立 Qdrant 服务、共享任务/会话状态和分布式协调，当前 Compose 没有实现这些能力。

运行服务使用只读根文件系统；临时缓存位于独立的 128 MiB `/tmp` tmpfs，能力集清空，禁止获取新权限。默认 API 限制为 2 CPU / 4 GiB，前端为 1 CPU / 1 GiB；训练和建索引各自限制为 2 CPU / 4 GiB，数据生成限制为 2 CPU / 2 GiB。默认最大进程数为 256，JSON 容器日志轮转为每文件 10 MiB、最多 3 个文件。这些是初始资源边界，不是经过容器压测得出的容量承诺。[Compose 服务配置参考](https://docs.docker.com/reference/compose-file/services/)

API 容器使用 `/ready` 健康检查，并为模型初始化预留 120 秒启动窗口；前端等待 API healthy 后启动，并检查自己的 `/_stcore/health`。`/ready` 只证明 ML 模型已经加载，不证明外部 LLM 可用、索引召回充分或端到端报告任务成功。API 启动时若本地 RAG 资产无效，会直接启动失败；容器应结合日志排查。Docker 标记 `unhealthy` 本身不会重启仍存活的进程，`unless-stopped` 处理的是进程退出后的重启。

API 的停止宽限期为 210 秒，以容纳有界任务取消和底层工作线程退出。进程重启保留持久化摘要与脱敏 trace，但不会恢复正在执行的任务、原始上传或临时结果；查询旧 trace 不能恢复这些数据。

默认端口只绑定宿主机 `127.0.0.1`。如需跨机器提供服务，应另行配置 TLS、入口身份认证和访问范围；当前 bearer token 对应已有的单一开发身份，不能当作多租户认证。不要把本配置中的共享 token 部署成互不信任用户之间的权限隔离。

停止服务并保留资产使用 `docker compose down`。`docker compose down -v` 会删除模型、知识索引、历史摘要和 trace 卷，只适合明确要清空该环境时使用。备份或迁移 Qdrant 文件前先停止 API，避免复制正在写入的本地数据库；同时保留模型、配置和索引清单以便核对版本。

交付验收仍需在有 Docker 的目标环境按上述命令准备镜像与资产，完成健康检查、CSV 分析、SSE 断线重放、重启后的 trace/记忆行为以及停止流程。已通过的云端镜像构建与配置静态检查不能替代这些运行验证；仅修改 Markdown 的提交不会再次触发 CI。
