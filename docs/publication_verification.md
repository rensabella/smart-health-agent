# GitHub 发布准备验证

日期：2026-10-12（Asia/Shanghai）。目标是将本目录作为公开 `smart-health-agent` 仓库根目录，供招聘方查看代码与可复现证据。

## 已执行

| 检查 | 实际结果 |
|---|---|
| 原工作区完整 pytest | 499 passed，7 warnings，21.80 秒 |
| Ruff lint | 通过 |
| Ruff format | 126 个 Python 文件已符合格式 |
| Python 依赖 | `pip check` 无冲突 |
| 发布源文件导出 | 从 Git 暂存区导出，保留 configs、合成 demo、报告快照及前端测试图片 |
| 导出源码完整 pytest | 499 passed，7 warnings，26.93 秒；复用本机已验证依赖，不声称全新依赖安装 |
| Markdown 链接 | 新发布入口和展示文档中的本地目标均存在 |
| Git whitespace 检查 | 通过；文档换行保留语义 |
| 发布内容审阅 | 未包含本机虚拟环境、模型二进制、E5 缓存、Qdrant 数据、原始报告附件或其他参考工程 |
| 凭据模式扫描 | 扫描拟发布文件，未发现常见 GitHub/provider/AWS 密钥模式或私钥头；扫描不代替后续人工审阅 |

首次导出核对时发现暂存清单缺少 `configs/`，已在发布前补齐七份配置并重新导出，完整回归通过。这样核验的是准备发布的源码，而不只是本机工作区。

原训练和真实 E5/领域模型评测报告仍是 2026-09-09 的记录，本次未重训、未下载新的模型，也没有调用外部 LLM。现有语义审核失败继续保留在评测报告中。

## 当前边界

已通过官方 CLI 验证 GitHub 账号 `rensabella` 并创建公开仓库 [smart-health-agent](https://github.com/rensabella/smart-health-agent)。本地首次提交为 `7f59172`，提交使用该账号的 GitHub noreply 邮箱，不公开本机配置的个人邮箱。

执行 `git push --set-upstream origin main` 时，自动审批系统因额度耗尽无法完成审核，命令未执行。这是审批失败，不是对操作不安全的判定，也不是 GitHub 认证或代码错误。没有改用其他权限、工具或传输方式重试；源码推送和云端 CI 仍待审批问题解决。本地核对未发现 `origin/main` 跟踪引用，此核对不包含新的远端网络请求。

本机没有 Docker；现有 CI 配置提供镜像构建任务，但即使后续构建通过，仍需单独验证容器资产准备、启动、HTTP/SSE 和停止行为。源代码目前支持源码 checkout/editable 安装；独立 wheel 的配置资源打包属于后续增强。

登录凭据不进入仓库。已在工作区外层的本地工具目录准备官方 GitHub CLI，并核对下载资产 SHA-256；它不属于发布内容，也未改动系统安装。

云端工作流包含 Python 测试/lint 和 Docker 镜像构建两个任务。仅 Markdown 文档变更跳过自动运行，源码、配置和测试资产变更仍会触发；也可手动运行。发布完成后更新 [求职补充清单](portfolio_checklist.md) 的提交与 CI 实际状态，本页保留已执行检查的范围和日期。
