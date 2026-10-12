# GitHub 发布验证

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
| 首次源码发布文件核验 | 206 个文件的路径、模式与 blob SHA 全部一致；该次 tree SHA 为 `9b353769136ff0a40ed21f12b342c7a80b2aa5fd` |
| 云端 Python CI | Ubuntu、Python 3.12，全新安装项目依赖；499 passed，5 warnings，25.02 秒；Ruff、126 个文件格式检查和 `pip check` 通过 |
| 云端 Docker CI | `docker build --tag smart-health-agent:ci .` 成功；未执行容器运行检查 |

首次导出核对时发现暂存清单缺少 `configs/`，已在发布前补齐七份配置并重新导出，完整回归通过。这样核验的是准备发布的源码，而不只是本机工作区。

原训练和真实 E5/领域模型评测报告仍是 2026-09-09 的记录，本次未重训、未下载新的模型，也没有调用外部 LLM。现有语义审核失败继续保留在评测报告中。

## 发布结果

已通过官方 CLI 验证 GitHub 账号 `rensabella`，并将完整项目发布到公开仓库 [smart-health-agent](https://github.com/rensabella/smart-health-agent)，默认分支 `main`。提交使用该账号的 GitHub noreply 邮箱。

首次源码发布因本机 Git HTTPS 传输错误，使用 GitHub 官方 API 完成。远端采用「初始化 README → 完整源码」两次初始提交，后续文档提交记录发布结果。更新 `main` 时使用 `force: false`，并核对完整远端 tree 与每个文件的内容哈希。

实际通过 CI 的源码提交是 [`e97b5e0787cbb8b95258d64bce817146e68c8a44`](https://github.com/rensabella/smart-health-agent/commit/e97b5e0787cbb8b95258d64bce817146e68c8a44)。[GitHub Actions 运行 38269692504](https://github.com/rensabella/smart-health-agent/actions/runs/38269692504) 的 `python` 与 `container-build` 两项任务均成功，完成时间为 2026-10-12 23:23（Asia/Shanghai）。测试数量、耗时和质量检查结果取自该次任务日志。

本地已导入并验证远端提交对象，`main` 与 `origin/main` 同步。发布前的 `7f59172` / `c9bd0d4` 历史保留在本地 `backup/pre-publication-20261012` 分支；工作文件没有因同步丢失或替换。

## 当前边界

本机没有 Docker；云端已验证镜像构建，仍需单独验证容器资产准备、启动、HTTP/SSE、重启和停止行为。源代码目前支持源码 checkout/editable 安装；独立 wheel 的配置资源打包属于后续增强。

登录凭据不进入仓库。已在工作区外层的本地工具目录准备官方 GitHub CLI，并核对下载资产 SHA-256；它不属于发布内容，也未改动系统安装。

云端工作流包含 Python 测试/lint 和 Docker 镜像构建两个任务。仅 Markdown 文档变更跳过自动运行，源码、配置和测试资产变更仍会触发；也可手动运行。本次 CI 后的文档更新仅记录发布结果，不声称这些文档提交重新运行过 CI。[求职补充清单](portfolio_checklist.md) 持续记录个人履历、许可选择、真实 LLM 实验与展示材料等后续补充项。
