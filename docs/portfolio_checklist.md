# 求职展示与待补充清单

更新日期：2026-10-13。此页将已经具备的展示材料、发布状态和需要项目所有者补充的信息分开记录。它不是临床、合规或线上性能认证。

## 当前可展示

- `0.2.0` 工程重构：`src/smarthealth`、Settings、结构化执行追踪、Makefile 和独立 wheel 安装。当前本地 523 项测试通过，新增 24 项回归；脱离源码目录的真实训练/API/图表验证通过，见 [包重构验证](package_verification.md)。

- 九个类型化 Skills、五种 Agent 角色、ReAct Runtime、Router、Memory、Harness、本地 RAG 和报告发布检查。
- FastAPI Tool/Agent API、Streamlit、SSE 回放、来源与版本校验、脱敏审计。
- [完整项目报告](project_report.md)、[简历与面试材料](resume_project.md)、[真实合成报告示例](report_example/run.json)。
- 2026-10-12 重新运行：499 项 pytest 通过，7 条已知上游弃用提示；Ruff lint/126 个文件格式检查及 `pip check` 通过。本次未重新运行 9 月保存的训练和 E5 系统评测。
- 从 Git 暂存区导出完整源码（不含本机 `.venv`、模型和索引缓存），复用相同 Python 依赖进行第二次回归：499 项通过，26.93 秒。发布文件、合成测试资产与配置目录已一并核对，见 [发布验证记录](publication_verification.md)。
- Docker/Compose、GitHub Actions、贡献及安全说明。2026-10-12 首轮 [云端 CI](https://github.com/rensabella/smart-health-agent/actions/runs/38269692504) 已通过：499 项测试、Ruff、依赖检查及 Docker 镜像构建；容器运行仍待验证。

## 首次公开发布记录

下表保留 `0.1` 首次发布的证据；`0.2.0` 重构的发布状态见 [包重构验证](package_verification.md)。

| 项目 | 状态 |
|---|---|
| 仓库名 | `smart-health-agent` |
| 可见性 | 所有者已确认公开 |
| 本地版本控制 | `main` 已与公开仓库同步，使用 GitHub noreply 邮箱；原发布前历史保留在本地备份分支 |
| GitHub 账号授权 | 已通过官方 CLI 验证 `rensabella`，凭据由系统 keyring 保管；不需要安装插件 |
| 远端仓库 | 已公开发布 [rensabella/smart-health-agent](https://github.com/rensabella/smart-health-agent)，默认分支 `main`；206 个文件的路径、模式与 Git 内容哈希已逐项核对 |
| 已验证源码提交 | [`e97b5e0`](https://github.com/rensabella/smart-health-agent/commit/e97b5e0787cbb8b95258d64bce817146e68c8a44)；其后文档提交用于记录发布结果 |
| GitHub Actions | [首轮运行成功](https://github.com/rensabella/smart-health-agent/actions/runs/38269692504)：Python 499 passed、5 warnings、25.02 秒；Ruff/126 文件格式检查、`pip check`、Docker 镜像构建通过 |

发布已完成，没有待解决的登录或上传阻塞。仅 Markdown 的后续文档提交按工作流配置跳过 CI；上表明确列出实际测试的源码提交。云端镜像构建通过不代表容器已完成端到端运行验证。

## 需要你补充的信息

| 优先级 | 补充项 | 具体需要 | 对求职展示的作用 |
|---|---|---|---|
| 建议补充 | 真实项目时间和本人职责 | 起止时间、个人/团队项目、本人负责的模块；历史单位关系按证明材料填写 | 让简历经历和面试表述准确 |
| 建议补充 | 代码公开许可 | 确认对当前原创代码有发布/授权权利，并选择是否授予 MIT、Apache-2.0 等开源许可 | 当前公开供审阅，尚未添加开源许可证；第三方材料保留原归属 |
| 优先验证 | 一次真实 LLM 实验 | 选择服务和模型，在本机配置凭据，并明确可接受的调用预算 | 测量真实规划、工具调用和语义审核；当前外部 LLM 请求为 0 |
| 优先验证 | Docker 运行 | 云端镜像构建已通过；在目标环境运行资产准备、健康检查、CSV 分析、SSE、重启及停止流程 | 补齐容器端到端运行证据 |
| 展示增强 | 演示材料 | 使用合成 CSV 的页面截图或短视频，覆盖上传、trace、引用、结果与失败提示 | 招聘方可以快速理解项目操作流程 |
| 质量增强 | 独立评测复核 | 对检索来源和 claim-source 支持标签做独立复核，增加难负例和域外问题 | 当前 8 个检索查询与 8 个安全病例仍是小型基线 |
| 后续研究 | 授权真实数据 | 设备采样协议、数据使用权限、去标识与标注方案 | 只有完成这些工作后才有依据讨论真实数据泛化效果 |

## 当前简历可以强调

优先突出类型化能力层、角色权限、任务生命周期、审核与内容版本绑定、SSE 重连、摘要 Memory 和失败可见的 Evaluation。量化数字可以使用 9 Skills、5 角色、35 维特征、523 项工程测试，以及附带 `scripted` 限定的 14 个系统场景。

模型 F1/AUC 必须注明合成测试集；本地顺序计时不能写成生产 HTTP QPS；SSE 首报告块不能写成模型首 token。已知语义失败保留在 [评测报告](../evaluation/evaluation_report.md)，不通过删除负例营造安全保证。

完成某一补充项后，更新其对应结果、运行日期、模式和证据链接。凭据本身、真实健康数据和模型缓存继续留在仓库之外。
