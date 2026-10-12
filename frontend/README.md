# Streamlit 客户端

在仓库根目录安装 UI extra 并运行：

```bash
python -m pip install -e '.[ui]'
export HEALTH_API_BASE_URL=http://127.0.0.1:8000
python -m streamlit run frontend/streamlit_app.py
```

API 地址只从管理员环境变量读取，不接受正文提供的地址；Docker 内可设为 `http://api:8000`。远程访问请由部署者通过 HTTPS 提供前端和 API。API token 由每个浏览器会话在侧栏输入，仅驻留该 Streamlit 会话；页面不用共享缓存、磁盘或日志保存它。

- 数据工具调用 `/predict`、`/analyze`；图表请求由 artifact ID 和当前 session/input version 重新构造，不访问服务返回的任意 URL。
- Agent 支持不带上传的简单问题、CSV 分析和报告请求。决策后端、模型、RAG 与审核能力由 API 配置；界面不会隐式启用脚本决策或调用外部 LLM。
- SSE 订阅每次有短读取窗口，空闲时从 Last-Event-ID 续接，页面取消按钮保持可用。断线不取消后端任务；可通过 Run ID 与 Session ID 重新连接。
- 仅 `publication.checked` 之后的 `report.chunk` 可进入正文缓冲；显示前还要匹配已完成、已审核快照的报告 SHA-256 和分块内容。trace、思考事件、草稿和工具返回文本不能作为报告。
- 回放缺失时使用当前服务端快照。正文过期或重启后 `content_available=false` 时清除分块；结果不可用时不继续显示旧报告。已完成任务定期刷新以观察服务端保留期。
- 引用以普通文字和链接按钮展示，链接由来源 schema 验证；客户端不下载引用页面，不使用 `unsafe_allow_html`。

界面的 JSON/CSV 下载均需用户点击。清除页面会话会移除页面中的凭据（可选）、上传和结果；不等于删除后端 Memory，也不会隐式取消仍在执行的任务。

`frontend/client.py` 不依赖 Streamlit，其 HTTP、大小限制、SSE 重连与发布边界可通过模拟 transport 离线测试。界面测试使用 Streamlit AppTest，不启动外部 LLM。
