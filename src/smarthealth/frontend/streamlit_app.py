"""Streamlit presentation; every computation and authorization stays behind HTTP."""

from __future__ import annotations

import json
import time
from uuid import uuid4

import streamlit as st

from smarthealth.agents.contracts import TERMINAL_STATUSES, CreateRunRequest
from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings
from smarthealth.frontend.client import (
    MAX_UPLOAD_BYTES,
    ClientError,
    HealthApiClient,
    ProgressTracker,
    approved_report,
)
from smarthealth.schemas.response import AnalyzeResponse, PredictResponse

API_BASE_URL = Settings.from_env().api_base_url
_ERRORS = {
    "unauthorized": "认证失败，请检查侧栏中的 API token。",
    "connection_failed": "暂时无法连接 API；请检查服务状态后重试。",
    "backend_unavailable": "Agent 决策后端尚未配置。数据工具仍可独立使用。",
    "model_unavailable": "后端尚未加载模型，请先完成训练并启动服务。",
    "safety_unavailable": "后端尚未开启报告检索和审核能力。",
    "session_busy": "当前会话已有执行中的任务，请等待完成或先取消。",
    "file_too_large": "CSV 超过 5 MiB 上传限制。",
    "artifact_expired": "临时数据已过期，请重新上传。",
    "artifact_not_found": "临时数据已不可用，请重新上传。",
    "run_not_found": "任务不存在或已超过保留期。",
    "publication_not_verified": "报告未通过客户端发布契约检查，正文未显示。",
    "report_chunk_mismatch": "报告正文与审核后快照不一致，正文未显示。",
}


def _error(error: ClientError):
    st.error(_ERRORS.get(error.code, "请求未完成，请查看错误代码并重试。") + f" ({error.code})")


def _reset_workspace(clear_token=False):
    token = "" if clear_token else st.session_state.get("health_api_token", "")
    version = st.session_state.get("health_upload_widget_version", 0) + 1
    for key in list(st.session_state):
        if key.startswith("health_"):
            del st.session_state[key]
    st.session_state["health_api_token"] = token
    st.session_state["health_upload_widget_version"] = version


def _initialize():
    defaults = {
        "health_session_id": uuid4().hex,
        "health_input_version": uuid4().hex,
        "health_upload_widget_version": 0,
        "health_api_token": "",
        "health_tool_result": None,
        "health_run": None,
        "health_tracker": None,
        "health_pending": None,
        "health_last_poll": 0.0,
        "health_connection_warning": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _client():
    return HealthApiClient(API_BASE_URL, st.session_state["health_api_token"])


def _prediction(predicted):
    columns = st.columns(4)
    columns[0].metric("合成异常分数", f"{predicted.risk_score:.3f}")
    columns[1].metric("模型输出", predicted.prediction)
    columns[2].metric("分析窗口", str(predicted.window_count))
    columns[3].metric("分数等级", predicted.risk_level)
    st.caption(f"{predicted.model.name} · v{predicted.model.version} · 会话取最高窗口分数")
    st.dataframe(
        [feature.model_dump() for feature in predicted.important_features],
        hide_index=True,
        width="stretch",
    )


def _analysis(result: AnalyzeResponse):
    st.text(result.summary)
    _prediction(result.prediction)
    for warning in result.warnings:
        st.warning(warning)
    with st.expander("特征解释与数据质量"):
        st.caption("特征贡献说明模型行为；不能据此推断个体病因。")
        st.dataframe(
            [feature.model_dump() for feature in result.explanation.local_evidence],
            hide_index=True,
            width="stretch",
        )
        st.json(result.quality_report)
    if result.visualization:
        with st.expander("查看图表", expanded=True):
            try:
                with _client() as client:
                    for plot in result.visualization:
                        content = client.image(
                            plot.artifact_ref, result.session_id, result.input_version
                        )
                        st.image(content, caption=plot.kind, width="stretch")
            except ClientError as exc:
                _error(exc)


def _accept_run(snapshot):
    st.session_state["health_run"] = snapshot
    st.session_state["health_session_id"] = snapshot.session_id
    st.session_state["health_input_version"] = snapshot.input_version
    st.session_state["health_tracker"] = ProgressTracker(snapshot.run_id)
    st.session_state["health_last_poll"] = 0.0
    st.session_state["health_pending"] = None
    st.session_state["health_connection_warning"] = False


def _submit_pending():
    pending = st.session_state["health_pending"]
    if pending:
        with _client() as client:
            _accept_run(client.create_run(pending["request"], pending["key"]))


def _render_agent_result(snapshot, tracker):
    if snapshot.status != "completed":
        return
    if not snapshot.result_available or not snapshot.final_result:
        tracker.unavailable()
        st.info("任务已完成，但正文已过期或在服务重启后不可用。执行记录仍保留。")
        return
    result = snapshot.final_result
    kind = result.get("kind")
    try:
        if kind == "health_report":
            report = approved_report(snapshot)
            if report is None:
                return
            prefix = tracker.checked_prefix(report)
            st.subheader(report.title)
            if result.get("review_mode") == "scripted_test":
                st.warning("这是脚本决策与脚本审核的演示结果，不能用于判断真实 LLM 的审核质量。")
            else:
                st.caption("独立 Safety 角色审核与发布检查已完成。该过程不等于临床验证。")
            if prefix:
                st.caption("通过 SSE 接收的审核后正文")
                st.text(prefix)
                if len(prefix) < len(report.body):
                    st.caption("正在续传剩余已审核正文。")
            else:
                st.caption("已同步服务端审核后报告快照。")
                st.text(report.body)
            st.warning(report.uncertainty)
            for recommendation in report.recommendations:
                st.info(recommendation)
            with st.expander("逐条主张与引用来源"):
                for claim in report.claims:
                    st.text(claim.text)
                    for source in claim.sources:
                        # Links open in the user's browser; this app never downloads source URLs.
                        st.link_button(source.source.title, source.source.url)
                        st.caption(source.locator)
            with st.expander("审核状态"):
                st.json(report.safety.model_dump(mode="json"))
            with st.expander("查看当前数值分析与图表"):
                _analysis(AnalyzeResponse.model_validate(result["analysis"], strict=True))
            st.download_button(
                "下载已审核报告 JSON",
                json.dumps(result, ensure_ascii=False, indent=2),
                file_name="reviewed_report.json",
                mime="application/json",
                key="health_report_download",
            )
        elif kind == "data_analysis":
            _analysis(AnalyzeResponse.model_validate(result["analysis"], strict=True))
            publication = result.get("publication", {})
            if publication.get("uncertainty"):
                st.warning(publication["uncertainty"])
        elif kind == "score_definition":
            st.text(str(result.get("summary", "")))
            source = result.get("source", {})
            st.caption(f"依据：{source.get('name', '')} · v{source.get('version', '')} model card")
    except ClientError as exc:
        _error(exc)
    except (ValueError, KeyError, TypeError):
        _error(ClientError("invalid_response"))


@st.fragment(run_every="1s")
def _run_panel():
    snapshot = st.session_state["health_run"]
    if snapshot is None:
        st.info("提交任务后，这里会显示角色与工具进度。只有审核完成的正文可以进入报告区域。")
        return
    tracker = st.session_state["health_tracker"]
    st.caption(f"Run ID: {snapshot.run_id} · Session: {snapshot.session_id}")
    active = snapshot.status not in TERMINAL_STATUSES
    cancel = st.button("取消当前任务", disabled=not active, key="health_cancel_run")
    now = time.monotonic()
    should_poll = cancel or active or now - st.session_state["health_last_poll"] >= 10
    # Even completed snapshots are refreshed so client-side display does not outlive server retention.
    if should_poll:
        try:
            with _client() as client:
                snapshot = (
                    client.cancel(snapshot.run_id, snapshot.session_id)
                    if cancel
                    else client.snapshot(snapshot.run_id, snapshot.session_id)
                )
                st.session_state["health_run"] = snapshot
                if (
                    tracker.cursor < snapshot.last_event_id
                    or snapshot.status not in TERMINAL_STATUSES
                ):
                    try:
                        batch = client.events(
                            snapshot.run_id, snapshot.session_id, after_id=tracker.cursor
                        )
                        st.session_state["health_connection_warning"] = batch.disconnected
                        for message in batch.messages:
                            tracker.apply(message)
                    except ClientError as exc:
                        if exc.code == "replay_unavailable":
                            tracker.replay_gap = True
                            tracker.chunks.clear()
                            tracker.cursor = snapshot.last_event_id
                        else:
                            raise
                st.session_state["health_last_poll"] = now
        except ClientError as exc:
            st.session_state["health_connection_warning"] = True
            # An unverified cached snapshot must not remain visible after a failed refresh.
            _error(exc)
            st.info("连接中断不会取消后端任务；页面将使用最后事件 ID 重新连接。")
            return
    columns = st.columns(4)
    columns[0].metric("任务状态", snapshot.status)
    columns[1].metric("决策次数", snapshot.decision_count)
    columns[2].metric("工具调用", snapshot.tool_calls)
    columns[3].metric("角色委派", snapshot.delegation_count)
    if snapshot.route:
        st.caption(f"路径：{snapshot.route.workflow_type} · 路由依据：{snapshot.route.reason_code}")
    if snapshot.message:
        st.text(snapshot.message)
    if snapshot.error_code:
        _error(ClientError(snapshot.error_code))
    if st.session_state["health_connection_warning"]:
        st.info("事件连接暂时中断，正在续接。任务仍由后端执行。")
    if tracker.replay_gap:
        st.info("部分进度事件已不在回放窗口内；当前状态和结果已从服务端快照同步。")
    if (
        not tracker.content_available
        and snapshot.status == "completed"
        and snapshot.final_result
        and snapshot.final_result.get("kind") == "health_report"
    ):
        st.caption("部分报告内容无法从事件流恢复，已同步服务端审核后结果。")
    with st.expander("执行进度（受限元数据）", expanded=active):
        if tracker.rows:
            st.dataframe(list(tracker.rows), hide_index=True, width="stretch")
    _render_agent_result(snapshot, tracker)


def main():
    st.set_page_config(page_title="SmartHealth Agent", page_icon="🩺", layout="wide")
    _initialize()
    with st.sidebar:
        st.header("连接设置")
        st.caption("API 地址由部署者配置。")
        st.text(API_BASE_URL)
        st.text_input(
            "API token", type="password", key="health_api_token", on_change=_reset_workspace
        )
        st.caption("凭据仅保存在当前浏览器会话对应的服务端会话中，不写入日志或共享缓存。")
        if st.button("检查连接"):
            try:
                with _client() as client:
                    if client.health().get("status") == "ok":
                        st.success("API 已连接")
                    else:
                        _error(ClientError("invalid_response"))
            except ClientError as exc:
                _error(exc)
        st.button("清除页面会话", on_click=_reset_workspace)
        st.button("清除凭据与页面会话", on_click=_reset_workspace, kwargs={"clear_token": True})
        st.caption("清除页面不会取消后端任务。取消请使用任务区域的按钮。")
    st.title("SmartHealth Agent")
    st.caption("生理数据分析 · 可追踪执行 · 有来源的知识解释")
    st.info("模型识别合成异常模式，分数不是临床患病概率。当前系统用于工程验证与展示。")
    file = st.file_uploader(
        "上传生理信号 CSV（最多 5 MiB）",
        type=["csv"],
        key=f"health_file_{st.session_state['health_upload_widget_version']}",
    )
    st.caption(
        "heart_rate · EDA · EEG_feature · activity · sleep_quality；每秒一行，每个分析窗口 30 秒。"
    )
    demo = resource_path("data", "sample", "demo.csv")
    if demo.is_file():
        st.download_button(
            "下载合成示例 CSV", demo.read_bytes(), file_name="demo.csv", mime="text/csv"
        )
    content = file.getvalue() if file is not None else None
    if content and len(content) > MAX_UPLOAD_BYTES:
        _error(ClientError("file_too_large"))
        content = None
    tools_tab, agent_tab = st.tabs(["数据工具", "Agent 工作台"])
    with tools_tab:
        st.subheader("可独立调用的数据分析")
        show_plots = st.checkbox("生成图表", value=True, key="health_tool_plots")
        first, second = st.columns(2)
        predict = first.button("快速预测", disabled=not content, width="stretch")
        analyze = second.button("完整数据分析", disabled=not content, width="stretch")
        if predict or analyze:
            try:
                with st.spinner("正在分析数据…"), _client() as client:
                    st.session_state["health_tool_result"] = (
                        client.predict(content)
                        if predict
                        else client.analyze(content, include_visualizations=show_plots)
                    )
            except ClientError as exc:
                st.session_state["health_tool_result"] = None
                _error(exc)
        result = st.session_state["health_tool_result"]
        if isinstance(result, AnalyzeResponse):
            _analysis(result)
        elif isinstance(result, PredictResponse):
            _prediction(result)
            for warning in result.warnings:
                st.warning(warning)
        if result is not None:
            st.download_button(
                "下载数据分析 JSON",
                result.model_dump_json(indent=2),
                file_name="analysis.json",
                mime="application/json",
            )
    with agent_tab:
        st.subheader("按任务选择执行路径")
        query = st.text_area("任务", value="解释 risk score", max_chars=4000, key="health_query")
        use_file = st.checkbox(
            "使用上方 CSV", value=bool(content), disabled=not content, key="health_agent_use_file"
        )
        plots = st.checkbox("任务结果包含图表", value=False, key="health_agent_plots")
        st.caption("简单定义无需上传；可尝试“分析这个 CSV”或“分析 CSV 并生成有来源的健康报告”。")
        current = st.session_state["health_run"]
        active = current is not None and current.status not in TERMINAL_STATUSES
        if st.button("开始任务", disabled=active or not query.strip(), type="primary"):
            try:
                session_id = st.session_state["health_session_id"]
                input_version = st.session_state["health_input_version"]
                refs = []
                if use_file and content:
                    with _client() as client:
                        uploaded = client.upload(content, session_id)
                    input_version, refs = uploaded.input_version, [uploaded.upload_ref]
                request = CreateRunRequest(
                    query=query,
                    session_id=session_id,
                    input_version=input_version,
                    upload_refs=refs,
                    include_visualizations=plots,
                )
                st.session_state["health_pending"] = {"request": request, "key": uuid4().hex}
                _submit_pending()
            except ClientError as exc:
                _error(exc)
        if st.session_state["health_pending"]:
            st.caption("提交结果不确定时，重试复用同一个幂等键，避免重复执行。")
            if st.button("重试提交"):
                try:
                    _submit_pending()
                except ClientError as exc:
                    _error(exc)
        with st.expander("重新连接已有任务"):
            run_id = st.text_input("Run ID", key="health_reconnect_run")
            session_id = st.text_input("Session ID", key="health_reconnect_session")
            if st.button("连接任务", disabled=not run_id or not session_id):
                try:
                    with _client() as client:
                        _accept_run(client.snapshot(run_id, session_id))
                except ClientError as exc:
                    _error(exc)
        _run_panel()


if __name__ == "__main__":
    main()
