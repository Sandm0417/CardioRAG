"""CardioRAG v2.4 — 3-tab Streamlit app: Overview / Screening / Chat.

Authoritative demo UI (executor phase6 gate target). Run:
    streamlit run app/streamlit_app.py

Tabs:
  1. 📊 Overview  — agent intro + LightRAG build status + guardrail/rule-engine
                    status + index-freshness pill (vs current rules.yaml)
  2. 🛡️ Screening — guardrail pre/post inspection on pasted text
  3. 💬 Chat      — RAG-augmented Q&A (DeepSeek when API available; graceful
                    degradation to guardrail-only response otherwise)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# Make src importable
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from cardiorag.guardrail import CVGuardrail  # noqa: E402
from cardiorag.normalize import detect_emergency, detect_out_of_bounds, normalize_input  # noqa: E402

st.set_page_config(page_title="CardioRAG（心智脉）", page_icon="🫀", layout="wide")
st.title("🫀 CardioRAG（心智脉）")
st.caption("Local research demo. Bind to 127.0.0.1. Chat uses a local key only if .env is present.")

# ── helpers ─────────────────────────────────────────────────────────────
@st.cache_data
def load_progress() -> dict:
    p = ROOT / ".executor" / "progress.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def graph_stats() -> dict:
    """Read node and edge counts from the current persisted GraphML file."""
    graph = ROOT / "lightrag_index" / "graph_chunk_entity_relation.graphml"
    if not graph.exists():
        return {"nodes": None, "edges": None}
    nodes = edges = 0
    for _event, elem in ET.iterparse(graph, events=("end",)):
        if elem.tag.endswith("node"):
            nodes += 1
        elif elem.tag.endswith("edge"):
            edges += 1
        elem.clear()
    return {"nodes": nodes, "edges": edges}


def freshness_pill(progress: dict) -> str:
    """Report the current persisted graph, never a historical progress snapshot."""
    graph = graph_stats()
    phase4 = (progress.get("phases") or {}).get("phase4") or {}
    if graph["nodes"] is not None:
        return (
            "🟢 local LightRAG index present. "
            f"Current persisted graph: {graph['nodes']} nodes / {graph['edges']} edges. "
            "Manuscript counts: raw registry 401/501, persisted index 369/489."
        )
    if phase4:
        return (
            "🟠 local GraphML index is missing. "
            "The frozen manuscript counts are 369 nodes / 489 edges; "
            "historical progress-file counts are not displayed."
        )
    return "🔴 no local index; frozen manuscript counts are 369 nodes / 489 edges"


@st.cache_data
def build_stats() -> dict:
    """Count CardioKG JSONL as node/relation rows, not as head-tail edges."""
    node_rows = 0
    relation_rows = 0
    names: set[str] = set()
    other = 0
    kg_dir = ROOT / "data" / "cardiokg"
    for kgf in sorted(kg_dir.glob("*.jsonl")):
        with open(kgf, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                kind = r.get("type")
                if kind == "node":
                    node_rows += 1
                    name = r.get("name")
                    if name:
                        names.add(str(name))
                elif kind == "relation":
                    relation_rows += 1
                else:
                    other += 1
    return {
        "nodes": node_rows,
        "unique_names": len(names),
        "relations": relation_rows,
        "other": other,
    }


@st.cache_data
def scenario_counts() -> dict:
    """Count the bundled virtual scenarios by split."""
    scen_dir = ROOT / "data" / "scenarios"
    counts = {"development": 0, "held_out_synthetic": 0}
    for filename, key in (("internal.jsonl", "development"), ("external.jsonl", "held_out_synthetic")):
        path = scen_dir / filename
        if path.exists():
            counts[key] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return counts


# ── session state ───────────────────────────────────────────────────────
if "guardrail" not in st.session_state:
    st.session_state.guardrail = CVGuardrail()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab_overview, tab_screening, tab_chat = st.tabs(
    ["📊 Overview", "🛡️ Screening", "💬 Chat"]
)

# ── Tab 1: Overview ─────────────────────────────────────────────────────
with tab_overview:
    st.subheader("🧩 Agent 简介")
    st.markdown(
        "CardioRAG 是面向**心血管疾病（CAD / HF / AF）患者–医师沟通**的专科对话辅助系统："
        "CardioKG 指南知识图谱 + LightRAG 图检索 + DeepSeek 生成 + CV-Safety Guardrail（输入急症拦截 / 输出越界转诊）。"
        "评测使用 120 个标准化虚拟场景（开发集 96 + 留出合成集 24），不是真实外部验证队列。"
    )

    progress = load_progress()
    st.subheader("📊 CardioKG 与索引")
    phase4 = (progress.get("phases") or {}).get("phase4") or {}
    kg = build_stats()
    index = graph_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("原始登记实体", kg["nodes"])
    c2.metric("原始登记关系", kg["relations"])
    c3.metric("唯一实体名", kg["unique_names"])
    c4.metric(
        "持久化索引",
        f"{index['nodes']}/{index['edges']}" if index["nodes"] is not None else "369/489",
    )
    st.caption(
        "原始登记按 JSONL 的 node / relation 行计数，应为 401 实体 / 501 关系；"
        f"唯一 `name` 为 {kg['unique_names']}。"
        "当前持久化索引为 369 nodes / 489 edges；历史 progress 文件不用于显示索引数字。"
    )

    st.subheader("🛡️ 规则与场景")
    scen = scenario_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("安全规则来源", "代码内 Guardrail")
    c2.metric("rules.yaml", "空文件（有意）")
    c3.metric("虚拟场景", f"{scen['development']} + {scen['held_out_synthetic']}")
    st.caption(
        "`configs/rules.yaml` 为空是设计如此，急诊/超界规则在 `normalize.py` 与 `guardrail.py`。"
        f"`data/scenarios/` 含开发集 {scen['development']} 例与留出合成集 {scen['held_out_synthetic']} 例；"
        "`external.jsonl` 是历史文件名，该留出集由同一流程构建，不是真实外部临床验证队列。"
    )

    st.subheader("🧪 本地索引")
    st.markdown(freshness_pill(progress))

# ── Tab 2: Screening ────────────────────────────────────────────────────
with tab_screening:
    st.subheader("🛡️ 护栏检测演示（输入侧 + 输出侧）")
    probe = st.text_area(
        "粘贴患者问题 / AI 回答进行检测：",
        placeholder="例如：突然胸口剧痛伴冷汗…",
        height=100,
    )
    if st.button("运行检测", type="primary"):
        if probe.strip():
            is_emg, kw = detect_emergency(probe)
            is_oob, pat = detect_out_of_bounds(probe)
            pre = st.session_state.guardrail.pre_generation_check(probe)
            st.markdown("**输入侧（pre-generation）**")
            st.write({"emergency_detected": is_emg, "keywords": kw, "blocked": pre["block"]})
            st.markdown("**输出侧（post-generation）**")
            st.write({"oob_detected": is_oob, "patterns": pat, "sanitized": is_oob})
            if is_emg:
                st.success("🚨 触发紧急路由：将直接引导拨打 120 / 前往急诊")
            elif is_oob:
                st.warning("⚠️ 触发越界转诊：将替换为医师预约建议")
            else:
                st.info("未触发拦截。")
        else:
            st.warning("请输入检测文本。")

# ── Tab 3: Chat ─────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("💬 RAG 增强问答")
    st.caption(
        "问题先经 CV-Guardrail 输入检测。仅当 DeepSeek API 可用时才生成回答；"
        "证书或网络失败会明确报 API 不可用，不会伪装成护栏回复。"
    )

    from cardiorag.env import load_env

    load_env(ROOT)
    _env = {
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
    }

    def llm_answer(question: str) -> tuple[str, str | None]:
        """Return (answer, error). error is set when the API call did not succeed."""
        if not _env.get("DEEPSEEK_API_KEY"):
            return "", "未配置 DEEPSEEK_API_KEY，无法调用生成接口。"
        try:
            from openai import OpenAI

            client = OpenAI(api_key=_env.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一名心血管专科医师-患者沟通助手。请基于临床指南（ESC/AHA/CSC 2024）回答患者问题，注意：不提供具体药物剂量、不指导停药、不做手术决策；紧急症状引导拨打120。"},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=300,
                timeout=25,
            )
            return resp.choices[0].message.content or "", None
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "CERTIFICATE_VERIFY_FAILED" in msg or "SSL" in msg:
                return "", (
                    "DeepSeek API 证书校验失败（SSL: CERTIFICATE_VERIFY_FAILED）。"
                    "这是本机 Python/代理的 TLS 问题，不是护栏拦截，也不是模型回答。"
                )
            return "", f"DeepSeek API 调用失败：{msg[:160]}"

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("输入您的心血管问题…（例如：LVEF 35% 该注意什么？）")
    if user_input and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            pre = st.session_state.guardrail.pre_generation_check(user_input)
            if pre["block"]:
                reply = pre["response"]
                st.write(reply)
            else:
                reply, api_error = llm_answer(user_input)
                if api_error:
                    reply = f"API 不可用：{api_error}"
                    st.error(reply)
                else:
                    post = st.session_state.guardrail.post_generation_check(reply)
                    if post["flagged"]:
                        reply = post["sanitized_response"]
                    reply = st.session_state.guardrail.append_disclaimer(reply)
                    st.write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

st.divider()
st.caption("CardioRAG demo | CardioKG + LightRAG + CV-Guardrail | 本页为本地演示，不是投稿包状态板")
