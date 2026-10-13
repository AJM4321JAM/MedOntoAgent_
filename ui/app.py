"""MedOntoAgent | Clinical Agent Command Center v3.

Start backend: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Start UI:      python -m streamlit run ui/app.py
"""
from __future__ import annotations

import html
import os
from urllib.parse import urlparse

import requests
import streamlit as st

API_URL = os.getenv("MEDONTO_API_URL", "http://127.0.0.1:8000/query")
TIMEOUT_SECONDS = 180
EXAMPLES = {
    "Write my own question": "",
    "ACE inhibitor restriction (ontology path)": "What pregnancy-related restriction applies to ACE inhibitor therapy?",
    "Labetalol alternatives (guideline evidence)": "What alternatives are suggested when labetalol is not suitable for chronic hypertension during pregnancy?",
    "Pneumonia treatment (out-of-scope check)": "What antibiotic treatment does the corpus recommend for bacterial pneumonia during pregnancy?",
}

st.set_page_config(page_title="MedOntoAgent | Agent Command Center", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root{color-scheme:dark}
.stApp{background:#080f1c;color:#eef5ff}
[data-testid="stSidebar"]{background:#0b1628;border-right:1px solid #29415e}
.block-container{max-width:1840px;padding:1.2rem 2.3rem 3rem}
header[data-testid="stHeader"]{background:#080f1ce6}
h1,h2,h3{letter-spacing:-.025em;color:#f0f7ff!important}
h3{font-size:1.45rem!important;margin-top:.75rem!important}
.stMarkdown p, [data-testid="stWidgetLabel"] p{font-size:1rem;line-height:1.6}
.hero{background:radial-gradient(circle at 84% 22%,#126e8170,transparent 40%),linear-gradient(110deg,#132c48,#101b31);border:1px solid #355d7a;border-radius:18px;padding:1.4rem 1.8rem;margin-bottom:1.25rem}
.hero h1{font-size:2.8rem;margin:.2rem 0 .25rem}
.kicker{font-size:.82rem;letter-spacing:.13em;font-weight:800;color:#86d9f7}
.lead{color:#d0dfed;font-size:1.06rem;max-width:1150px}
.panel{background:linear-gradient(145deg,#12233b,#0e1b2e);border:1px solid #345270;border-radius:16px;padding:1.3rem 1.4rem;margin:.55rem 0 1.1rem}
.section-label{font-size:.86rem;letter-spacing:.09em;font-weight:800;color:#91cbff;margin-bottom:.8rem}
.big-answer{font-size:1.32rem;line-height:1.75;color:#fff}
.pill{display:inline-block;border:1px solid #587b9c;background:#253d57;color:#deebff;padding:.43rem .85rem;border-radius:25px;font-size:.9rem;font-weight:800;margin-right:.45rem;margin-bottom:.5rem}
.pill.green{color:#abf8df;background:#16453e;border-color:#378d77}
.pill.amber{color:#f8dda0;background:#49391d;border-color:#a38342}
.pill.red{color:#ffc8d1;background:#482630;border-color:#8b4757}
.subtle{font-size:.94rem;line-height:1.65;color:#b5c9e0}
.metricline{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7rem;margin-top:.75rem}
.metriccard{padding:1rem;background:#122740;border:1px solid #365675;border-radius:12px;min-width:0}
.metriccard .num{font-size:1.45rem;font-weight:850;color:#f6fbff;overflow-wrap:anywhere}
.metriccard .hint{font-size:.88rem;color:#bed1e4}
.workflow{background:linear-gradient(135deg,#12233b,#0b1b30);border:1px solid #365b79;border-radius:18px;padding:1.25rem 1.4rem;margin:.6rem 0 1.2rem}
.workflow-head{display:flex;justify-content:space-between;gap:1rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem}
.workflow-head strong{font-size:1.1rem}
.workflow-legend{font-size:.87rem;color:#acc7e4}
.route-start,.route-end{border:1px solid #49688a;border-radius:12px;padding:.75rem 1rem;text-align:center;background:#1b304b;font-weight:800;font-size:1.03rem}
.route-arrow{text-align:center;color:#79d7ed;font-size:1.6rem;line-height:1.3;margin:.12rem 0}
.route-branches{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.route-branch{border:2px solid #3b5570;border-radius:15px;padding:1rem 1.05rem;background:#10213a;min-width:0}
.route-branch.active{border-color:#49ceb9;background:#10323e;box-shadow:0 0 0 1px #317f81 inset}
.route-branch.inactive{opacity:.52;border-style:dashed}
.route-branch .branch-title{font-weight:850;font-size:1.1rem;color:#f2fbff;margin-bottom:.3rem}
.route-branch .branch-note{font-size:.9rem;color:#a9c6de;line-height:1.4;margin-bottom:.8rem}
.route-steps{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center}
.route-step{font-size:.87rem;font-weight:700;border:1px solid #46728e;border-radius:8px;padding:.48rem .68rem;color:#e4f5ff;background:#1c3950}
.route-step-arrow{color:#86cde1;font-size:1.2rem}
.workflow-note{margin-top:.9rem;font-size:.88rem;color:#adc4dd;line-height:1.5}
.evidence{border:1px solid #375c7a;border-radius:13px;background:#122641;padding:1.1rem;margin:.55rem 0}
.evidence-title{color:#f4faff;font-size:1.05rem;font-weight:800;overflow-wrap:anywhere}
.evidence-meta{font-size:.96rem;color:#bccfe2;margin:.5rem 0;overflow-wrap:anywhere}
a.source{color:#8fddff!important;text-decoration:none!important;font-weight:800;font-size:.98rem}
.empty{padding:1.2rem;background:#10233a;border:1px dashed #3b5875;border-radius:13px;color:#b6cce1;font-size:.95rem}
[data-testid="stTextArea"] textarea{background:#142840!important;color:#f8fbff!important;border:1px solid #497090!important;font-size:1.08rem!important}
[data-testid="stForm"]{border:1px solid #3b6081;border-radius:16px;padding:1.1rem;background:#101e31}
[data-testid="stFormSubmitButton"] button{background:#25a6bf!important;color:#061726!important;border:0!important;font-size:1.08rem!important;font-weight:850!important;min-height:3rem!important}
.footer{margin-top:1.5rem;color:#9db5ce;font-size:.86rem;border-top:1px solid #34516f;padding-top:1rem}
@media(max-width:1050px){.route-branches{grid-template-columns:1fr}.metricline{grid-template-columns:repeat(3,1fr)}.block-container{padding:1rem}.hero h1{font-size:2.1rem}}
</style>
""", unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def safe_link(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"https", "http"} and bool(parsed.netloc)
    except ValueError:
        return False


def query_backend(question: str) -> dict:
    try:
        r = requests.post(API_URL, json={"query": question}, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise RuntimeError("Could not connect to FastAPI. Confirm Uvicorn is running on port 8000.") from exc
    if r.status_code == 503:
        raise RuntimeError("The provider request budget was exhausted or the service is temporarily unavailable.")
    if not r.ok:
        raise RuntimeError(f"The API returned HTTP {r.status_code}; no answer was displayed.")
    try:
        result = r.json()
    except ValueError as exc:
        raise RuntimeError("The API returned invalid JSON.") from exc
    if not isinstance(result, dict) or not all(k in result for k in ("status", "execution_mode", "answer")):
        raise RuntimeError("The API returned an unexpected response shape.")
    return result


def state_info(result: dict) -> tuple[str, str, str]:
    mode = result.get("execution_mode")
    status = result.get("status")
    if mode == "MEDONTO" and status == "VERIFIED":
        return "ONTOLOGY VERIFIED", "green", "The backend reports that the answer passed its required evidence and ontology checks."
    if status == "ABSTAINED":
        return "ABSTAINED", "amber", "The system did not return a sufficiently supported answer."
    if mode == "BASELINE_FALLBACK":
        return "BASELINE FALLBACK", "amber", "The ontology-controlled path did not complete; baseline answered without ontology verification."
    if mode == "BASELINE_ROUTED":
        return "BASELINE ANSWER", "", "The controller selected conventional evidence-grounded RAG. Not ontology verified."
    return "NOT ONTOLOGY VERIFIED", "amber", "Do not interpret this result as ontology verified."


def route_panel(result: dict | None) -> None:
    """Post-response route visualization, not a fabricated live node trace."""
    mode = result.get("execution_mode") if result else None
    status = result.get("status") if result else None
    medonto = mode == "MEDONTO"
    baseline = mode in {"BASELINE_ROUTED", "BASELINE_FALLBACK"}
    ontology_count = len(result.get("ontology_constraints") or []) if result else 0
    retrieval = result.get("retrieval") or {} if result else {}
    label = "ARCHITECTURE PREVIEW" if not result else f"OBSERVED RESULT · {esc(mode)} / {esc(status)}"
    med_class = "active" if medonto else "inactive" if baseline else ""
    base_class = "active" if baseline else "inactive" if medonto else ""
    med_note = (
        f"{ontology_count} eligible rule(s) returned · ontology filter "
        f"{'applied' if retrieval.get('used_ontology_filter') else 'not applied'}"
        if medonto else "Constrained evidence search and answer verification"
    )
    base_note = (
        "Fallback after MedOnto could not complete" if mode == "BASELINE_FALLBACK"
        else "Selected when no eligible ontology control applied"
        if mode == "BASELINE_ROUTED" else "Conventional evidence-grounded response"
    )
    outcome = "Awaiting a question" if not result else f"{esc(status)} · {esc(mode)}"
    st.markdown(f"""
    <div class="workflow">
      <div class="workflow-head"><strong>LANGGRAPH · AGENT DECISION FLOW</strong><span class="workflow-legend">{label}</span></div>
      <div class="route-start">Question received → clinical concept mapping → ontology eligibility decision</div>
      <div class="route-arrow">↓</div>
      <div class="route-branches">
        <div class="route-branch {med_class}">
          <div class="branch-title">◆ Ontology-controlled path</div>
          <div class="branch-note">{esc(med_note)}</div>
          <div class="route-steps"><span class="route-step">Constraint selection</span><span class="route-step-arrow">→</span><span class="route-step">Qdrant retrieval</span><span class="route-step-arrow">→</span><span class="route-step">Gemini generation</span><span class="route-step-arrow">→</span><span class="route-step">Dual validation</span></div>
        </div>
        <div class="route-branch {base_class}">
          <div class="branch-title">◇ Baseline RAG path</div>
          <div class="branch-note">{esc(base_note)}</div>
          <div class="route-steps"><span class="route-step">Evidence retrieval</span><span class="route-step-arrow">→</span><span class="route-step">Grounded generation</span><span class="route-step-arrow">→</span><span class="route-step">Answer / abstain</span></div>
        </div>
      </div>
      <div class="route-arrow">↓</div>
      <div class="route-end">Agent response decision: {outcome}</div>
      <div class="workflow-note">Highlighted branch is derived from the actual API execution mode. This is a workflow summary, not a live per-node event trace. The alternative path is shown for architectural context.</div>
    </div>""", unsafe_allow_html=True)


def ontology_svg(constraint: dict) -> str:
    """Render only the actual returned triple; no inferred edges."""
    subject = esc(constraint.get("subject_name", "Unknown"))
    predicate = esc(constraint.get("predicate_name", "relatedTo"))
    obj = esc(constraint.get("object_name", "Unknown"))
    # HTML text is escaped; SVG foreign text remains text, not markup.
    return (f'<svg width="100%" viewBox="0 0 740 145" role="img" aria-label="Ontology relationship: {subject}, {predicate}, {obj}" xmlns="http://www.w3.org/2000/svg">'
            '<defs><linearGradient id="ontology-bg" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#172c49"/><stop offset="1" stop-color="#143744"/></linearGradient></defs>'
            '<rect x="12" y="44" width="218" height="60" rx="15" fill="url(#ontology-bg)" stroke="#5394ce" stroke-width="2"/>'
            f'<text x="121" y="79" font-size="16" font-family="sans-serif" text-anchor="middle" fill="#e8f5ff" font-weight="bold">{subject}</text>'
            '<path d="M230 74 H510" stroke="#6bcbbf" stroke-width="3"/><path d="M497 65 L511 74 L497 83" stroke="#6bcbbf" stroke-width="3" fill="none"/>'
            '<rect x="250" y="16" width="240" height="37" rx="12" fill="#1b3c43" stroke="#43978f"/>'
            f'<text x="370" y="40" font-size="15" font-family="sans-serif" text-anchor="middle" fill="#99f4db">{predicate}</text>'
            '<rect x="511" y="44" width="217" height="60" rx="15" fill="#203651" stroke="#668ec8" stroke-width="2"/>'
            f'<text x="619" y="79" font-size="16" font-family="sans-serif" text-anchor="middle" fill="#edf5ff" font-weight="bold">{obj}</text>'
            '</svg>')


def show_ontology(result: dict | None) -> None:
    st.markdown('<div class="section-label">ONTOLOGY INTELLIGENCE</div>', unsafe_allow_html=True)
    constraints = (result or {}).get("ontology_constraints") or []
    if not constraints:
        st.markdown('<div class="empty">No eligible ontology relationship was returned for this question.</div>' if result else '<div class="empty">Actual ontology relationships will appear here after a qualifying query.</div>', unsafe_allow_html=True)
        return
    for c in constraints:
        st.markdown(f'<div class="panel">{ontology_svg(c)}<div class="subtle">Returned constraint type: <strong>{esc(c.get("constraint_type", "Unspecified"))}</strong>. This displays a returned relationship, not a newly inferred medical fact.</div></div>', unsafe_allow_html=True)


def show_sources(result: dict) -> None:
    st.markdown('<div class="section-label">SOURCE EVIDENCE</div>', unsafe_allow_html=True)
    citations = result.get("citations") or []
    if not citations:
        st.markdown('<div class="empty">No verified source citations were returned with this result.</div>', unsafe_allow_html=True)
        return
    for n, c in enumerate(citations, 1):
        url = c.get("source_url")
        link = f'<a class="source" target="_blank" rel="noopener noreferrer" href="{esc(url)}">Open original guideline ↗</a>' if safe_link(url) else ""
        page = f'Page {esc(c["page_number"])}' if c.get("page_number") is not None else "Page not specified"
        st.markdown(f'<div class="evidence"><div class="evidence-title">{n:02d} · {esc(c.get("title") or "Untitled source")}</div><div class="evidence-meta">{esc(c.get("section_title") or "Section unavailable")} · {page}</div>{link}</div>', unsafe_allow_html=True)
    st.caption("Source links come from the backend citation metadata; a citation does not imply independent clinical endorsement.")


def load_example() -> None:
    selected = st.session_state.get("example_choice", "Write my own question")
    if EXAMPLES[selected]:
        st.session_state["question_input"] = EXAMPLES[selected]


with st.sidebar:
    st.markdown("### MEDONTOAGENT")
    st.caption("Clinical agent command center · v3")
    st.markdown("**Core architecture**")
    st.write("LangGraph orchestration")
    st.write("RDF ontology constraints")
    st.write("Qdrant evidence retrieval")
    st.write("Gemini generation")
    st.write("Citation + ontology validation")
    st.divider()
    st.caption("Backend endpoint")
    st.code(API_URL, language=None)
    st.info("Research/education prototype. Not for patient care or independent clinical decisions.")
    if st.button("Clear result", use_container_width=True):
        st.session_state.pop("answer_response", None)
        st.session_state.pop("question_submitted", None)
        st.rerun()

st.markdown('<div class="hero"><div class="kicker">ONTOLOGY-CONSTRAINED RAG · AGENTIC CLINICAL INTELLIGENCE</div><h1>MedOntoAgent</h1><div class="lead">Watch the controller choose its execution path, apply medical knowledge constraints, retrieve approved evidence and decide whether an answer can be returned.</div></div>', unsafe_allow_html=True)

left, right = st.columns([1.25, .75], gap="large")
with left:
    st.markdown("### Ask the clinical agent")
    st.selectbox("Try a demonstration question", list(EXAMPLES), key="example_choice", on_change=load_example)
    with st.form("ask", clear_on_submit=False):
        question = st.text_area("Clinical question", key="question_input", height=110, placeholder="Ask about pregnancy-specific treatment guidance, restrictions or evidence...")
        submitted = st.form_submit_button("▶ Run MedOntoAgent", type="primary", use_container_width=True)
    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Running the real MedOntoAgent backend. Awaiting final outcome..."):
                try:
                    result = query_backend(question.strip())
                    st.session_state["answer_response"] = result
                    st.session_state["question_submitted"] = question.strip()
                except RuntimeError as exc:
                    st.session_state.pop("answer_response", None)
                    st.session_state.pop("question_submitted", None)
                    st.error(str(exc))
with right:
    st.markdown("### Current run")
    response = st.session_state.get("answer_response")
    if response:
        label, color, description = state_info(response)
        st.markdown(f'<div class="panel"><div class="section-label">RESPONSE DECISION</div><span class="pill {color}">{esc(label)}</span><span class="pill">{esc(response.get("execution_mode", "UNKNOWN"))}</span><div class="subtle">{esc(description)}</div></div>', unsafe_allow_html=True)
        retrieval = response.get("retrieval") or {}
        metrics = [
            (str(retrieval.get("result_count", "—")), "Retrieved chunks"),
            ("Applied" if retrieval.get("used_ontology_filter") else "Not applied", "Ontology filter"),
            ("Yes" if retrieval.get("retrieval_relaxed") else "No", "Retrieval relaxed"),
        ]
        cards = "".join(f'<div class="metriccard"><div class="num">{esc(val)}</div><div class="hint">{esc(lbl)}</div></div>' for val, lbl in metrics)
        st.markdown(f'<div class="metricline">{cards}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel"><div class="section-label">READY FOR EXECUTION</div><div class="subtle">Ask a question to populate the agent pathway, ontology relationships, evidence and response decision with real API data.</div></div>', unsafe_allow_html=True)

st.markdown("### Agent execution center · selected route")
route_panel(st.session_state.get("answer_response"))

response = st.session_state.get("answer_response")
if response:
    st.markdown("### Clinical response · verified evidence")
    st.markdown(f'<div class="panel"><div class="section-label">ANSWER · {esc(response.get("status", "UNKNOWN"))}</div><div class="big-answer">{esc(response.get("answer") or "No answer was returned.")}</div></div>', unsafe_allow_html=True)
    lower_left, lower_right = st.columns([1, 1], gap="large")
    with lower_left:
        show_ontology(response)
    with lower_right:
        show_sources(response)
else:
    st.markdown("### Knowledge controls and supporting evidence")
    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        show_ontology(None)
    with lower_right:
        st.markdown('<div class="section-label">SOURCE EVIDENCE</div><div class="empty">Verified citation metadata and guideline links will appear here after a supported answer.</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">MedOntoAgent · University research prototype · Automated verification is not a guarantee of clinical correctness. Execution visualization is a summary of the public API result, not a per-node telemetry trace.</div>', unsafe_allow_html=True)
