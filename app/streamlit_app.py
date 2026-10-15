import json, time, pathlib, streamlit as st
from pypdf import PdfReader
from rag import build_store, answer

DATA = pathlib.Path(__file__).parents[1] / "data"
st.set_page_config(page_title="Resume & Career Chatbot", layout="wide")
st.title("AI Resume Tailoring & Career Guidance")

@st.cache_resource
def store():
    return build_store([p.read_text() for p in DATA.glob("*.md")])

with st.sidebar:
    up = st.file_uploader("Upload resume (PDF)", type="pdf")
    job = st.text_area("Paste job description")
resume = " ".join((p.extract_text() or "") for p in PdfReader(up).pages) if up else ""

if "msgs" not in st.session_state: st.session_state.msgs = []
for m in st.session_state.msgs:
    st.chat_message(m["role"]).write(m["content"])

if q := st.chat_input("Ask how to tailor your resume..."):
    st.session_state.msgs.append({"role": "user", "content": q}); st.chat_message("user").write(q)
    t0 = time.time(); a = answer(store(), q, resume, job)
    st.session_state.msgs.append({"role": "assistant", "content": a})
    st.chat_message("assistant").write(a); st.caption(f"{time.time()-t0:.1f}s")
    c1, c2 = st.columns(10)[:2]
    def fb(v):
        with open(DATA / "feedback.jsonl", "a") as f: f.write(json.dumps({"q": q, "a": a, "vote": v}) + "\n")
    c1.button("👍", on_click=fb, args=(1,)); c2.button("👎", on_click=fb, args=(-1,))
