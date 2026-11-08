import html
import random
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Encoder vs Decoder", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- config
BERT_ID = "TahiraAbbas/bert-multilingual-imdb-sentiment"
GPT_ID = "TahiraAbbas/gpt2-wikitext"
BERT_URL = f"https://huggingface.co/{BERT_ID}"
GPT_URL = f"https://huggingface.co/{GPT_ID}"
LABELS = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1100px;}
.hero {padding: 2rem 2.2rem; border-radius: 22px; color: #ffffff; margin-bottom: 1rem;}
.hero.enc {background: linear-gradient(120deg, #0f172a 0%, #0e7490 60%, #06b6d4 130%);
    box-shadow: 0 12px 32px rgba(14,116,144,0.35);}
.hero.dec {background: linear-gradient(120deg, #1c1917 0%, #b45309 60%, #f59e0b 130%);
    box-shadow: 0 12px 32px rgba(180,83,9,0.35);}
.hero h1 {margin: 0; font-size: 2.3rem; color: #ffffff;}
.hero p {margin: 0.4rem 0 0 0; font-size: 1.05rem; color: #f1f5f9;}
.pill {display: inline-block; padding: 0.25rem 0.8rem; margin: 0.8rem 0.4rem 0 0;
    border-radius: 999px; background: rgba(255,255,255,0.18); font-size: 0.85rem; color: #ffffff;}
.mcard {border-radius: 16px; padding: 1.1rem 1.3rem; border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.06); line-height: 1.7;}
.mcard.enc {border-top: 6px solid #06b6d4;}
.mcard.dec {border-top: 6px solid #f59e0b;}
.side {border-radius: 12px; padding: 0.8rem 0.9rem; margin-bottom: 0.7rem; font-size: 0.88rem;
    border: 1px solid rgba(128,128,128,0.3); background: rgba(128,128,128,0.08); line-height: 1.5;}
.side.enc {border-left: 5px solid #06b6d4;}
.side.dec {border-left: 5px solid #f59e0b;}
.result {border-radius: 16px; padding: 1.3rem; text-align: center; color: #ffffff;
    font-size: 1.8rem; font-weight: 800; margin-bottom: 0.8rem;}
.pos {background: linear-gradient(135deg, #047857, #10b981);}
.neg {background: linear-gradient(135deg, #be123c, #f43f5e);}
.gen {border-left: 5px solid #f59e0b; border-radius: 12px; padding: 1rem 1.2rem;
    background: rgba(245,158,11,0.09); line-height: 1.65; margin-bottom: 0.7rem;}
.gen .prompt {background: rgba(245,158,11,0.3); padding: 0 4px; border-radius: 4px; font-weight: 700;}
.small {font-size: 0.85rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- models (only one kept in memory)
@st.cache_resource(max_entries=1)
def load_model(name):
    if name == "encoder":
        return pipeline("sentiment-analysis", model=BERT_ID)
    return pipeline("text-generation", model=GPT_ID)

@st.cache_data
def load_samples():
    try:
        return pd.read_csv("sample_reviews.csv")["review"].tolist()
    except Exception:
        return []

samples = load_samples()

st.session_state.setdefault("bert_text", "The movie was excellent.")
st.session_state.setdefault("gpt_prompt", "Artificial intelligence will")
st.session_state.setdefault("bert_history", [])
st.session_state.setdefault("gpt_history", [])

# ================================================================ sidebar: section navigation
with st.sidebar:
    st.header("Sections")
    section = st.radio("Choose a section", ["🔍 Encoder: BERT", "✍️ Decoder: GPT-2"],
                       label_visibility="collapsed")
    st.markdown(f"""
<div class="side enc"><b>🔍 Encoder</b> · multilingual BERT<br>
Dataset: IMDB movie reviews<br><a href="{BERT_URL}" target="_blank">Open on Hugging Face</a></div>
<div class="side dec"><b>✍️ Decoder</b> · monolingual GPT-2<br>
Dataset: WikiText-2 (Wikipedia)<br><a href="{GPT_URL}" target="_blank">Open on Hugging Face</a></div>
""", unsafe_allow_html=True)
    st.caption("Two different datasets: IMDB for the encoder, WikiText-2 for the decoder.")
    st.caption("Models run on CPU, so the first request can take a while.")
    if st.button("Clear history", use_container_width=True):
        st.session_state["bert_history"] = []
        st.session_state["gpt_history"] = []
        st.rerun()

# ================================================================ ENCODER SECTION
def encoder_section():
    st.markdown("""
<div class="hero enc">
  <h1>🔍 Encoder: Multilingual BERT</h1>
  <p>An encoder-only Transformer that reads the whole text and predicts POSITIVE or NEGATIVE.</p>
  <span class="pill">Encoder-only</span><span class="pill">Multilingual (104 languages)</span>
  <span class="pill">IMDB dataset</span><span class="pill">Hugging Face</span>
</div>""", unsafe_allow_html=True)

    tab_run, tab_info = st.tabs(["🔍 Classify", "📚 Model & Dataset"])

    with tab_run:
        st.caption("Fine-tuned on English IMDB reviews. Try other languages to see cross-lingual transfer.")
        st.write("**Try an example:**")
        ex = st.columns(4)
        examples = ["The movie was excellent.", "La película fue excelente.",
                    "Ce film était terrible.", "یہ فلم بہت اچھی تھی۔"]
        for col, e in zip(ex, examples):
            if col.button(e, key="enc_ex_" + e, use_container_width=True):
                st.session_state["bert_text"] = e
        if samples and st.button("🎲 Load a random IMDB review", key="enc_rand"):
            st.session_state["bert_text"] = random.choice(samples)

        text = st.text_area("Input text", key="bert_text", height=150)
        st.markdown(f"<span class='small'>{len(text.split())} words</span>", unsafe_allow_html=True)

        if st.button("Classify", type="primary", key="enc_run"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Loading BERT and classifying..."):
                    clf = load_model("encoder")
                    res = clf(text, truncation=True, max_length=512, top_k=None)
                    if res and isinstance(res[0], list):
                        res = res[0]
                scores = {LABELS.get(r["label"], r["label"]): r["score"] for r in res}
                best = max(scores, key=scores.get)
                css = "pos" if best == "POSITIVE" else "neg"
                icon = "😊" if best == "POSITIVE" else "😞"
                st.markdown(f"<div class='result {css}'>{icon} {best}</div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric("Confidence", f"{scores[best]:.1%}")
                c2.metric("Words analysed", min(len(text.split()), 512))
                for lab in ["POSITIVE", "NEGATIVE"]:
                    if lab in scores:
                        st.write(f"{lab}: {scores[lab]:.1%}")
                        st.progress(float(scores[lab]))
                with st.expander("Raw model output"):
                    st.json(res)
                st.session_state["bert_history"].insert(0, {
                    "Input": text[:80] + ("..." if len(text) > 80 else ""),
                    "Output": best, "Confidence": f"{scores[best]:.1%}"})

        if st.session_state["bert_history"]:
            st.markdown("#### History")
            st.dataframe(pd.DataFrame(st.session_state["bert_history"]),
                         use_container_width=True, hide_index=True)

    with tab_info:
        st.markdown(f"""
<div class="mcard enc">
<h3>🔍 Multilingual BERT</h3>
<b>Architecture:</b> encoder-only (bidirectional attention)<br>
<b>Base model:</b> bert-base-multilingual-cased (104 languages)<br>
<b>Fine-tuned model:</b> <a href="{BERT_URL}" target="_blank">{BERT_ID}</a><br>
<b>Task:</b> sentiment classification (POSITIVE / NEGATIVE)<br>
<b>Dataset:</b> IMDB movie reviews (labelled positive / negative)<br>
<b>Why this dataset:</b> classification needs labelled examples.<br>
<b>Pre-training:</b> masked word prediction<br>
<b>Best for:</b> classification, NER, question answering
</div>""", unsafe_allow_html=True)
        st.success("The encoder reads the whole input at once, using words on both the left and the right.")

# ================================================================ DECODER SECTION
def decoder_section():
    st.markdown("""
<div class="hero dec">
  <h1>✍️ Decoder: Monolingual GPT-2</h1>
  <p>A decoder-only Transformer that predicts the next token and continues your text.</p>
  <span class="pill">Decoder-only</span><span class="pill">Monolingual (English)</span>
  <span class="pill">WikiText-2 dataset</span><span class="pill">Hugging Face</span>
</div>""", unsafe_allow_html=True)

    tab_run, tab_info = st.tabs(["✍️ Generate", "📚 Model & Dataset"])

    with tab_run:
        st.caption("Fine-tuned on WikiText-2. GPT-2 is monolingual: it works with English only.")
        st.write("**Try a prompt:**")
        pr = st.columns(4)
        prompts = ["Artificial intelligence will", "The history of", "Once upon a time", "In the future,"]
        for col, p in zip(pr, prompts):
            if col.button(p, key="dec_pr_" + p, use_container_width=True):
                st.session_state["gpt_prompt"] = p

        prompt = st.text_input("Start of a sentence", key="gpt_prompt")
        c1, c2, c3, c4 = st.columns(4)
        max_new = c1.slider("New tokens", 10, 100, 40)
        temp = c2.slider("Temperature", 0.1, 1.5, 0.8)
        top_k = c3.slider("Top-k", 1, 100, 50)
        n_out = c4.slider("Variations", 1, 3, 2)
        st.caption("Higher temperature = more creative. Lower = more predictable.")

        if st.button("Generate", type="primary", key="dec_run"):
            if not prompt.strip():
                st.warning("Please enter a prompt.")
            else:
                with st.spinner("Loading GPT-2 and generating..."):
                    gen = load_model("decoder")
                    outs = gen(prompt, max_new_tokens=max_new, do_sample=True, temperature=temp,
                               top_k=top_k, num_return_sequences=n_out, pad_token_id=50256)
                st.markdown("#### Generated text")
                for i, o in enumerate(outs, 1):
                    full = o["generated_text"]
                    cont = full[len(prompt):] if full.startswith(prompt) else full
                    st.markdown(
                        f"<div class='gen'><b>Version {i}</b><br>"
                        f"<span class='prompt'>{html.escape(prompt)}</span>{html.escape(cont)}</div>",
                        unsafe_allow_html=True)
                    st.session_state["gpt_history"].insert(0, {"Prompt": prompt, "Generated": full})
                st.info("Press Generate again: the output changes because GPT-2 samples the next token.")

        if st.session_state["gpt_history"]:
            st.markdown("#### History")
            st.dataframe(pd.DataFrame(st.session_state["gpt_history"][:10]),
                         use_container_width=True, hide_index=True)

    with tab_info:
        st.markdown(f"""
<div class="mcard dec">
<h3>✍️ GPT-2</h3>
<b>Architecture:</b> decoder-only (causal attention)<br>
<b>Base model:</b> gpt2 (English only)<br>
<b>Fine-tuned model:</b> <a href="{GPT_URL}" target="_blank">{GPT_ID}</a><br>
<b>Task:</b> text generation<br>
<b>Dataset:</b> WikiText-2 (Wikipedia articles)<br>
<b>Why this dataset:</b> text generation learns from plain text and needs no labels.<br>
<b>Pre-training:</b> next-token prediction<br>
<b>Best for:</b> text generation, completion
</div>""", unsafe_allow_html=True)
        st.success("The decoder generates text one token at a time, using only the previous tokens.")

# ================================================================ route
if section.startswith("🔍"):
    encoder_section()
else:
    decoder_section()
