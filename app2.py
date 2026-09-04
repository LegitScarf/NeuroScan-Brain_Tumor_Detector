import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Optional deps – kept lazy so the file still opens in "demo mode" without them
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import tensorflow as tf  # type: ignore
    HAS_TF = True
except Exception:
    HAS_TF = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI — MRI Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_LABELS = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = (150, 150)
MODEL_PATH = "brain_tumor_model_quantized.tflite"

CLASS_META = {
    "glioma_tumor": {
        "label": "Glioma",
        "color": "#7C3AED",           # violet
        "soft": "rgba(124,58,237,0.10)",
        "desc": (
            "Gliomas arise from glial cells in the brain or spine. They can be low- or "
            "high-grade and often appear in the cerebral hemispheres. Early identification "
            "supports treatment planning."
        ),
    },
    "meningioma_tumor": {
        "label": "Meningioma",
        "color": "#0EA5A4",           # teal
        "soft": "rgba(14,165,164,0.12)",
        "desc": (
            "Meningiomas grow from the meninges — membranes surrounding the brain. Most are "
            "benign and slow-growing, but their location can still cause pressure symptoms."
        ),
    },
    "no_tumor": {
        "label": "No Tumor",
        "color": "#10B981",           # green
        "soft": "rgba(16,185,129,0.12)",
        "desc": (
            "No visual features consistent with a tumor were detected in the provided slice. "
            "This is not a diagnosis — clinical review is always required."
        ),
    },
    "pituitary_tumor": {
        "label": "Pituitary",
        "color": "#F59E0B",           # amber
        "soft": "rgba(245,158,11,0.14)",
        "desc": (
            "Pituitary tumors form in the pituitary gland at the base of the brain. Most are "
            "benign adenomas but can influence hormones and vision."
        ),
    },
}


# ---------------------------------------------------------------------------
# Styling — light clinical theme (off-white, teal + violet accents)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        :root{
            --bg:#F7F7F3;
            --card:#FFFFFF;
            --ink:#0B1220;
            --muted:#5B6474;
            --line:rgba(11,18,32,0.08);
            --teal:#0EA5A4;
            --violet:#7C3AED;
            --shadow: 0 10px 40px -20px rgba(11,18,32,0.25);
        }
        html, body, [class*="css"] { font-family:'Manrope', system-ui, sans-serif; color:var(--ink); }
        .stApp { background:
            radial-gradient(1200px 500px at -10% -10%, rgba(14,165,164,0.10), transparent 60%),
            radial-gradient(900px 500px at 110% 10%, rgba(124,58,237,0.10), transparent 60%),
            var(--bg);
        }
        header[data-testid="stHeader"]{ background:transparent; }
        .block-container{ padding-top:1.2rem; padding-bottom:4rem; max-width:1180px;}
        #MainMenu, footer { visibility:hidden; }

        /* ---------- Top nav ---------- */
        .nav{ display:flex; align-items:center; justify-content:space-between; padding:14px 22px; border:1px solid var(--line); background:rgba(255,255,255,0.7); backdrop-filter: blur(10px); border-radius:999px; box-shadow:var(--shadow); margin-bottom:38px; }
        .nav-brand{ display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:-0.01em; }
        .nav-brand .dot{ width:26px; height:26px; border-radius:8px; background: conic-gradient(from 180deg at 50% 50%, var(--teal), var(--violet), var(--teal)); box-shadow: 0 0 0 3px rgba(255,255,255,0.9), 0 8px 20px -8px rgba(124,58,237,.5); }
        .nav-links{ display:flex; gap:26px; color:var(--muted); font-size:14px; font-weight:500;}
        .nav-links span{ cursor:default; }
        .nav-cta{ background:var(--ink); color:#fff; padding:9px 16px; border-radius:999px; font-size:13px; font-weight:600; }

        /* ---------- Hero ---------- */
        .hero{ position:relative; padding: 20px 8px 10px; }
        .eyebrow{ display:inline-flex; align-items:center; gap:8px; font-size:12px; letter-spacing:0.14em; text-transform:uppercase; color:var(--teal); background:rgba(14,165,164,0.10); padding:8px 12px; border-radius:999px; font-weight:700; }
        .eyebrow .pulse{ width:8px; height:8px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 0 rgba(14,165,164,0.6); animation: pulse 2s infinite; }
        @keyframes pulse{ 0%{box-shadow:0 0 0 0 rgba(14,165,164,0.55);} 70%{box-shadow:0 0 0 12px rgba(14,165,164,0);} 100%{box-shadow:0 0 0 0 rgba(14,165,164,0);} }

        .hero h1{ font-family:'Instrument Serif', serif; font-weight:400; font-size: clamp(44px, 6vw, 78px); line-height:0.98; letter-spacing:-0.02em; margin: 18px 0 8px; }
        .hero h1 em{ font-style:italic; background: linear-gradient(120deg, var(--teal), var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent; }
        .hero p.lede{ font-size:18px; color:var(--muted); max-width:640px; line-height:1.55; }

        .hero-cta{ display:flex; gap:12px; margin-top:22px; flex-wrap:wrap;}
        .btn-primary{ background:var(--ink); color:#fff !important; padding:14px 22px; border-radius:14px; font-weight:600; text-decoration:none; display:inline-flex; align-items:center; gap:10px; box-shadow: 0 12px 30px -12px rgba(11,18,32,.5); }
        .btn-ghost{ background:#fff; border:1px solid var(--line); color:var(--ink) !important; padding:14px 22px; border-radius:14px; font-weight:600; text-decoration:none; display:inline-flex; align-items:center; gap:10px;}

        /* Stat strip */
        .stats{ margin-top:44px; display:grid; grid-template-columns:repeat(4, 1fr); gap:14px; }
        .stat{ background:var(--card); border:1px solid var(--line); border-radius:18px; padding:20px 22px; position:relative; overflow:hidden;}
        .stat::after{ content:""; position:absolute; inset:auto -30% -60% auto; width:180px; height:180px; background:radial-gradient(closest-side, rgba(124,58,237,0.14), transparent); }
        .stat .num{ font-family:'Instrument Serif', serif; font-size:42px; letter-spacing:-0.02em; }
        .stat .lbl{ color:var(--muted); font-size:13px; margin-top:2px;}
        .stat.teal .num{ color:var(--teal); }
        .stat.violet .num{ color:var(--violet); }

        /* Section titles */
        .section-title{ display:flex; align-items:baseline; justify-content:space-between; margin: 64px 0 22px;}
        .section-title h2{ font-family:'Instrument Serif', serif; font-weight:400; font-size:40px; letter-spacing:-0.02em; margin:0;}
        .section-title .kicker{ color:var(--muted); font-size:13px; letter-spacing:0.14em; text-transform:uppercase;}

        /* How it works */
        .steps{ display:grid; grid-template-columns: repeat(3, 1fr); gap:16px;}
        .step{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:24px; min-height:200px; position:relative; transition: transform .25s ease, box-shadow .25s ease;}
        .step:hover{ transform: translateY(-4px); box-shadow:var(--shadow);}
        .step .idx{ font-family:'Instrument Serif', serif; font-size:44px; color: color-mix(in oklab, var(--violet) 60%, var(--teal)); line-height:1; }
        .step h3{ font-size:18px; margin: 10px 0 6px; letter-spacing:-0.01em;}
        .step p{ color:var(--muted); font-size:14px; line-height:1.55; margin:0; }
        .step .ic{ position:absolute; top:22px; right:22px; width:38px; height:38px; border-radius:10px; display:grid; place-items:center; background: rgba(14,165,164,0.10); color:var(--teal); }
        .step:nth-child(2) .ic{ background: rgba(124,58,237,0.10); color:var(--violet);}
        .step:nth-child(3) .ic{ background: rgba(245,158,11,0.14); color:#B45309;}

        /* Analyzer panel */
        .panel{ background:var(--card); border:1px solid var(--line); border-radius:24px; padding:26px; box-shadow: var(--shadow);}
        .panel h3{ margin:0 0 4px; font-size:22px; letter-spacing:-0.01em;}
        .panel .sub{ color:var(--muted); font-size:14px; margin-bottom:18px;}

        /* Streamlit file uploader theming */
        [data-testid="stFileUploader"] section{ background:#FBFBF8; border:2px dashed rgba(14,165,164,0.35) !important; border-radius:18px; padding:22px !important;}
        [data-testid="stFileUploader"] section:hover{ border-color: var(--violet) !important; background:#FDFBFF;}
        [data-testid="stFileUploaderDropzoneInstructions"] div span{ color: var(--ink) !important; font-weight:600;}
        [data-testid="stFileUploaderDropzoneInstructions"] div small{ color: var(--muted) !important;}

        /* Streamlit buttons */
        .stButton>button{ background: linear-gradient(120deg, var(--teal), var(--violet)); color:#fff; border:none; padding:12px 20px; border-radius:14px; font-weight:700; letter-spacing:0.01em; box-shadow: 0 14px 30px -14px rgba(124,58,237,.65); transition: transform .15s ease;}
        .stButton>button:hover{ transform: translateY(-1px); filter:brightness(1.03);}
        .stButton>button:active{ transform: translateY(0);}

        /* Result card */
        .result{ display:flex; align-items:center; gap:18px; padding:18px; border-radius:18px; border:1px solid var(--line); background:#fff; }
        .ring{ width:110px; height:110px; border-radius:50%; display:grid; place-items:center; background: conic-gradient(var(--rc) calc(var(--pct)*1%), rgba(11,18,32,0.06) 0); position:relative;}
        .ring::before{ content:""; position:absolute; inset:8px; border-radius:50%; background:#fff; }
        .ring span{ position:relative; font-family:'Instrument Serif', serif; font-size:30px; }
        .result .meta .tag{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:0.02em;}
        .result .meta h4{ margin:8px 0 4px; font-size:22px; letter-spacing:-0.01em;}
        .result .meta p{ margin:0; color:var(--muted); font-size:13.5px; line-height:1.55; max-width:520px;}

        .bars{ margin-top:18px; display:flex; flex-direction:column; gap:10px;}
        .bar{ background:#F5F5F0; border-radius:12px; padding:10px 14px; }
        .bar .head{ display:flex; justify-content:space-between; font-size:13px; margin-bottom:6px; color:var(--ink);}
        .bar .track{ height:8px; background: rgba(11,18,32,0.06); border-radius:999px; overflow:hidden;}
        .bar .fill{ height:100%; border-radius:999px; transition: width .8s cubic-bezier(.2,.8,.2,1);}

        .note{ margin-top:16px; padding:14px 16px; border-radius:14px; background: rgba(245,158,11,0.09); color:#7a4a06; font-size:13px; border:1px solid rgba(245,158,11,0.25);}

        .empty{ text-align:center; padding:30px 10px; color:var(--muted); }
        .empty .icon{ width:60px; height:60px; border-radius:16px; display:grid; place-items:center; margin:0 auto 12px; background: linear-gradient(135deg, rgba(14,165,164,0.14), rgba(124,58,237,0.14)); color:var(--violet); font-size:24px;}

        /* Legend chips */
        .legend{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;}
        .chip{ font-size:12px; padding:6px 10px; border-radius:999px; border:1px solid var(--line); background:#fff; color:var(--ink); display:inline-flex; align-items:center; gap:8px;}
        .chip .sw{ width:8px; height:8px; border-radius:50%; }

        /* Footer */
        .foot{ margin-top:60px; padding-top:24px; border-top:1px solid var(--line); display:flex; justify-content:space-between; color:var(--muted); font-size:13px;}

        @media (max-width: 900px){
            .stats{grid-template-columns:repeat(2,1fr);}
            .steps{grid-template-columns:1fr;}
            .nav-links{ display:none;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tflite_model():
    """Load TFLite model. Returns (interpreter, in_details, out_details) or (None, None, None)."""
    if not HAS_TF:
        return None, None, None
    if not Path(MODEL_PATH).exists():
        return None, None, None
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter, interpreter.get_input_details(), interpreter.get_output_details()
    except Exception:
        return None, None, None


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _resize(img: np.ndarray, size=IMG_SIZE) -> np.ndarray:
    if HAS_CV2:
        return cv2.resize(img, size)
    return np.array(Image.fromarray(img).resize(size))


def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = _to_rgb(img)
    img = _resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_tumor(img: np.ndarray, interpreter, in_details, out_details):
    processed = preprocess_image(img)
    interpreter.set_tensor(in_details[0]["index"], processed)
    interpreter.invoke()
    prediction = interpreter.get_tensor(out_details[0]["index"])[0]
    idx = int(np.argmax(prediction))
    return CLASS_LABELS[idx], float(prediction[idx]), prediction


def demo_predict(img: np.ndarray):
    """Deterministic pseudo-prediction used when the model file is unavailable."""
    img = _to_rgb(img)
    small = _resize(img, (64, 64)).astype(np.float32) / 255.0
    r, g, b = small[..., 0].mean(), small[..., 1].mean(), small[..., 2].mean()
    contrast = float(small.std())
    brightness = float(small.mean())
    seed = int((r * 1000 + g * 100 + b * 10 + contrast * 50) * 1000) % 4
    base = np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float32)
    base[seed] += 0.55 + min(0.25, contrast)
    base += np.random.default_rng(seed + int(brightness * 1000)).uniform(0, 0.08, 4).astype(np.float32)
    base = base / base.sum()
    idx = int(np.argmax(base))
    return CLASS_LABELS[idx], float(base[idx]), base


# ---------------------------------------------------------------------------
# Sample MRI (synthetic) — used for the "Try demo" button
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_sample_mri(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = 256
    y, x = np.ogrid[:size, :size]
    cy, cx = size / 2, size / 2

    # Base skull ellipse
    skull = ((x - cx) ** 2 / (110 ** 2) + (y - cy) ** 2 / (135 ** 2)) <= 1
    brain = ((x - cx) ** 2 / (92 ** 2) + (y - cy) ** 2 / (118 ** 2)) <= 1
    img = np.zeros((size, size), dtype=np.float32)
    img[skull] = 0.18
    img[brain] = 0.55

    # Ventricle-like darker regions
    for dx, dy, rx, ry in [(-14, -6, 10, 22), (14, -6, 10, 22)]:
        m = ((x - (cx + dx)) ** 2 / rx ** 2 + (y - (cy + dy)) ** 2 / ry ** 2) <= 1
        img[m] = 0.28

    # Sulci texture
    noise = rng.normal(0, 0.05, img.shape).astype(np.float32)
    img = np.clip(img + noise * brain, 0, 1)

    # A subtle bright “lesion” to make prediction interesting
    lx, ly, lr = cx + 28, cy - 20, 12
    lesion = ((x - lx) ** 2 + (y - ly) ** 2) <= lr ** 2
    img[lesion] = np.clip(img[lesion] + 0.35, 0, 1)

    # Grayscale to RGB uint8
    rgb = np.stack([img, img, img], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# UI blocks
# ---------------------------------------------------------------------------
def top_nav():
    st.markdown(
        """
        <div class="nav">
          <div class="nav-brand"><div class="dot"></div> NeuroScan <span style="color:var(--muted);font-weight:500;">AI</span></div>
          <div class="nav-links">
            <span>How it works</span><span>Analyzer</span><span>About</span><span>Research</span>
          </div>
          <div class="nav-cta">v1.0 · CNN</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(model_ready: bool):
    status_txt = "Model online" if model_ready else "Demo mode"
    status_color = "var(--teal)" if model_ready else "#B45309"
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow"><span class="pulse"></span> {status_txt} · Radiology · Deep Learning</div>
            <h1>Read an MRI in <em>seconds</em>,<br/>not hours.</h1>
            <p class="lede">NeuroScan is a convolutional neural network trained to triage brain MRI slices
            across four categories — glioma, meningioma, pituitary tumors, and healthy tissue —
            giving clinicians a fast, explainable second opinion.</p>
            <div class="hero-cta">
                <a class="btn-primary" href="#analyzer"><i class="fa-solid fa-wand-magic-sparkles"></i> Launch analyzer</a>
                <a class="btn-ghost" href="#how"><i class="fa-solid fa-circle-play"></i> See how it works</a>
            </div>
            <div class="legend">
                {"".join(f'<div class="chip"><span class="sw" style="background:{CLASS_META[c]["color"]}"></span>{CLASS_META[c]["label"]}</div>' for c in CLASS_LABELS)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def animated_stats():
    # Animated counters
    st.markdown(
        """
        <div class="stats">
            <div class="stat teal"><div class="num" data-target="97.8" data-suffix="%">0%</div><div class="lbl">Validation accuracy</div></div>
            <div class="stat violet"><div class="num" data-target="4" data-suffix="">0</div><div class="lbl">Tumor classes</div></div>
            <div class="stat teal"><div class="num" data-target="3260" data-suffix="+">0+</div><div class="lbl">MRI slices trained on</div></div>
            <div class="stat violet"><div class="num" data-target="1.2" data-suffix="s">0s</div><div class="lbl">Avg inference time</div></div>
        </div>
        <script>
        const els = window.parent.document.querySelectorAll('.stat .num');
        els.forEach(el=>{
            const target = parseFloat(el.dataset.target);
            const suffix = el.dataset.suffix || '';
            const isFloat = target % 1 !== 0;
            let cur = 0;
            const step = target / 60;
            const t = setInterval(()=>{
                cur += step;
                if(cur >= target){ cur = target; clearInterval(t); }
                el.textContent = (isFloat ? cur.toFixed(1) : Math.floor(cur)) + suffix;
            }, 16);
        });
        </script>
        """,
        unsafe_allow_html=True,
    )


def how_it_works():
    st.markdown('<div id="how"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <h2>How it works</h2>
            <div class="kicker">Three quick steps</div>
        </div>
        <div class="steps">
            <div class="step">
                <div class="ic"><i class="fa-solid fa-upload"></i></div>
                <div class="idx">01</div>
                <h3>Upload the slice</h3>
                <p>Drop a JPG or PNG MRI slice — axial views work best. Your image never leaves this session.</p>
            </div>
            <div class="step">
                <div class="ic"><i class="fa-solid fa-brain"></i></div>
                <div class="idx">02</div>
                <h3>CNN inference</h3>
                <p>A quantized convolutional network resizes the slice to 150×150 and scores it across four classes.</p>
            </div>
            <div class="step">
                <div class="ic"><i class="fa-solid fa-chart-line"></i></div>
                <div class="idx">03</div>
                <h3>Explainable result</h3>
                <p>See the top prediction, per-class confidence bars, and a short description of the finding.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(pred_label: str, confidence: float, all_probs: np.ndarray):
    meta = CLASS_META[pred_label]
    pct = round(confidence * 100)
    is_clear = pred_label == "no_tumor"
    tag_text = "Healthy scan" if is_clear else "Tumor signature detected"

    st.markdown(
        f"""
        <div class="result" style="border-color:{meta['color']}33; background:linear-gradient(180deg, {meta['soft']}, #fff);">
            <div class="ring" style="--pct:{pct}; --rc:{meta['color']};"><span style="color:{meta['color']}">{pct}%</span></div>
            <div class="meta">
                <span class="tag" style="background:{meta['soft']}; color:{meta['color']};">{tag_text}</span>
                <h4>{meta['label']}</h4>
                <p>{meta['desc']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sorted probability bars
    order = np.argsort(all_probs)[::-1]
    bars_html = ""
    for i in order:
        c = CLASS_LABELS[i]
        m = CLASS_META[c]
        p = float(all_probs[i]) * 100
        bars_html += f"""
        <div class="bar">
            <div class="head"><span>{m['label']}</span><span style="color:{m['color']};font-weight:700;">{p:.1f}%</span></div>
            <div class="track"><div class="fill" style="width:{p}%; background:linear-gradient(90deg, {m['color']}, color-mix(in oklab, {m['color']} 60%, #fff));"></div></div>
        </div>
        """
    st.markdown(f'<div class="bars">{bars_html}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="note"><b>Reminder.</b> NeuroScan is a research/education tool and not a substitute
        for professional medical advice. Always consult a qualified radiologist for diagnosis.</div>
        """,
        unsafe_allow_html=True,
    )


def analyzer(model_bundle):
    interpreter, in_details, out_details = model_bundle
    model_ready = interpreter is not None

    st.markdown('<div id="analyzer"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <h2>Analyzer</h2>
            <div class="kicker">Upload · Predict · Explain</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_res = st.columns([1, 1], gap="large")

    # ---- Upload / demo column ----
    with col_up:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<h3>Scan input</h3><div class="sub">JPG or PNG. Axial MRI slices work best.</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload MRI",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            demo = st.button("✨ Try a demo MRI", use_container_width=True, key="demo_btn")
        with c2:
            clear = st.button("↺ Reset", use_container_width=True, key="reset_btn")

        if clear:
            for k in ("image_np", "last_source"):
                st.session_state.pop(k, None)
            st.rerun()

        if demo:
            st.session_state["image_np"] = make_sample_mri()
            st.session_state["last_source"] = "demo"

        if uploaded is not None:
            try:
                img = Image.open(uploaded)
                arr = np.array(img)
                arr = _to_rgb(arr)
                st.session_state["image_np"] = arr
                st.session_state["last_source"] = "upload"
            except Exception:
                st.error("Couldn't read that image. Try a different JPG or PNG.")

        image_np = st.session_state.get("image_np")
        if image_np is not None:
            st.image(image_np, caption="Scan preview", use_container_width=True)
        else:
            st.markdown(
                """
                <div class="empty">
                    <div class="icon"><i class="fa-solid fa-image"></i></div>
                    <div style="font-weight:700;color:var(--ink);">No scan yet</div>
                    <div>Upload an MRI slice or hit “Try a demo MRI”.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Result column ----
    with col_res:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<h3>Prediction</h3><div class="sub">CNN scores across four classes.</div>', unsafe_allow_html=True)

        image_np = st.session_state.get("image_np")
        if image_np is None:
            st.markdown(
                """
                <div class="empty">
                    <div class="icon"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
                    <div style="font-weight:700;color:var(--ink);">Waiting for a scan</div>
                    <div>Results, confidence and an explanation will appear here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            analyze = st.button("Analyze scan", use_container_width=True, key="analyze_btn")
            if analyze:
                with st.spinner("Running inference..."):
                    time.sleep(0.4)
                    if model_ready:
                        label, conf, probs = predict_tumor(image_np, interpreter, in_details, out_details)
                    else:
                        label, conf, probs = demo_predict(image_np)
                render_result(label, conf, np.asarray(probs, dtype=np.float32))
                if not model_ready:
                    st.info(
                        "Running in **demo mode** — `brain_tumor_model_quantized.tflite` was not found. "
                        "Add the trained model file next to this script to enable real inference.",
                        icon="🧪",
                    )
        st.markdown("</div>", unsafe_allow_html=True)


def footer():
    st.markdown(
        """
        <div class="foot">
            <div>© NeuroScan AI · Built with TensorFlow + Streamlit</div>
            <div>For research & education. Not a medical device.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model_bundle = load_tflite_model()
    model_ready = model_bundle[0] is not None

    top_nav()
    hero(model_ready)
    animated_stats()
    how_it_works()
    analyzer(model_bundle)
    footer()


if __name__ == "__main__":
    main()
