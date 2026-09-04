import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Class labels
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
IMG_SIZE = (150, 150)

# ---------------------------------------------------------------------------
# Design system
#
# Palette is built around the idea of a scan readout rather than a generic
# SaaS dashboard: white canvas, ink text, a cobalt "signal" accent for the
# brand and normal-range results, and a clinical coral reserved only for
# a positive tumor finding. Space Grotesk carries headlines (a technical,
# slightly geometric voice); Inter carries everything functional.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #ffffff;
        --panel: #FAFAFB;
        --ink: #14161C;
        --ink-soft: #5B6472;
        --ink-faint: #9AA1AE;
        --border: #E4E6EB;
        --signal: #2436D6;
        --signal-soft: #EEF0FE;
        --teal: #0E8C82;
        --teal-soft: #E7F5F3;
        --alert: #D8352B;
        --alert-soft: #FCEBEA;
        --amber: #9A5B10;
        --amber-soft: #FBF3E7;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .display { font-family: 'Space Grotesk', sans-serif; }

    .stApp { background: var(--bg); }
    .main > div { padding-top: 1.5rem; }
    #MainMenu, footer, header { visibility: hidden; }

    .shell { max-width: 1080px; margin: 0 auto; padding: 0 1rem 3rem; }

    /* ---------- hero ---------- */
    .hero { padding: 2.5rem 0 2rem; border-bottom: 1px solid var(--border); margin-bottom: 2.5rem; }
    .brand-mark {
        display: inline-flex; align-items: center; gap: 8px;
        font-size: 13px; color: var(--signal); font-weight: 600;
        letter-spacing: 0.02em; margin-bottom: 1rem;
    }
    .brand-mark .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--signal); }
    .hero-title {
        font-size: 2.75rem; font-weight: 700; color: var(--ink);
        line-height: 1.1; margin: 0 0 0.9rem; max-width: 620px;
    }
    .hero-subtitle {
        font-size: 1.05rem; color: var(--ink-soft); line-height: 1.6;
        max-width: 560px; margin: 0 0 1.5rem; font-weight: 400;
    }
    .legend-row { display: flex; flex-wrap: wrap; gap: 8px; }
    .legend-chip {
        display: inline-flex; align-items: center; gap: 7px;
        border: 1px solid var(--border); border-radius: 999px;
        padding: 6px 14px; font-size: 13px; color: var(--ink-soft);
        background: var(--panel);
    }
    .legend-chip .swatch { width: 7px; height: 7px; border-radius: 50%; }

    /* ---------- scan frame (upload + image display) ---------- */
    .scan-frame {
        position: relative; background: var(--panel);
        border: 1px solid var(--border); padding: 2.75rem 2rem;
        text-align: center; margin-bottom: 0.5rem;
    }
    .scan-frame.has-image { padding: 0.75rem; background: #fff; text-align: left; }
    .corner { position: absolute; width: 16px; height: 16px; }
    .corner.tl { top: -1px; left: -1px; border-top: 2px solid var(--signal); border-left: 2px solid var(--signal); }
    .corner.tr { top: -1px; right: -1px; border-top: 2px solid var(--signal); border-right: 2px solid var(--signal); }
    .corner.bl { bottom: -1px; left: -1px; border-bottom: 2px solid var(--signal); border-left: 2px solid var(--signal); }
    .corner.br { bottom: -1px; right: -1px; border-bottom: 2px solid var(--signal); border-right: 2px solid var(--signal); }

    .scan-frame-label {
        font-size: 12px; color: var(--ink-faint); font-weight: 500;
        margin: 0 0 4px; padding: 6px 4px 0;
    }
    .empty-copy h3 { font-size: 1.05rem; color: var(--ink); font-weight: 600; margin: 0.75rem 0 0.25rem; }
    .empty-copy p { color: var(--ink-faint); font-size: 0.9rem; margin: 0; }
    .empty-icon {
        width: 44px; height: 44px; margin: 0 auto; border-radius: 50%;
        border: 1.5px solid var(--signal); position: relative;
    }
    .empty-icon::before, .empty-icon::after {
        content: ''; position: absolute; background: var(--signal);
    }
    .empty-icon::before { width: 16px; height: 2px; top: 21px; left: 14px; }
    .empty-icon::after { width: 2px; height: 16px; top: 14px; left: 21px; }

    /* uploader chrome */
    [data-testid="stFileUploader"] section {
        background: transparent; border: none; padding: 0;
    }
    [data-testid="stFileUploaderDropzone"] { background: transparent; }

    /* ---------- analysis panel ---------- */
    .panel-label {
        font-size: 12px; color: var(--ink-faint); font-weight: 600;
        text-transform: none; margin-bottom: 0.9rem;
    }
    .diagnosis-row { display: flex; align-items: flex-start; gap: 1.5rem; margin-bottom: 1.5rem; }
    .diagnosis-text { flex: 1; }
    .diagnosis-title { font-size: 1.6rem; font-weight: 700; color: var(--ink); margin: 0 0 0.35rem; }
    .diagnosis-strip {
        display: inline-block; font-size: 0.85rem; font-weight: 500;
        padding: 5px 12px; border-radius: 3px;
    }
    .diagnosis-strip.clear { background: var(--teal-soft); color: var(--teal); }
    .diagnosis-strip.flag { background: var(--alert-soft); color: var(--alert); }

    .ring-wrap { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
    .confidence-ring {
        width: 108px; height: 108px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
    }
    .ring-inner {
        width: 84px; height: 84px; border-radius: 50%; background: #fff;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .ring-value { font-size: 1.15rem; font-weight: 700; color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
    .ring-caption { font-size: 10.5px; color: var(--ink-faint); margin-top: 1px; }

    /* probability breakdown */
    .breakdown { border-top: 1px solid var(--border); padding-top: 1.25rem; margin-top: 0.5rem; }
    .breakdown-title { font-size: 13px; font-weight: 600; color: var(--ink); margin-bottom: 0.9rem; }
    .prob-row { display: grid; grid-template-columns: 130px 1fr 46px; align-items: center; gap: 10px; margin-bottom: 9px; }
    .prob-label { font-size: 13px; color: var(--ink-soft); }
    .prob-label.top { color: var(--ink); font-weight: 600; }
    .prob-track { height: 6px; background: #EDEEF2; border-radius: 3px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 3px; }
    .prob-pct { font-size: 12.5px; color: var(--ink-soft); text-align: right; font-variant-numeric: tabular-nums; }

    /* note / disclaimer */
    .note {
        margin-top: 1.5rem; padding: 0.9rem 1rem; background: var(--amber-soft);
        border-left: 2px solid var(--amber); font-size: 12.5px; color: var(--amber); line-height: 1.55;
    }
    .note strong { color: var(--amber); }

    /* ---------- widgets ---------- */
    .stButton > button {
        background: var(--ink); color: #fff; border: none;
        padding: 0.7rem 1.5rem; border-radius: 6px; font-size: 0.95rem;
        font-weight: 600; width: 100%; transition: background 0.15s ease;
        box-shadow: none;
    }
    .stButton > button:hover { background: var(--signal); }
    .stButton > button:active { background: var(--signal); }

    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .diagnosis-row { flex-direction: column-reverse; align-items: center; text-align: center; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Function to load the TFLite model
@st.cache_resource
def load_tflite_model():
    try:
        # Get the model path relative to the script
        model_path = "brain_tumor_model_quantized.tflite"

        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


# Function to preprocess the image
def preprocess_image(img):
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Resize to the expected input size
    img = cv2.resize(img, IMG_SIZE)

    # Normalize pixel values
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# Function to make prediction using TFLite model
def predict_tumor(img, interpreter, input_details, output_details):
    # Preprocess the image
    processed_img = preprocess_image(img)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_img)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class and confidence
    predicted_class_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_index]

    return CLASS_LABELS[predicted_class_index], confidence, prediction[0]


# Function to format tumor type for display
def format_tumor_type(tumor_type):
    type_mapping = {
        'glioma_tumor': 'Glioma tumor',
        'meningioma_tumor': 'Meningioma tumor',
        'no_tumor': 'No tumor',
        'pituitary_tumor': 'Pituitary tumor'
    }
    return type_mapping.get(tumor_type, tumor_type)


# Reference legend of what the model detects, and a shared color per class
# for the chips, the confidence ring, and the probability breakdown.
CLASS_COLORS = {
    'glioma_tumor': '#D8352B',
    'meningioma_tumor': '#9A5B10',
    'no_tumor': '#0E8C82',
    'pituitary_tumor': '#2436D6',
}


def render_hero():
    chips = "".join(
        f'<span class="legend-chip"><span class="swatch" style="background:{CLASS_COLORS[c]}"></span>'
        f'{format_tumor_type(c)}</span>'
        for c in CLASS_LABELS
    )
    st.markdown(
        f"""
        <div class="hero">
            <div class="brand-mark"><span class="dot"></span>NeuroScan AI</div>
            <h1 class="hero-title">Read an MRI scan in seconds.</h1>
            <p class="hero-subtitle">
                Upload a brain MRI slice and the model classifies it across four
                categories, trained to spot the signal a radiologist looks for first.
            </p>
            <div class="legend-row">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_breakdown(all_predictions, predicted_class):
    order = np.argsort(all_predictions)[::-1]
    rows = ""
    for idx in order:
        label = CLASS_LABELS[idx]
        pct = float(all_predictions[idx]) * 100
        color = CLASS_COLORS[label]
        is_top = label == predicted_class
        rows += f"""
            <div class="prob-row">
                <span class="prob-label {'top' if is_top else ''}">{format_tumor_type(label)}</span>
                <div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
                <span class="prob-pct">{pct:.1f}%</span>
            </div>
        """
    st.markdown(
        f"""
        <div class="breakdown">
            <div class="breakdown-title">Full breakdown</div>
            {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Main function
def main():
    # Load model
    interpreter, input_details, output_details = load_tflite_model()

    if interpreter is None:
        st.error("Failed to load the model. Please check if the model file exists in the repository.")
        st.stop()

    st.markdown('<div class="shell">', unsafe_allow_html=True)

    render_hero()

    # Upload section
    st.markdown(
        """
        <div class="scan-frame">
            <span class="corner tl"></span><span class="corner tr"></span>
            <span class="corner bl"></span><span class="corner br"></span>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], help="Upload a clear MRI brain scan image")

    if uploaded_file is None:
        st.markdown(
            """
            <div class="empty-copy">
                <div class="empty-icon"></div>
                <h3>Drop an MRI scan here</h3>
                <p>or click to browse — JPG or PNG</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        # Convert to numpy array for cv2 processing
        image_np = np.array(image)

        # Ensure the image is in RGB format
        if image_np.ndim == 3:
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif image_np.shape[2] == 1:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        else:
            st.error("Invalid image format. Please upload a valid image.")
            return

        # Create two columns for layout
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(
                """
                <div class="scan-frame has-image">
                    <span class="corner tl"></span><span class="corner tr"></span>
                    <span class="corner bl"></span><span class="corner br"></span>
                    <div class="scan-frame-label">Uploaded scan</div>
                """,
                unsafe_allow_html=True,
            )
            st.image(image_np, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="panel-label">Analysis</div>', unsafe_allow_html=True)

            analyze_clicked = st.button('Analyze scan', use_container_width=True)

            if analyze_clicked:
                with st.spinner("Processing image..."):
                    predicted_class, confidence, all_predictions = predict_tumor(
                        image_np, interpreter, input_details, output_details
                    )

                formatted_class = format_tumor_type(predicted_class)
                is_clear = predicted_class == 'no_tumor'
                ring_color = CLASS_COLORS[predicted_class]
                pct = confidence * 100

                st.markdown(
                    f"""
                    <div class="diagnosis-row">
                        <div class="diagnosis-text">
                            <div class="diagnosis-title">{formatted_class}</div>
                            <span class="diagnosis-strip {'clear' if is_clear else 'flag'}">
                                {'No tumor detected' if is_clear else 'Tumor detected'}
                            </span>
                        </div>
                        <div class="ring-wrap">
                            <div class="confidence-ring" style="background:conic-gradient({ring_color} {pct:.1f}%, #EDEEF2 0);">
                                <div class="ring-inner">
                                    <span class="ring-value">{pct:.0f}%</span>
                                    <span class="ring-caption">confidence</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                render_probability_breakdown(all_predictions, predicted_class)

                st.markdown(
                    """
                    <div class="note">
                        <strong>Note.</strong> This is an AI-assisted tool for educational purposes.
                        Always consult a qualified medical professional for diagnosis and treatment.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
