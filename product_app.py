
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best_resnet_finetuned.pth"

CLASSES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Handbags",
    "Sandals",
]


# --------------------------------------------------
# 2. IMAGE PREPROCESSING
# --------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 3. LOAD THE EXISTING TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASSES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, device


# --------------------------------------------------
# 4. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fashion Product Classifier",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# --------------------------------------------------
# 5. CUSTOM UI DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main page width and spacing */
    .block-container {
        max-width: 1100px !important;
        padding: 1.5rem 2rem !important;
    }

    /* Compact, readable typography */
    html, body, [class*="st-"] {
        font-size: 16px !important;
    }

    p, label, li {
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }

    h1 {
        font-size: 2.4rem !important;
        line-height: 1.15 !important;
    }

    h2 {
        font-size: 1.8rem !important;
    }

    h3 {
        font-size: 1.35rem !important;
    }

    /* Main hero panel */
    .hero {
        padding: 2rem !important;
        border-radius: 20px;
        border: 1px solid #454b55;
        background: linear-gradient(
            120deg,
            #20251c,
            #151820 75%
        );
        margin-bottom: 1.5rem !important;
    }

    .hero-tag {
        color: #c8f169;
        font-size: 0.85rem !important;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        color: #f5f5f5;
        font-size: 2.7rem !important;
        font-weight: 850;
        line-height: 1.15;
        letter-spacing: -1px;
        margin: 0.7rem 0;
    }

    .hero-description {
        color: #c5c8d0;
        font-size: 1.05rem !important;
        line-height: 1.6;
        max-width: 850px;
    }

    /* Section labels */
    .section-label {
        color: #c8f169;
        font-size: 0.8rem !important;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Upload box */
    [data-testid="stFileUploader"] {
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
        border: 2px dashed #727b59;
        border-radius: 14px;
        padding: 1rem;
        background: rgba(200, 241, 105, 0.04);
    }

    [data-testid="stFileUploader"] section {
        padding: 0.75rem;
    }

    [data-testid="stImage"] {
        margin-bottom: 1rem;
    }

    /* Results */
    .result-category {
        font-size: 2.2rem !important;
        font-weight: 850;
        line-height: 1.2;
        overflow-wrap: anywhere;
        margin: 0.75rem 0;
    }

    .result-label {
        color: #a5a8b5;
        font-size: 0.85rem !important;
        font-weight: 750;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    .confidence-number {
        color: #c8f169;
        font-size: 2.7rem !important;
        font-weight: 850;
        line-height: 1.2;
    }

    .small-note {
        color: #b5b8c2;
        font-size: 0.9rem !important;
        line-height: 1.55;
    }

    /* Metrics and progress bars */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
    }

    [data-testid="stProgressBar"] {
        margin-top: 0.3rem;
        margin-bottom: 0.5rem;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: #c8f169 !important;
    }

    /* About section and footer */
    div[data-testid="stExpander"] {
        margin-top: 1rem;
        border-radius: 12px;
        font-size: 1rem !important;
    }

    .footer {
        text-align: center;
        color: #a5a8b5;
        font-size: 0.85rem !important;
        line-height: 1.6;
        padding: 1.25rem 0;
    }

    /* Mobile layout */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.75rem !important;
        }

        .hero {
            padding: 1.25rem !important;
            border-radius: 16px;
        }

        .hero-title {
            font-size: 2rem !important;
        }

        .hero-description {
            font-size: 0.95rem !important;
        }

        .result-category {
            font-size: 1.8rem !important;
        }

        .confidence-number {
            font-size: 2.2rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 6. HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">AI / COMPUTER VISION</div>
        <div class="hero-title">
            Fashion Product Classifier
        </div>
        <div class="hero-description">
            Upload a fashion product image and let our fine-tuned
            ResNet18 model identify its category. Explore the
            prediction and compare scores across six product classes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)




# --------------------------------------------------
# 7. IMAGE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-label">Upload</div>',
    unsafe_allow_html=True,
)

st.header("Bring a product into focus")

st.markdown(
    "Choose a clear photo of a shirt, shoe, handbag, or sandal."
)

uploaded_file = st.file_uploader(
    "Drop your image here or browse your files",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, and WebP.",
)


# --------------------------------------------------
# 8. IMAGE PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:
        st.error(
            "We couldn't read this image. Please upload a valid "
            "JPG, JPEG, PNG, or WebP file."
        )
        st.stop()

    # Prepare image and make prediction
    model, device = load_model()

    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = probabilities.argmax().item()
    predicted_class = CLASSES[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    
    # ----------------------------------------------
    # 9. IMAGE + PREDICTION WORKSPACE
    # ----------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-label">ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.header("Your results")

    image_col, result_col = st.columns(
        [1, 1],
        gap="large",
    )

    with image_col:
        st.markdown("### Product image")

        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch",
        )

    
    with result_col:
        with st.container(border=True):
            st.markdown("### AI prediction")

            st.markdown(
                '<div class="result-label">PREDICTED CATEGORY</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="result-category">{predicted_class}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="result-label">MODEL CONFIDENCE</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="confidence-number">{confidence:.2f}%</div>',
                unsafe_allow_html=True,
            )

            st.progress(confidence / 100)

            st.markdown(
                '<div class="small-note">'
                'This score is not a guarantee that the prediction '
                'is correct.'
                '</div>',
                unsafe_allow_html=True,
            )


    # ----------------------------------------------
    # 10. CLASS SCORE BREAKDOWN
    # ----------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-label">BREAKDOWN</div>',
        unsafe_allow_html=True,
    )

    st.header("Category scores")

    st.markdown(
        "See how the model scores each of the six categories."
    )

    ranked_scores = sorted(
        enumerate(probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    for class_index, score in ranked_scores:
        score_col1, score_col2 = st.columns(
            [1, 4],
            gap="medium",
        )

        with score_col1:
            st.markdown(f"**{CLASSES[class_index]}**")

        with score_col2:
            st.progress(score)
            st.caption(f"{score * 100:.2f}%")

    st.caption(
        "Scores are relative model outputs. A high score does "
        "not guarantee a correct prediction."
    )


# --------------------------------------------------
# 11. ABOUT THE MODEL
# --------------------------------------------------

st.divider()

with st.expander("ABOUT THIS PROJECT  ·  MODEL & EVALUATION"):

    about_col1, about_col2 = st.columns(2, gap="large")

    with about_col1:
        st.subheader("Model")

        st.write(
            "Fine-tuned ResNet18 convolutional neural network, "
            "implemented with PyTorch."
        )

        st.write(
            "Input images are resized to 224 × 224 pixels "
            "and normalized using ImageNet statistics."
        )

    with about_col2:
        st.subheader("Evaluation")

        st.metric(
            "Held-out test accuracy",
            "95.63%",
        )

        st.write(
            "Evaluated on the project's held-out test set. "
            "Performance may vary on images from different "
            "sources, styles, lighting conditions, and backgrounds."
        )

    st.caption(
        "Categories: Tshirts · Shirts · Casual Shoes · Sports Shoes "
        "· Handbags · Sandals"
    )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Fashion Product Classification<br>
        Built with PyTorch and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
