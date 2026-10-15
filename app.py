
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
from model import CNN


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Fashion AI Classifier",
    page_icon="👕",
    layout="centered"
)

st.title("👕 Fashion AI Classifier")
st.write(
    "Upload a clothing image and let a CNN predict its category."
)

st.info(
    "This model was trained on Fashion-MNIST. "
    "Predictions on ordinary product photos may be unreliable."
)


# -------------------------
# Class names
# -------------------------
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image(image):
    image = image.convert("L")

    # Preserve proportions while fitting inside 28 x 28
    image.thumbnail((28, 28))

    # Center the image on a black background
    canvas = Image.new("L", (28, 28), 255)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))

    # Convert pixels to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensor = transform(canvas)

    # Add batch dimension
    return image_tensor.unsqueeze(0), canvas


# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    model = CNN()

    model_path = Path(__file__).resolve().parent / "best_model.pth"

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()
    return model


try:
    model = load_model()
    
    @st.cache_data
    def load_test_dataset():
        return datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
    )

    test_dataset = load_test_dataset()

except (FileNotFoundError, RuntimeError) as error:
    st.error(
        "Could not load best_model.pth. "
        "Make sure the checkpoint matches your CNN architecture."
    )
    st.exception(error)
    st.stop()


# -------------------------
# Upload image
# -------------------------
st.divider()


st.subheader("🎓 Try a Fashion-MNIST example")
st.write("Test the CNN on an image from its own dataset.")

if st.button("Show a random test image"):
    import random

    random_index = random.randrange(len(test_dataset))
    demo_image, true_label = test_dataset[random_index]

    with torch.no_grad():
        demo_outputs = model(demo_image.unsqueeze(0))
        demo_probabilities = F.softmax(demo_outputs, dim=1)[0]
        demo_prediction = demo_probabilities.argmax().item()

    st.image(
        demo_image.squeeze().numpy(),
        caption=f"Actual: {class_names[true_label]}",
        clamp=True,
        width=180
    )

    st.write(f"**Predicted:** {class_names[demo_prediction]}")
    st.write(f"**Actual:** {class_names[true_label]}")
    st.write(
        f"**Prediction score:** "
        f"{demo_probabilities[demo_prediction].item() * 100:.2f}%"
    )


uploaded_file = st.file_uploader(
    "Choose a clothing image",
    type=["png", "jpg", "jpeg"]
)


# -------------------------
# Predict
# -------------------------
if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        image_tensor, processed_image = preprocess_image(original_image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original_image,
                caption="Uploaded image",
                width="stretch"
            )

        with col2:
            st.image(
                processed_image,
                caption="Image sent to the CNN",
                width="stretch"
            )

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        predicted_index = probabilities.argmax().item()
        confidence = probabilities[predicted_index].item()

        st.divider()
        st.subheader("Prediction")

        st.success(
            f"Predicted class: {class_names[predicted_index]}"
        )

        st.metric(
            "Top prediction score",
            f"{confidence * 100:.2f}%"
        )

        st.caption(
            "This score comes from softmax and is not a guarantee "
            "that the prediction is correct."
        )

        # Display scores for all 10 classes
        st.subheader("Scores for all classes")

        for index in torch.argsort(
            probabilities, descending=True
        ).tolist():
            score = probabilities[index].item()

            st.write(
                f"**{class_names[index]}** — {score * 100:.2f}%"
            )
            st.progress(score)

    except Exception as error:
        st.error(f"Could not process this image: {error}")
