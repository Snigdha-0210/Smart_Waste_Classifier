import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from datetime import datetime

from model_resnet import WasteResNet


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide"
)


# ============================================================
# 2. CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]


# ============================================================
# 3. OOD CONFIGURATION
# ============================================================
#
# These are intentionally conservative.
#
# The model was trained ONLY on the six classes above.
# It should therefore not blindly classify every image
# on the internet as one of these classes.
#
# This is OOD GUARD V1.
# Later we can replace this with a trained OOD detector.
# ============================================================

MIN_CONFIDENCE = 0.70
MIN_MARGIN = 0.18
MAX_NORMALIZED_ENTROPY = 0.72
MIN_AUGMENTATION_AGREEMENT = 0.60


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = WasteResNet(num_classes=6)

    model.load_state_dict(
        torch.load(
            "waste_resnet18_best.pth",
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


model = load_model()


# ============================================================
# 6. BASE IMAGE TRANSFORMATION
# ============================================================

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. OOD AUGMENTATIONS
# ============================================================

augmentation_transforms = [

    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(
            224,
            padding=12
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
]


# ============================================================
# 8. BASIC PREDICTION FUNCTION
# ============================================================

def predict_tensor(image_tensor):

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

    return probabilities[0]


# ============================================================
# 9. CALCULATE ENTROPY
# ============================================================

def calculate_normalized_entropy(probabilities):

    probabilities = probabilities + 1e-10

    entropy = -torch.sum(
        probabilities * torch.log(probabilities)
    )

    max_entropy = np.log(len(CLASS_NAMES))

    normalized_entropy = (
        entropy.item() / max_entropy
    )

    return normalized_entropy


# ============================================================
# 10. OOD ANALYSIS
# ============================================================

def analyze_ood(image):

    predictions = []

    probability_vectors = []

    # --------------------------------------------------------
    # Run multiple views of the image
    # --------------------------------------------------------

    for transform in augmentation_transforms:

        tensor = transform(image)

        probabilities = predict_tensor(tensor)

        probability_vectors.append(
            probabilities.detach().cpu().numpy()
        )

        predictions.append(
            torch.argmax(probabilities).item()
        )


    # --------------------------------------------------------
    # Original prediction
    # --------------------------------------------------------

    original_probabilities = torch.tensor(
        probability_vectors[0]
    )

    sorted_probabilities, sorted_indices = torch.sort(
        original_probabilities,
        descending=True
    )

    top1_index = sorted_indices[0].item()
    top2_index = sorted_indices[1].item()

    top1_confidence = sorted_probabilities[0].item()
    top2_confidence = sorted_probabilities[1].item()

    margin = (
        top1_confidence -
        top2_confidence
    )


    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    entropy = calculate_normalized_entropy(
        original_probabilities
    )


    # --------------------------------------------------------
    # Augmentation agreement
    # --------------------------------------------------------

    agreement = (
        predictions.count(top1_index)
        / len(predictions)
    )


    # --------------------------------------------------------
    # Individual signals
    # --------------------------------------------------------

    confidence_ok = (
        top1_confidence >= MIN_CONFIDENCE
    )

    margin_ok = (
        margin >= MIN_MARGIN
    )

    entropy_ok = (
        entropy <= MAX_NORMALIZED_ENTROPY
    )

    agreement_ok = (
        agreement >= MIN_AUGMENTATION_AGREEMENT
    )


    # --------------------------------------------------------
    # OOD decision
    # --------------------------------------------------------
    #
    # We don't rely on ONE signal.
    #
    # If several signals are suspicious, reject.
    # --------------------------------------------------------

    failed_checks = 0

    if not confidence_ok:
        failed_checks += 1

    if not margin_ok:
        failed_checks += 1

    if not entropy_ok:
        failed_checks += 1

    if not agreement_ok:
        failed_checks += 1


    if failed_checks >= 2:

        status = "unsupported"

    elif failed_checks == 1:

        status = "uncertain"

    else:

        status = "supported"


    return {
        "predicted_index": top1_index,
        "predicted_class": CLASS_NAMES[top1_index],
        "confidence": top1_confidence,
        "second_class": CLASS_NAMES[top2_index],
        "second_confidence": top2_confidence,
        "margin": margin,
        "entropy": entropy,
        "agreement": agreement,
        "status": status,
        "confidence_ok": confidence_ok,
        "margin_ok": margin_ok,
        "entropy_ok": entropy_ok,
        "agreement_ok": agreement_ok,
        "probabilities": original_probabilities.numpy()
    }


# ============================================================
# 11. SESSION HISTORY
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# 12. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("♻️ Smart Waste")

page = st.sidebar.radio(
    "Navigation",
    [
        "Classifier",
        "Dashboard",
        "History",
        "About Model"
    ]
)


# ============================================================
# 13. CLASSIFIER PAGE
# ============================================================

if page == "Classifier":

    st.title("♻️ Smart Waste Classifier")

    st.write(
        "Upload a waste image and the trained ResNet18 model "
        "will classify it into one of six supported waste categories."
    )

    st.info(
        "The OOD Guard checks whether the image looks sufficiently "
        "similar and stable for the six waste categories the model knows."
    )


    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    with st.expander("Model Information"):

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("**Model**")
            st.write("ResNet18")

        with col2:
            st.write("**Classes**")
            st.write("6")

        with col3:
            st.write("**Test Accuracy**")
            st.write("93.34%")

        with col4:
            st.write("**Device**")
            st.write(str(device))


    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # Display image
        # ----------------------------------------------------

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

        st.divider()


        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing image and checking OOD signals..."
        ):

            result = analyze_ood(image)


        predicted_class = result["predicted_class"]

        confidence = (
            result["confidence"] * 100
        )

        second_class = result["second_class"]

        second_confidence = (
            result["second_confidence"] * 100
        )

        margin = (
            result["margin"] * 100
        )

        entropy = (
            result["entropy"] * 100
        )

        agreement = (
            result["agreement"] * 100
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.subheader("Prediction")


        # ----------------------------------------------------
        # SUPPORTED
        # ----------------------------------------------------

        if result["status"] == "supported":

            st.success(
                f"♻️ {predicted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "The model's prediction was stable across "
                "multiple image views."
            )


        # ----------------------------------------------------
        # UNCERTAIN
        # ----------------------------------------------------

        elif result["status"] == "uncertain":

            st.warning(
                f"⚠️ Possibly {predicted_class}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.write(
                "The model produced a prediction, but one of "
                "the OOD checks was suspicious."
            )

            st.info(
                "Please verify the material manually."
            )


        # ----------------------------------------------------
        # UNSUPPORTED
        # ----------------------------------------------------

        else:

            st.error(
                "⚠️ Unsupported or mixed waste image"
            )

            st.write(
                f"The model's strongest guess was "
                f"**{predicted_class} ({confidence:.2f}%)**, "
                f"but the image did not pass the OOD checks."
            )

            st.info(
                "This image may contain mixed waste, an "
                "unsupported material such as e-waste, or a "
                "scene that differs significantly from the "
                "training data."
            )


        # ====================================================
        # OOD DETAILS
        # ====================================================

        with st.expander("🔍 OOD Analysis Details"):

            st.write(
                "**These checks are diagnostic signals, "
                "not probabilities that the image is OOD.**"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Top prediction:** "
                    f"{predicted_class}"
                )

                st.write(
                    f"**Top confidence:** "
                    f"{confidence:.2f}%"
                )

                st.write(
                    f"**Second prediction:** "
                    f"{second_class}"
                )

                st.write(
                    f"**Second confidence:** "
                    f"{second_confidence:.2f}%"
                )

            with col2:

                st.write(
                    f"**Top-2 margin:** "
                    f"{margin:.2f}%"
                )

                st.write(
                    f"**Normalized entropy:** "
                    f"{entropy:.2f}%"
                )

                st.write(
                    f"**Augmentation agreement:** "
                    f"{agreement:.0f}%"
                )


            st.divider()

            st.write("### OOD Checks")

            if result["confidence_ok"]:
                st.success(
                    "✅ Confidence check passed"
                )
            else:
                st.warning(
                    "⚠️ Confidence check failed"
                )


            if result["margin_ok"]:
                st.success(
                    "✅ Top-2 separation check passed"
                )
            else:
                st.warning(
                    "⚠️ Top-2 separation check failed"
                )


            if result["entropy_ok"]:
                st.success(
                    "✅ Entropy check passed"
                )
            else:
                st.warning(
                    "⚠️ Entropy check failed"
                )


            if result["agreement_ok"]:
                st.success(
                    "✅ Augmentation consistency check passed"
                )
            else:
                st.warning(
                    "⚠️ Augmentation consistency check failed"
                )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader("Class Probabilities")

        probability_values = result["probabilities"]


        for class_name, probability in zip(
            CLASS_NAMES,
            probability_values
        ):

            percentage = (
                probability * 100
            )

            st.write(
                f"**{class_name}** — "
                f"{percentage:.2f}%"
            )

            st.progress(
                float(probability)
            )


        # ====================================================
        # SAVE TO HISTORY
        # ====================================================

        history_entry = {
            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "filename": uploaded_file.name,

            "prediction": predicted_class,

            "confidence": confidence,

            "status": result["status"]
        }


        # Avoid repeatedly adding the exact same upload
        if (
            len(st.session_state.history) == 0
            or st.session_state.history[-1]["filename"]
            != uploaded_file.name
            or st.session_state.history[-1]["time"]
            != history_entry["time"]
        ):

            st.session_state.history.append(
                history_entry
            )


# ============================================================
# 14. DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.title("📊 Dashboard")

    history = st.session_state.history


    if len(history) == 0:

        st.info(
            "No predictions yet. Go to Classifier and "
            "upload some images."
        )

    else:

        total_predictions = len(history)

        supported_count = sum(
            x["status"] == "supported"
            for x in history
        )

        uncertain_count = sum(
            x["status"] == "uncertain"
            for x in history
        )

        unsupported_count = sum(
            x["status"] == "unsupported"
            for x in history
        )


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Predictions",
                total_predictions
            )

        with col2:
            st.metric(
                "Supported",
                supported_count
            )

        with col3:
            st.metric(
                "Uncertain",
                uncertain_count
            )

        with col4:
            st.metric(
                "Unsupported",
                unsupported_count
            )


        st.divider()


        # ----------------------------------------------------
        # Category counts
        # ----------------------------------------------------

        st.subheader(
            "Predictions by Category"
        )

        category_counts = {}

        for entry in history:

            category = entry["prediction"]

            category_counts[category] = (
                category_counts.get(category, 0) + 1
            )


        for category in CLASS_NAMES:

            count = category_counts.get(
                category,
                0
            )

            st.write(
                f"**{category}** — {count}"
            )


        st.divider()


        # ----------------------------------------------------
        # Average confidence
        # ----------------------------------------------------

        average_confidence = sum(
            x["confidence"]
            for x in history
        ) / len(history)


        st.metric(
            "Average Model Confidence",
            f"{average_confidence:.2f}%"
        )


# ============================================================
# 15. HISTORY
# ============================================================

elif page == "History":

    st.title("🕘 Prediction History")

    history = st.session_state.history


    if len(history) == 0:

        st.info(
            "No prediction history yet."
        )

    else:

        for index, entry in enumerate(
            reversed(history)
        ):

            status = entry["status"]

            if status == "supported":
                icon = "🟢"

            elif status == "uncertain":
                icon = "🟡"

            else:
                icon = "🔴"


            st.write(
                f"{icon} **{entry['prediction']}** — "
                f"{entry['confidence']:.2f}%"
            )

            st.caption(
                f"{entry['filename']} • "
                f"{entry['time']} • "
                f"{status.upper()}"
            )

            st.divider()


        if st.button(
            "Clear History"
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# 16. ABOUT MODEL
# ============================================================

elif page == "About Model":

    st.title("🧠 About the Model")

    st.write(
        "This application uses a ResNet18 image classification "
        "model trained to recognize six waste categories."
    )


    st.subheader(
        "Supported Categories"
    )

    for category in CLASS_NAMES:

        st.write(
            f"♻️ {category}"
        )


    st.subheader(
        "Model Performance"
    )

    st.write(
        "**Test accuracy:** 93.34%"
    )

    st.write(
        "**Test images:** 706"
    )

    st.write(
        "**Correct predictions:** 659"
    )

    st.write(
        "**Incorrect predictions:** 47"
    )


    st.subheader(
        "Why OOD Detection?"
    )

    st.write(
        "A classifier trained on six classes will normally "
        "choose one of those six classes even when the uploaded "
        "image does not belong to any of them."
    )

    st.write(
        "For example, electronic waste or a photograph of a "
        "large mixed garbage dump may not correspond to one "
        "of the six training categories."
    )

    st.write(
        "The OOD Guard attempts to detect these situations "
        "instead of blindly trusting the highest softmax score."
    )


    st.warning(
        "OOD Guard V1 is a heuristic system. A future version "
        "should be calibrated using real out-of-distribution "
        "images and a dedicated OOD detector."
    )