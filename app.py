import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from datetime import datetime
import json
import os
import pandas as pd


from model_resnet import WasteResNet


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

MODEL_PATH = "waste_resnet18_best.pth"
HISTORY_FILE = "prediction_history.json"


# ============================================================
# WASTE INFORMATION
# ============================================================

WASTE_INFO = {

    "Cardboard": {
        "bin": "Paper / Dry Waste",
        "icon": "📦",
        "description": (
            "Cardboard is generally recyclable when it is clean "
            "and dry."
        ),
        "recommendation": (
            "Flatten the cardboard and place it in the dry "
            "recyclable waste stream."
        ),
        "tips": [
            "Remove plastic packaging where possible.",
            "Keep cardboard dry.",
            "Flatten large boxes before disposal."
        ]
    },

    "Food Organics": {
        "bin": "Organic / Compost Waste",
        "icon": "🍎",
        "description": (
            "Food and other organic waste can often be "
            "processed through composting or organic-waste systems."
        ),
        "recommendation": (
            "Place suitable food waste in the organic/compost "
            "waste stream."
        ),
        "tips": [
            "Separate food waste from recyclable materials.",
            "Use a compost bin where available.",
            "Avoid mixing plastic with organic waste."
        ]
    },

    "Glass": {
        "bin": "Glass Recycling",
        "icon": "🍾",
        "description": (
            "Glass containers are commonly recyclable, although "
            "local collection rules can differ."
        ),
        "recommendation": (
            "Place recyclable glass in the designated glass "
            "recycling stream."
        ),
        "tips": [
            "Empty containers before disposal.",
            "Follow your local glass-recycling rules.",
            "Handle broken glass carefully."
        ]
    },

    "Metal": {
        "bin": "Metal / Dry Recyclable Waste",
        "icon": "🥫",
        "description": (
            "Many metal containers and objects can be recovered "
            "through recycling."
        ),
        "recommendation": (
            "Place recyclable metal in the appropriate dry "
            "recyclable or metal collection stream."
        ),
        "tips": [
            "Empty containers before recycling.",
            "Keep metal separate from organic waste.",
            "Follow local recycling requirements."
        ]
    },

    "Paper": {
        "bin": "Paper / Dry Waste",
        "icon": "📄",
        "description": (
            "Clean and dry paper is commonly recyclable."
        ),
        "recommendation": (
            "Place clean paper in the paper or dry-recyclable "
            "waste stream."
        ),
        "tips": [
            "Keep paper dry.",
            "Remove non-paper attachments when practical.",
            "Avoid contaminating recyclable paper with food."
        ]
    },

    "Plastic": {
        "bin": "Plastic / Dry Recyclable Waste",
        "icon": "🧴",
        "description": (
            "Many plastic products can be recycled, but accepted "
            "plastic types vary by local waste-management systems."
        ),
        "recommendation": (
            "Check the local recycling rules for the specific "
            "plastic item before disposal."
        ),
        "tips": [
            "Empty containers before disposal.",
            "Check the recycling symbol where available.",
            "Do not assume every type of plastic is recyclable."
        ]
    }
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = WasteResNet(
        num_classes=6
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def load_history():

    if not os.path.exists(
        HISTORY_FILE
    ):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return []


def save_history(history):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


def add_prediction_to_history(
    predicted_class,
    confidence
):

    history = load_history()

    prediction = {

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "class":
            predicted_class,

        "confidence":
            round(
                confidence,
                2
            )
    }

    history.insert(
        0,
        prediction
    )

    # Keep latest 100 predictions
    history = history[:100]

    save_history(
        history
    )


# ============================================================
# SESSION STATE
# ============================================================

if "current_prediction" not in st.session_state:

    st.session_state.current_prediction = None


if "current_image" not in st.session_state:

    st.session_state.current_image = None


if "current_probabilities" not in st.session_state:

    st.session_state.current_probabilities = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "♻️ Smart Waste"
)

st.sidebar.caption(
    "AI-powered waste classification"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",

    [
        "🏠 Classifier",
        "📊 Dashboard",
        "🕒 History",
        "ℹ️ About Model"
    ]
)


st.sidebar.divider()


st.sidebar.write(
    "**Model:** ResNet18"
)

st.sidebar.write(
    "**Test Accuracy:** 93.34%"
)

st.sidebar.write(
    "**Classes:** 6"
)

st.sidebar.write(
    "**Input:** 224 × 224 RGB"
)

st.sidebar.write(
    "**Device:** " + str(device)
)


# ============================================================
# CLASSIFIER PAGE
# ============================================================

if page == "🏠 Classifier":

    st.title(
        "♻️ Smart Waste Classifier"
    )

    st.write(
        "Upload an image of waste and our trained "
        "ResNet18 model will classify it into one of "
        "six waste categories."
    )

    st.divider()


    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a waste image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )


        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        st.image(
            image,
            caption="Uploaded Waste Image",
            width="stretch"
        )

        st.divider()


        # ----------------------------------------------------
        # CLASSIFY BUTTON
        # ----------------------------------------------------

        if st.button(
            "🔍 Classify Waste",
            type="primary",
            width="stretch"
        ):

            with st.spinner(
                "Analyzing image with ResNet18..."
            ):

                image_tensor = transform(
                    image
                )

                image_tensor = (
                    image_tensor
                    .unsqueeze(0)
                    .to(device)
                )


                with torch.no_grad():

                    outputs = model(
                        image_tensor
                    )

                    probabilities = F.softmax(
                        outputs,
                        dim=1
                    )

                    confidence, predicted = torch.max(
                        probabilities,
                        dim=1
                    )


                predicted_class = CLASS_NAMES[
                    predicted.item()
                ]

                confidence_value = (
                    confidence.item() * 100
                )


                probability_values = (
                    probabilities[0]
                    .cpu()
                    .numpy()
                )


            # Save current result
            st.session_state.current_prediction = {

                "class":
                    predicted_class,

                "confidence":
                    confidence_value
            }


            st.session_state.current_image = image


            st.session_state.current_probabilities = (
                probability_values
            )


            # Save to history
            add_prediction_to_history(
                predicted_class,
                confidence_value
            )


            st.success(
                "Prediction completed successfully!"
            )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if (
            st.session_state.current_prediction
            is not None
        ):

            result = (
                st.session_state.current_prediction
            )

            predicted_class = result[
                "class"
            ]

            confidence_value = result[
                "confidence"
            ]


            st.divider()

            st.header(
                "Prediction Result"
            )


            # ------------------------------------------------
            # MAIN RESULT
            # ------------------------------------------------

            col1, col2 = st.columns(
                2
            )


            with col1:

                st.success(
                    f"♻️ {predicted_class}"
                )


            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence_value:.2f}%"
                )


            # ------------------------------------------------
            # CONFIDENCE STATUS
            # ------------------------------------------------

            if confidence_value >= 90:

                st.success(
                    "🟢 Very high confidence prediction"
                )

            elif confidence_value >= 70:

                st.warning(
                    "🟡 Moderate confidence prediction"
                )

            else:

                st.error(
                    "🔴 Low confidence prediction. "
                    "Consider using a clearer image."
                )


            # =================================================
            # WASTE MANAGEMENT RECOMMENDATION
            # =================================================

            info = WASTE_INFO[
                predicted_class
            ]


            st.divider()

            st.header(
                "♻️ What should you do with it?"
            )


            recommendation_col1, recommendation_col2 = (
                st.columns(
                    [1, 2]
                )
            )


            with recommendation_col1:

                st.markdown(
                    f"""
                    <div style="
                        padding:20px;
                        border-radius:15px;
                        border:1px solid #ddd;
                        text-align:center;
                    ">

                    <div style="font-size:50px;">
                    {info["icon"]}
                    </div>

                    <h3>{predicted_class}</h3>

                    <p>
                    <b>Suggested stream:</b><br>
                    {info["bin"]}
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with recommendation_col2:

                st.subheader(
                    "Recommended Action"
                )

                st.write(
                    info["recommendation"]
                )

                st.write(
                    info["description"]
                )


            st.subheader(
                "💡 Disposal Tips"
            )


            for tip in info["tips"]:

                st.write(
                    f"• {tip}"
                )


            st.caption(
                "Waste-management rules vary by location. "
                "Always follow your local authority's disposal "
                "and recycling guidelines."
            )


            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            st.divider()

            st.subheader(
                "📊 Class Probabilities"
            )


            probability_values = (
                st.session_state
                .current_probabilities
            )


            probability_data = []


            for class_name, probability in zip(
                CLASS_NAMES,
                probability_values
            ):

                percentage = (
                    float(probability) * 100
                )

                probability_data.append({

                    "Class":
                        class_name,

                    "Probability":
                        percentage
                })


            probability_data.sort(
                key=lambda x:
                    x["Probability"],

                reverse=True
            )


            for item in probability_data:

                st.write(
                    f"**{item['Class']}** — "
                    f"{item['Probability']:.2f}%"
                )

                st.progress(
                    min(
                        float(
                            item["Probability"]
                            / 100
                        ),
                        1.0
                    )
                )


# ============================================================
# DASHBOARD PAGE
# ============================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Waste Classification Dashboard"
    )

    st.write(
        "Overview of your AI-powered waste "
        "classification activity."
    )

    st.divider()


    history = load_history()


    if len(history) == 0:

        st.info(
            "No predictions have been recorded yet. "
            "Go to the Classifier and analyze some "
            "waste images."
        )


    else:

        df = pd.DataFrame(
            history
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        total_predictions = len(
            df
        )


        average_confidence = df[
            "confidence"
        ].mean()


        most_common_class = df[
            "class"
        ].mode()[0]


        high_confidence = len(
            df[
                df["confidence"] >= 90
            ]
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "Total Predictions",
                total_predictions
            )


        with col2:

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2f}%"
            )


        with col3:

            st.metric(
                "Most Detected",
                most_common_class
            )


        with col4:

            st.metric(
                "High Confidence",
                high_confidence
            )


        st.divider()


        # ----------------------------------------------------
        # WASTE DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "♻️ Waste Type Distribution"
        )


        class_counts = (
            df["class"]
            .value_counts()
            .reindex(
                CLASS_NAMES,
                fill_value=0
            )
        )


        chart_df = pd.DataFrame({

            "Waste Type":
                class_counts.index,

            "Predictions":
                class_counts.values

        })


        st.bar_chart(
            chart_df.set_index(
                "Waste Type"
            )
        )


        st.divider()


        # ----------------------------------------------------
        # CONFIDENCE HISTORY
        # ----------------------------------------------------

        st.subheader(
            "🎯 Confidence Over Time"
        )


        confidence_df = df[
            [
                "timestamp",
                "confidence"
            ]
        ].copy()


        confidence_df = (
            confidence_df
            .set_index(
                "timestamp"
            )
        )


        st.line_chart(
            confidence_df
        )


        st.divider()


        # ----------------------------------------------------
        # RECENT PREDICTIONS
        # ----------------------------------------------------

        st.subheader(
            "🕒 Recent Predictions"
        )


        recent_df = df.head(
            10
        ).copy()


        recent_df.columns = [

            "Time",
            "Waste Type",
            "Confidence"

        ]


        st.dataframe(
            recent_df,
            width="stretch",
            hide_index=True
        )


# ============================================================
# HISTORY PAGE
# ============================================================

elif page == "🕒 History":

    st.title(
        "🕒 Prediction History"
    )

    st.write(
        "View previous waste classifications."
    )

    st.divider()


    history = load_history()


    if len(history) == 0:

        st.info(
            "No prediction history yet."
        )


    else:

        df = pd.DataFrame(
            history
        )


        # ----------------------------------------------------
        # FILTER
        # ----------------------------------------------------

        selected_class = st.selectbox(

            "Filter by waste type",

            [
                "All"
            ] + CLASS_NAMES
        )


        if selected_class != "All":

            filtered_df = df[
                df["class"]
                == selected_class
            ]

        else:

            filtered_df = df


        st.write(
            f"Showing {len(filtered_df)} prediction(s)"
        )


        st.dataframe(
            filtered_df,
            width="stretch",
            hide_index=True
        )


        st.divider()


        # ----------------------------------------------------
        # CLEAR HISTORY
        # ----------------------------------------------------

        st.subheader(
            "History Management"
        )


        if st.button(
            "🗑️ Clear Prediction History"
        ):

            save_history(
                []
            )


            st.success(
                "Prediction history cleared."
            )


            st.rerun()


# ============================================================
# ABOUT MODEL PAGE
# ============================================================

elif page == "ℹ️ About Model":

    st.title(
        "ℹ️ About the Model"
    )


    st.write(
        "Smart Waste Classifier uses a trained "
        "ResNet18 deep learning model to classify "
        "images into six waste categories."
    )


    st.divider()


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(
        2
    )


    with col1:

        st.subheader(
            "🧠 Model Architecture"
        )

        st.write(
            "**Architecture:** ResNet18"
        )

        st.write(
            "**Input:** 224 × 224 RGB image"
        )

        st.write(
            "**Output:** 6 classes"
        )

        st.write(
            "**Training:** Transfer learning"
        )


    with col2:

        st.subheader(
            "📈 Performance"
        )

        st.metric(
            "Test Accuracy",
            "93.34%"
        )

        st.write(
            "**Test images:** 706"
        )

        st.write(
            "**Correct:** 659"
        )

        st.write(
            "**Incorrect:** 47"
        )


    st.divider()


    # --------------------------------------------------------
    # CLASSES
    # --------------------------------------------------------

    st.subheader(
        "♻️ Supported Waste Categories"
    )


    for class_name in CLASS_NAMES:

        info = WASTE_INFO[
            class_name
        ]

        st.write(
            f"{info['icon']} **{class_name}**"
        )


    st.divider()


    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    st.subheader(
        "⚙️ System Configuration"
    )


    config_col1, config_col2 = (
        st.columns(2)
    )


    with config_col1:

        st.write(
            "**Image Size:** 224 × 224"
        )

        st.write(
            "**Channels:** RGB / 3"
        )

        st.write(
            "**Normalization:** ImageNet"
        )


    with config_col2:

        st.write(
            "**Device:** " + str(device)
        )

        if torch.cuda.is_available():

            st.write(
                "**GPU:** "
                + torch.cuda.get_device_name(0)
            )

        else:

            st.write(
                "**GPU:** Not available"
            )


    st.divider()


    st.info(
        "The model's 93.34% test accuracy was measured "
        "on the held-out test set of 706 images."
    )


    st.caption(
        "Smart Waste Classifier • ResNet18 • "
        "Machine Learning Project"
    )