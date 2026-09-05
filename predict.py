import os
import sys
import io

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Paper",
    "Plastic"
]

IMAGE_SIZE = 224

# Model is in the same folder as this file
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "waste_resnet18_best.pth"
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("SMART WASTE CLASSIFIER")
print("=" * 65)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# 3. CREATE RESNET18 MODEL
# ============================================================

print("Creating ResNet18 model...")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model = model.to(device)

print("Model architecture created.")


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model file not found!")
    print()
    print("Expected location:")
    print(os.path.abspath(MODEL_PATH))
    print()

    sys.exit(1)


try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

except Exception as e:

    print("\nERROR: Could not load model!")
    print(e)

    sys.exit(1)


# ============================================================
# 5. HANDLE CHECKPOINT
# ============================================================

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

    state_dict = checkpoint["state_dict"]

else:

    state_dict = checkpoint


clean_state_dict = {}

for key, value in state_dict.items():

    # Remove DataParallel prefix if present
    if key.startswith("module."):
        key = key[7:]

    # Remove model wrapper prefix if present
    if key.startswith("model."):
        key = key[6:]

    clean_state_dict[key] = value


# ============================================================
# 6. LOAD WEIGHTS
# ============================================================

try:

    model.load_state_dict(
        clean_state_dict,
        strict=True
    )

except Exception as e:

    print("\nERROR: Model architecture does not match!")
    print(e)

    sys.exit(1)


model.eval()

print("Trained model loaded successfully!")
print()


# ============================================================
# 7. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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
# 8. CHECK IF INPUT IS A URL
# ============================================================

def is_url(path):

    return (
        path.startswith("http://")
        or
        path.startswith("https://")
    )


# ============================================================
# 9. LOAD IMAGE FROM LOCAL FILE
# ============================================================

def load_local_image(image_path):

    print("Loading local image...")

    if not os.path.exists(image_path):

        print()
        print("ERROR: Image file not found!")
        print()
        print("Path:")
        print(os.path.abspath(image_path))

        return None


    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        print("Local image loaded successfully.")

        return image

    except Exception as e:

        print()
        print("ERROR: Could not open image.")
        print(e)

        return None


# ============================================================
# 10. LOAD IMAGE FROM URL
# ============================================================

def load_url_image(url):

    print("Downloading image from web...")
    print("URL:", url)
    print()

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()


        # ----------------------------------------------------
        # Convert downloaded bytes into an image
        # ----------------------------------------------------

        image = Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")


        print("Web image downloaded successfully.")

        return image


    except requests.exceptions.RequestException as e:

        print()
        print("ERROR: Could not download image.")
        print(e)

        return None


    except Exception as e:

        print()
        print("ERROR: URL does not contain a valid image.")
        print(e)

        return None


# ============================================================
# 11. LOAD IMAGE
# ============================================================

def load_image(image_source):

    if is_url(image_source):

        return load_url_image(
            image_source
        )

    else:

        return load_local_image(
            image_source
        )


# ============================================================
# 12. PREDICT IMAGE
# ============================================================

def predict_image(image_source):

    print("=" * 65)
    print("IMAGE PREDICTION")
    print("=" * 65)

    print()
    print("Input:", image_source)
    print()


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(
        image_source
    )

    if image is None:

        return


    # --------------------------------------------------------
    # Show original image information
    # --------------------------------------------------------

    print()
    print("Original image size:", image.size)
    print("Image mode:", image.mode)


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    print()
    print("Input tensor shape:", image_tensor.shape)
    print()


    # ========================================================
    # 13. MODEL PREDICTION
    # ========================================================

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


    # --------------------------------------------------------
    # Get prediction
    # --------------------------------------------------------

    predicted_index = predicted_class.item()

    predicted_name = CLASS_NAMES[
        predicted_index
    ]

    confidence_percentage = (
        confidence.item() * 100
    )


    # ========================================================
    # 14. MAIN RESULT
    # ========================================================

    print("=" * 65)
    print("PREDICTION RESULT")
    print("=" * 65)

    print()

    print(
        f"Predicted waste : {predicted_name}"
    )

    print(
        f"Confidence      : {confidence_percentage:.2f}%"
    )

    print()


    # ========================================================
    # 15. ALL CLASS PROBABILITIES
    # ========================================================

    print("=" * 65)
    print("CLASS PROBABILITIES")
    print("=" * 65)

    probabilities = probabilities[
        0
    ].cpu()


    sorted_indices = torch.argsort(
        probabilities,
        descending=True
    )


    for index in sorted_indices:

        class_index = index.item()

        class_name = CLASS_NAMES[
            class_index
        ]

        probability = (
            probabilities[
                class_index
            ].item() * 100
        )

        print(
            f"{class_name:<20}"
            f": {probability:6.2f}%"
        )


    # ========================================================
    # 16. CONFIDENCE LEVEL
    # ========================================================

    print()

    print("=" * 65)
    print("CONFIDENCE ANALYSIS")
    print("=" * 65)


    if confidence_percentage >= 90:

        confidence_level = "VERY HIGH"

    elif confidence_percentage >= 75:

        confidence_level = "HIGH"

    elif confidence_percentage >= 60:

        confidence_level = "MODERATE"

    else:

        confidence_level = "LOW"


    print(
        f"Confidence level : {confidence_level}"
    )


    # ========================================================
    # 17. WARNING FOR LOW CONFIDENCE
    # ========================================================

    if confidence_percentage < 60:

        print()

        print(
            "WARNING: Model confidence is low."
        )

        print(
            "The image may be difficult to classify."
        )

        print(
            "Try a clearer image with the waste item"
        )

        print(
            "more visible."
        )


    # ========================================================
    # 18. FINISHED
    # ========================================================

    print()
    print("=" * 65)
    print("PREDICTION COMPLETE")
    print("=" * 65)


# ============================================================
# 19. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # No image provided
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("=" * 65)
        print("HOW TO USE")
        print("=" * 65)

        print()

        print("LOCAL IMAGE:")

        print(
            r'python predict.py "C:\Users\misty\Pictures\bottle.jpg"'
        )

        print()

        print("WEB IMAGE:")

        print(
            'python predict.py "https://example.com/bottle.jpg"'
        )

        print()

        print("The URL must point directly to an image.")

        print()

        print("=" * 65)

        sys.exit(1)


    # --------------------------------------------------------
    # Get input
    # --------------------------------------------------------

    image_source = sys.argv[1]


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predict_image(
        image_source
    )