"""
SMART WASTE CLASSIFIER
TACO IMAGE DOWNLOADER

Downloads the actual TACO images referenced by:
detection/raw_taco/annotations.json

The official TACO downloader downloads images from Flickr.
"""

import os
import json
import time
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANNOTATION_FILE = os.path.join(
    PROJECT_DIR,
    "detection",
    "raw_taco",
    "annotations.json"
)

IMAGE_DIR = os.path.join(
    PROJECT_DIR,
    "detection",
    "raw_taco",
    "images"
)


# ============================================================
# SETTINGS
# ============================================================

TIMEOUT = 30

MAX_RETRIES = 3

SLEEP_BETWEEN_RETRIES = 2


# ============================================================
# CREATE IMAGE DIRECTORY
# ============================================================

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)


# ============================================================
# CHECK ANNOTATIONS
# ============================================================

print("=" * 70)
print("TACO IMAGE DOWNLOADER")
print("=" * 70)

print()
print("Annotation file:")
print(ANNOTATION_FILE)

if not os.path.isfile(ANNOTATION_FILE):

    print()
    print("ERROR:")
    print("annotations.json was not found.")

    raise SystemExit(1)


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print()
print("Reading annotations...")

with open(
    ANNOTATION_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


images = data.get(
    "images",
    []
)


print(
    f"Images listed in annotations: {len(images)}"
)


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_image(
    image_info,
    index
):

    file_name = image_info.get(
        "file_name"
    )

    original_url = image_info.get(
        "flickr_url"
    )

    resized_url = image_info.get(
        "flickr_640_url"
    )

    if not file_name:

        return False, "missing filename"

    # --------------------------------------------------------
    # TACO filenames can contain subdirectories
    # --------------------------------------------------------

    output_path = os.path.join(
        IMAGE_DIR,
        file_name
    )

    output_directory = os.path.dirname(
        output_path
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Already downloaded
    # --------------------------------------------------------

    if os.path.isfile(output_path):

        try:

            # Verify that it is actually an image
            with Image.open(output_path) as img:
                img.verify()

            return True, "already exists"

        except Exception:

            # Corrupted file
            try:
                os.remove(output_path)
            except:
                pass


    # --------------------------------------------------------
    # Try original URL first
    # --------------------------------------------------------

    urls = []

    if original_url:
        urls.append(original_url)

    if resized_url:
        urls.append(resized_url)


    # --------------------------------------------------------
    # Try URLs
    # --------------------------------------------------------

    last_error = None


    for url in urls:

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                response = requests.get(
                    url,
                    timeout=TIMEOUT,
                    headers={
                        "User-Agent":
                        "Mozilla/5.0"
                    }
                )

                response.raise_for_status()


                # ------------------------------------------------
                # Convert response to image
                # ------------------------------------------------

                image = Image.open(
                    BytesIO(
                        response.content
                    )
                )


                # ------------------------------------------------
                # Make sure RGB/RGBA
                # ------------------------------------------------

                if image.mode not in (
                    "RGB",
                    "RGBA"
                ):

                    image = image.convert(
                        "RGB"
                    )


                # ------------------------------------------------
                # Save
                # ------------------------------------------------

                image.save(
                    output_path
                )


                return True, "downloaded"


            except Exception as e:

                last_error = str(e)

                if attempt < MAX_RETRIES:

                    time.sleep(
                        SLEEP_BETWEEN_RETRIES
                    )


    return False, last_error


# ============================================================
# DOWNLOAD ALL IMAGES
# ============================================================

print()
print("=" * 70)
print("DOWNLOADING TACO IMAGES")
print("=" * 70)

print()
print("Destination:")
print(IMAGE_DIR)

print()
print(
    "This can take some time because TACO contains many images."
)

print()


downloaded = 0
already_exists = 0
failed = 0


# ------------------------------------------------------------
# Progress bar
# ------------------------------------------------------------

progress = tqdm(
    images,
    desc="Downloading",
    unit="image"
)


for index, image_info in enumerate(
    progress
):

    success, status = download_image(
        image_info,
        index
    )


    if success:

        if status == "already exists":

            already_exists += 1

        else:

            downloaded += 1

    else:

        failed += 1

        file_name = image_info.get(
            "file_name",
            "unknown"
        )

        print()
        print(
            f"FAILED: {file_name}"
        )

        print(
            f"Reason: {status}"
        )


    progress.set_postfix(
        downloaded=downloaded,
        existing=already_exists,
        failed=failed
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print()
print("=" * 70)
print("TACO DOWNLOAD COMPLETE")
print("=" * 70)

print()

print(
    f"Total images in annotations : {len(images)}"
)

print(
    f"Downloaded                  : {downloaded}"
)

print(
    f"Already existed             : {already_exists}"
)

print(
    f"Failed                      : {failed}"
)

print()

print(
    "Images stored in:"
)

print(
    IMAGE_DIR
)

print("=" * 70)


# ============================================================
# COUNT ACTUAL FILES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

actual_images = 0


for root, dirs, files in os.walk(
    IMAGE_DIR
):

    for file in files:

        extension = os.path.splitext(
            file
        )[1].lower()

        if extension in extensions:

            actual_images += 1


print()
print(
    f"Actual image files found: {actual_images}"
)

print()


# ============================================================
# NEXT STEP
# ============================================================

if actual_images > 0:

    print(
        "SUCCESS!"
    )

    print()
    print(
        "The TACO images are now available."
    )

    print()
    print(
        "Next run:"
    )

    print(
        "python prepare_detection_data.py"
    )

else:

    print(
        "WARNING:"
    )

    print(
        "No images were downloaded."
    )

print()