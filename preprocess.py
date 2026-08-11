
"""
Skin Disease Dataset Augmentation Pipeline
============================================

Supports dataset structures like:

dataset/
├── train/
│   ├── Acne/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Eczema/
│   ├── Melanoma/
│   └── ...
│
└── test/
    ├── Acne/
    ├── Eczema/
    ├── Melanoma/
    └── ...

The output will preserve the same structure:

augmented_dataset/
├── train/
│   ├── Acne/
│   │   ├── original images
│   │   ├── augmented images
│   │   └── ...
│   └── ...
│
└── test/
    ├── Acne/
    │   └── original images ONLY
    └── ...

IMPORTANT:
- Training images are augmented.
- Test images are copied without augmentation.
- No GPU required.
- Uses OpenCV, NumPy and PIL.
"""

import argparse
import io
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# 1. SYNTHETIC HAIR
# ============================================================

def add_synthetic_hair(
    image,
    num_hairs=None,
    min_thickness=1,
    max_thickness=3
):
    """
    Add synthetic hair strands to the image.
    """

    h, w = image.shape[:2]
    out = image.copy()

    if num_hairs is None:
        num_hairs = random.randint(8, 25)

    hair_colors = [
        (20, 15, 10),
        (45, 30, 20),
        (90, 65, 45),
        (150, 130, 100),
        (60, 60, 60),
    ]

    overlay = out.copy()

    for _ in range(num_hairs):

        color = random.choice(hair_colors)
        thickness = random.randint(
            min_thickness,
            max_thickness
        )

        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)

        points = [(x0, y0)]

        min_length = max(10, int(0.2 * min(h, w)))
        max_length = max(min_length, int(0.9 * min(h, w)))

        length = random.randint(
            min_length,
            max_length
        )

        angle = random.uniform(0, 2 * np.pi)
        curve_strength = random.uniform(-0.6, 0.6)

        n_segments = random.randint(4, 8)

        x, y = x0, y0

        for _ in range(n_segments):

            angle += curve_strength * random.uniform(
                -0.4,
                0.4
            )

            step = length / n_segments

            x += step * np.cos(angle)
            y += step * np.sin(angle)

            # Keep points inside image
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)

            points.append((int(x), int(y)))

        pts = np.array(points, dtype=np.int32)

        cv2.polylines(
            overlay,
            [pts],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    alpha = random.uniform(0.55, 0.85)

    out = cv2.addWeighted(
        overlay,
        alpha,
        out,
        1 - alpha,
        0
    )

    return out


# ============================================================
# 2. SKIN TONE / COLOR VARIATION
# ============================================================

def shift_skin_tone(
    image,
    l_shift=None,
    a_shift=None,
    b_shift=None
):
    """
    Modify LAB color channels to simulate
    different illumination and skin-tone conditions.
    """

    if l_shift is None:
        l_shift = random.uniform(-25, 20)

    if a_shift is None:
        a_shift = random.uniform(-8, 8)

    if b_shift is None:
        b_shift = random.uniform(-8, 10)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    l, a, b = cv2.split(lab)

    l = np.clip(
        l + l_shift,
        0,
        255
    )

    a = np.clip(
        a + a_shift,
        0,
        255
    )

    b = np.clip(
        b + b_shift,
        0,
        255
    )

    lab_shifted = cv2.merge(
        [l, a, b]
    ).astype(np.uint8)

    return cv2.cvtColor(
        lab_shifted,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# 3. BLUR AND NOISE
# ============================================================

def add_blur_noise(image, mode=None):
    """
    Apply Gaussian blur, motion blur,
    sensor noise or JPEG compression.
    """

    if mode is None:
        mode = random.choice([
            "gaussian",
            "motion",
            "noise",
            "jpeg",
            "combo"
        ])

    out = image.copy()

    # Gaussian blur
    if mode in ("gaussian", "combo"):

        k = random.choice([3, 5, 7])

        out = cv2.GaussianBlur(
            out,
            (k, k),
            0
        )

    # Motion blur
    if mode in ("motion", "combo"):

        k = random.choice([5, 7, 9])

        kernel = np.zeros(
            (k, k),
            dtype=np.float32
        )

        kernel[k // 2, :] = 1.0

        angle = random.uniform(
            0,
            180
        )

        M = cv2.getRotationMatrix2D(
            (k / 2, k / 2),
            angle,
            1
        )

        kernel = cv2.warpAffine(
            kernel,
            M,
            (k, k)
        )

        kernel_sum = kernel.sum()

        if kernel_sum != 0:
            kernel /= kernel_sum

        out = cv2.filter2D(
            out,
            -1,
            kernel
        )

    # Sensor noise
    if mode in ("noise", "combo"):

        sigma = random.uniform(
            5,
            20
        )

        noise = np.random.normal(
            0,
            sigma,
            out.shape
        ).astype(np.float32)

        out = np.clip(
            out.astype(np.float32) + noise,
            0,
            255
        ).astype(np.uint8)

    # JPEG compression
    if mode in ("jpeg", "combo"):

        quality = random.randint(
            20,
            50
        )

        pil_img = Image.fromarray(
            cv2.cvtColor(
                out,
                cv2.COLOR_BGR2RGB
            )
        )

        buffer = io.BytesIO()

        pil_img.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        out = cv2.cvtColor(
            np.array(
                Image.open(buffer)
            ),
            cv2.COLOR_RGB2BGR
        )

    return out


# ============================================================
# 4. ZOOM IN
# ============================================================

def zoom_close(
    image,
    crop_fraction=None
):
    """
    Simulate a closer camera capture.
    """

    h, w = image.shape[:2]

    if crop_fraction is None:
        crop_fraction = random.uniform(
            0.5,
            0.75
        )

    ch = int(h * crop_fraction)
    cw = int(w * crop_fraction)

    max_y_shift = max(1, h // 12)
    max_x_shift = max(1, w // 12)

    y0 = (
        (h - ch) // 2
        + random.randint(
            -max_y_shift,
            max_y_shift
        )
    )

    x0 = (
        (w - cw) // 2
        + random.randint(
            -max_x_shift,
            max_x_shift
        )
    )

    y0 = max(
        0,
        min(y0, h - ch)
    )

    x0 = max(
        0,
        min(x0, w - cw)
    )

    cropped = image[
        y0:y0 + ch,
        x0:x0 + cw
    ]

    return cv2.resize(
        cropped,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )


# ============================================================
# 5. ZOOM OUT
# ============================================================

def zoom_far(
    image,
    pad_fraction=None
):
    """
    Simulate a farther camera capture.
    """

    h, w = image.shape[:2]

    if pad_fraction is None:
        pad_fraction = random.uniform(
            0.2,
            0.45
        )

    new_h = int(
        h * (1 - pad_fraction)
    )

    new_w = int(
        w * (1 - pad_fraction)
    )

    new_h = max(
        1,
        min(new_h, h)
    )

    new_w = max(
        1,
        min(new_w, w)
    )

    shrunk = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    pad_top = (h - new_h) // 2
    pad_bottom = h - new_h - pad_top

    pad_left = (w - new_w) // 2
    pad_right = w - new_w - pad_left

    out = cv2.copyMakeBorder(
        shrunk,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101
    )

    blurred_bg = cv2.GaussianBlur(
        out,
        (9, 9),
        0
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    mask[
        pad_top:pad_top + new_h,
        pad_left:pad_left + new_w
    ] = 255

    mask = cv2.GaussianBlur(
        mask,
        (21, 21),
        0
    )

    mask3 = cv2.merge([
        mask,
        mask,
        mask
    ]).astype(np.float32) / 255.0

    out = (
        out.astype(np.float32) * mask3
        + blurred_bg.astype(np.float32) * (1 - mask3)
    ).astype(np.uint8)

    return out


# ============================================================
# AUGMENTATION DICTIONARY
# ============================================================

AUGMENTATIONS = {
    "hair": add_synthetic_hair,
    "tone": shift_skin_tone,
    "blur": add_blur_noise,
    "zoom_close": zoom_close,
    "zoom_far": zoom_far,
}


# ============================================================
# AUGMENT ONE IMAGE
# ============================================================

def augment_image(
    image,
    num_ops=None
):
    """
    Apply 1-3 random augmentations.
    """

    if num_ops is None:
        num_ops = random.randint(1, 3)

    num_ops = min(
        num_ops,
        len(AUGMENTATIONS)
    )

    ops = random.sample(
        list(AUGMENTATIONS.keys()),
        k=num_ops
    )

    out = image.copy()

    for op in ops:
        out = AUGMENTATIONS[op](out)

    return out, ops


# ============================================================
# PROCESS A SINGLE FOLDER
# ============================================================

def process_split(
    input_split,
    output_split,
    per_image=4,
    augment=True
):
    """
    Process either train or test.

    TRAIN:
        Original + augmented images

    TEST:
        Original images only
    """

    input_split = Path(input_split)
    output_split = Path(output_split)

    if not input_split.exists():
        print(
            f"WARNING: Folder not found: {input_split}"
        )
        return 0

    total_original = 0
    total_augmented = 0

    # Find all class folders
    class_dirs = [
        d for d in input_split.iterdir()
        if d.is_dir()
    ]

    if not class_dirs:
        print(
            f"WARNING: No class folders found in {input_split}"
        )
        return 0

    for class_dir in sorted(class_dirs):

        out_class_dir = (
            output_split / class_dir.name
        )

        out_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        images = [
            f for f in class_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        ]

        print(
            f"\n[{input_split.name}/{class_dir.name}] "
            f"{len(images)} images"
        )

        for img_path in images:

            image = cv2.imread(
                str(img_path)
            )

            if image is None:
                print(
                    f"Could not read: {img_path}"
                )
                continue

            # ------------------------------------------------
            # COPY ORIGINAL
            # ------------------------------------------------

            original_output = (
                out_class_dir / img_path.name
            )

            cv2.imwrite(
                str(original_output),
                image
            )

            total_original += 1

            # ------------------------------------------------
            # AUGMENT ONLY TRAIN
            # ------------------------------------------------

            if augment:

                for i in range(per_image):

                    aug_img, ops_used = augment_image(
                        image
                    )

                    ops_name = "_".join(
                        ops_used
                    )

                    out_name = (
                        f"{img_path.stem}"
                        f"_aug{i}"
                        f"_{ops_name}"
                        f".jpg"
                    )

                    output_path = (
                        out_class_dir / out_name
                    )

                    cv2.imwrite(
                        str(output_path),
                        aug_img
                    )

                    total_augmented += 1

    return (
        total_original,
        total_augmented
    )


# ============================================================
# PROCESS COMPLETE DATASET
# ============================================================

def process_dataset(
    input_dir,
    output_dir,
    per_image=4
):
    """
    Process dataset containing train and test folders.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(
            f"ERROR: Input folder does not exist:\n"
            f"{input_dir}"
        )
        return

    train_dir = input_dir / "train"
    test_dir = input_dir / "val"

    output_train = output_dir / "train"
    output_test = output_dir / "val"

    print("=" * 70)
    print("SKIN DISEASE DATASET AUGMENTATION")
    print("=" * 70)

    print(f"\nInput dataset : {input_dir}")
    print(f"Output dataset: {output_dir}")

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESSING TRAIN DATA")
    print("=" * 70)

    train_original, train_augmented = process_split(
        train_dir,
        output_train,
        per_image=per_image,
        augment=True
    )

    # ========================================================
    # TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESSING TEST DATA")
    print("=" * 70)

    test_original, test_augmented = process_split(
        test_dir,
        output_test,
        per_image=per_image,
        augment=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nTRAIN:"
        f"\n  Original images   : {train_original}"
        f"\n  Augmented images  : {train_augmented}"
        f"\n  Total             : "
        f"{train_original + train_augmented}"
    )

    print(
        f"\nTEST:"
        f"\n  Original images   : {test_original}"
        f"\n  Augmented images  : {test_augmented}"
        f"\n  Total             : "
        f"{test_original + test_augmented}"
    )

    print(
        f"\nOutput saved to:\n{output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "CPU-only skin disease dataset augmentation"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Root dataset folder containing train and test"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for augmented dataset"
    )

    parser.add_argument(
        "--per_image",
        type=int,
        default=4,
        help=(
            "Number of augmented copies "
            "per training image"
        )
    )

    args = parser.parse_args()

    process_dataset(
        args.input,
        args.output,
        per_image=args.per_image
    )

