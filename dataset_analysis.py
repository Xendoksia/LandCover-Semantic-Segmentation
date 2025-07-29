import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# 🔧 Ana dizin
BASE_DIR = r"--------------------------------------------------------------------------------"

REGIONS = ["Urban", "Rural"]

CLASS_COLORS = {
    0: [0, 0, 0],           # No data
    1: [169, 169, 169],     # Background
    2: [255, 0, 0],         # Building
    3: [255, 255, 0],       # Road
    4: [0, 0, 255],         # Water
    5: [139, 69, 19],       # Barren
    6: [34, 139, 34],       # Forest
    7: [0, 255, 255],       # Agriculture
}

CLASS_NAMES = {
    0: "No data",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Forest",
    7: "Agriculture"
}


def load_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def visualize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def analyze_class_distribution(mask_paths):
    pixel_counts = {class_id: 0 for class_id in CLASS_NAMES.keys()}
    for mask_path in mask_paths:
        mask = load_mask(mask_path)
        for class_id in np.unique(mask):
            pixel_counts[class_id] += np.sum(mask == class_id)
    return pixel_counts


def plot_distribution(distribution, region_name):
    class_ids = list(distribution.keys())
    pixel_counts = [distribution[cid] for cid in class_ids]
    class_labels = [CLASS_NAMES[cid] for cid in class_ids]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_labels, pixel_counts, color=[np.array(CLASS_COLORS[cid]) / 255 for cid in class_ids])
    plt.title(f"{region_name} - Sınıf Piksel Dağılımı")
    plt.ylabel("Toplam Piksel Sayısı")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def show_random_examples(image_dir, mask_dir, region_name, num=2):
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    selected_files = random.sample(mask_files, min(num, len(mask_files)))

    for filename in selected_files:
        mask_path = os.path.join(mask_dir, filename)
        image_path = os.path.join(image_dir, filename)

        image = load_image(image_path)
        mask = load_mask(mask_path)
        color_mask = visualize_mask(mask)

        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Orijinal Görsel")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Grayscale Maske")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(color_mask)
        plt.title("Renkli Maske")
        plt.axis('off')

        plt.suptitle(f"[{region_name}] {filename}")
        plt.tight_layout()
        plt.show()


def analyze_region(region_name):
    print(f"\n Bölge: {region_name}")

    region_path = os.path.join(BASE_DIR, region_name)
    image_dir = os.path.join(region_path, "images")
    mask_dir = os.path.join(region_path, "masks_png")

    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")]
    print(f" - Toplam maske: {len(mask_paths)}")

    class_distribution = analyze_class_distribution(mask_paths)
    for cid, count in class_distribution.items():
        name = CLASS_NAMES.get(cid, "Bilinmeyen")
        print(f"   {cid} ({name}): {count} piksel")

    plot_distribution(class_distribution, region_name)
    show_random_examples(image_dir, mask_dir, region_name)


def main():
    print(" LOVELDA Dataset Analysis Initialized")
    for region in REGIONS:
        analyze_region(region)


if __name__ == "__main__":
    main()
