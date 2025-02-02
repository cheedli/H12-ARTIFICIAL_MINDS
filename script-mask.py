import cv2
import numpy as np
from pathlib import Path
import os

def resize_image(image, width=1100):
    height, current_width = image.shape[:2]
    aspect_ratio = height / current_width
    new_height = int(width * aspect_ratio)
    resized_image = cv2.resize(image, (width, new_height))
    return resized_image

def create_mask(image_path, mask_path, brush_size=10, log_file="processed.log"):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image: {image_path}")
        return False

    original_img = img.copy()
    img = resize_image(img)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    drawing = {"is_drawing": False, "brush_size": brush_size}

    scale = 1.0
    translation = [0, 0]

    def draw_circle(event, x, y, flags, param):
        nonlocal scale, translation

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["is_drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing["is_drawing"]:
                adjusted_x = int((x - translation[0]) / scale)
                adjusted_y = int((y - translation[1]) / scale)
                if 0 <= adjusted_x < mask.shape[1] and 0 <= adjusted_y < mask.shape[0]:
                    cv2.circle(mask, (adjusted_x, adjusted_y), drawing["brush_size"], (255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["is_drawing"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                scale = min(scale + 0.1, 5.0)
            else:
                scale = max(scale - 0.1, 0.5)
            print(f"Zoom scale: {scale}")

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_circle)

    while True:
        height, width = img.shape[:2]
        display = cv2.resize(img, (int(width * scale), int(height * scale)))
        tx, ty = translation

        mask_resized = cv2.resize(mask, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST)
        display[mask_resized > 0] = [0, 0, 255]

        cv2.imshow("Image", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  
            cv2.destroyAllWindows()
            os._exit(0)  
        elif key == ord('s'):  
            cv2.imwrite(str(mask_path), mask)
            print(f"Mask saved to {mask_path}")
            return True
        elif key == ord('r'):  
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            print("Mask reset")
        elif key == ord('+') or key == ord('='):  
            drawing["brush_size"] = min(drawing["brush_size"] + 1, 50)
            print(f"Brush size increased to {drawing['brush_size']}")
        elif key == ord('-'):  
            drawing["brush_size"] = max(drawing["brush_size"] - 1, 1)
            print(f"Brush size decreased to {drawing['brush_size']}")

    cv2.destroyAllWindows()

def process_images(input_dir, output_dir, log_file="processed.log"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_path = Path(log_file)
    processed_files = set()
    if log_path.exists():
        with log_path.open("r") as log:
            processed_files = {line.strip() for line in log}

    images = [f for f in input_path.iterdir() if f.is_file()]
    images.sort()

    

    
    for image_file in input_path.iterdir():
        if image_file.name in processed_files:
            print(f"Skipping already processed file: {image_file.name}")
            continue

        print(f"Processing: {image_file.name}")
        mask_file = output_path / f"mask_{image_file.stem}.jpg"
        try:
            if create_mask(image_file, mask_file, log_file=log_file):
                with log_path.open("a") as log:
                    log.write(f"{image_file.name}\n")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    input_dir = "E:\Mosaic data\mosaic-complete-incomplete\Hack"
    output_dir = "E:\Mosaic data\mosaic-complete-incomplete\HackMasks"
    process_images(input_dir, output_dir)


