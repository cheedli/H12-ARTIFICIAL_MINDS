import fitz  # PyMuPDF
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Directory to store extracted images (specific to this PDF)
IMAGE_DIR = "extracted_images_volume_1"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define the size threshold (2KB)
SIZE_THRESHOLD = 2 * 1024  # 2KB in bytes

# Function to extract text and images
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_data = []
    image_index = 0
    site_data = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Extract images
        images = page.get_images(full=True)
        page_images = []
        for img in images:
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                print(f"Error extracting image {xref} on page {page_num + 1}: {e}")
                continue  # Skip this image if extraction fails

            image_bytes = base_image.get("image")
            img_ext = base_image.get("ext", "png")  # Default to png if extension is missing

            # Check image size
            if len(image_bytes) < SIZE_THRESHOLD:
                print(f"Skipping image {xref} on page {page_num + 1} (size {len(image_bytes)} bytes)")
                continue  # Skip this image

            # Define image path with volume-specific naming
            image_path = os.path.join(IMAGE_DIR, f"volume_1_{image_index}.{img_ext}")

            try:
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
            except Exception as e:
                print(f"Error saving image {image_path}: {e}")
                continue  # Skip saving this image if an error occurs

            page_images.append(image_path)
            image_index += 1

        # Process site data based on specific keywords
        if "Site n°" in text or "INTRODUCTION GENERALE" in text:
            if site_data:
                extracted_data.append(site_data)
            site_data = {"Text": text.strip(), "Images": page_images}
        else:
            if site_data:
                site_data["Text"] += "\n" + text.strip()
                site_data["Images"].extend(page_images)

    if site_data:
        extracted_data.append(site_data)

    return extracted_data

# Function to create FAISS index
def create_faiss_index(data, index_filename):
    texts = [site["Text"] for site in data]
    embeddings = np.array([model.encode(text) for text in texts]).astype("float32")

    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index to file
    faiss.write_index(index, index_filename)
    return index

# Main processing function
def process_pdf(pdf_path, output_json, faiss_index_file):
    print("Extracting text and images...")
    extracted_data = extract_text_and_images(pdf_path)

    print("Creating FAISS index...")
    index = create_faiss_index(extracted_data, faiss_index_file)

    # Save extracted data to JSON
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON data to {output_json}: {e}")
        return

    print(f"Data saved to {output_json} and FAISS index saved to {faiss_index_file}.")

# Run script
if __name__ == "__main__":
    pdf_path = "Synthèse volume I (3).pdf"  # Replace with your actual PDF path
    output_json = "structured_data_volume_1.json"
    faiss_index_file = "faiss_index_volume_1.bin"

    process_pdf(pdf_path, output_json, faiss_index_file)
