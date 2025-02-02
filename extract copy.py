import fitz  # PyMuPDF
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Directory to store extracted images
IMAGE_DIR = "extracted_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define the size threshold (2KB)
SIZE_THRESHOLD = 2 * 1024  # 2KB in bytes

# Function to extract text and images
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_data = []
    image_index = 1
    site_data = None  # Initialize as None

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        page_images = []

        # Image extraction code (keep your existing implementation)
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]

            if len(image_bytes) < SIZE_THRESHOLD:
                continue

            image_name = f"Fig. {image_index}.{img_ext}"
            image_path = os.path.join(IMAGE_DIR, image_name)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            page_images.append(image_name)
            image_index += 1

        # Handle text organization
        if "Site n°" in text:
            if site_data:  # Commit previous site data
                extracted_data.append(site_data)
            # Start new site entry
            site_data = {
                "Text": text.strip(),
                "Images": page_images.copy()
            }
        else:
            if not site_data:  # First page without site header
                site_data = {
                    "Text": text.strip(),
                    "Images": page_images.copy()
                }
            else:  # Continue existing site
                site_data["Text"] += "\n" + text.strip()
                site_data["Images"].extend(page_images)

    # Add the final site data if exists
    if site_data:
        extracted_data.append(site_data)

    return extracted_data
# Function to create FAISS index
def create_faiss_index(data):
    texts = [site["Text"] for site in data]
    embeddings = np.array([model.encode(text) for text in texts]).astype("float32")

    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts

# Main processing function
def process_pdf(pdf_path, output_json):
    print("Extracting text and images...")
    extracted_data = extract_text_and_images(pdf_path)

    print("Creating FAISS index...")
    index, texts = create_faiss_index(extracted_data)

    # Save data
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)

    faiss.write_index(index, "faiss_index.bin")

    print(f"Data saved to {output_json} and FAISS index saved to faiss_index.bin.")

# Run script
if __name__ == "__main__":
    pdf_path = "Volume II catalogue_removed.pdf"  # Replace with your actual PDF
    output_json = "structured_data.json"

    process_pdf(pdf_path, output_json)
