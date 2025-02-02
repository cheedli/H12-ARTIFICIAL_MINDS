import fitz
import os
import json
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple


@dataclass
class Reference:
    source: str
    page: Optional[str]
    details: str


@dataclass
class SiteData:
    site_number: str
    location: Optional[str]
    description: str
    references: List[Reference]
    coordinates: Optional[str]
    images: List[str]


class PDFSiteExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.image_dir = "extracted_images"
        self.size_threshold = 2 * 1024  # 2KB (Ignore small images)
        os.makedirs(self.image_dir, exist_ok=True)

    def extract_full_text(self, pdf_path: str) -> Tuple[str, List[int]]:
        """Extracts full text and keeps track of where each page starts."""
        doc = fitz.open(pdf_path)
        page_texts = [page.get_text("text") for page in doc]
        
        # Keep track of where each page starts in the full text
        start_indices = []
        full_text = ""
        for i, text in enumerate(page_texts):
            start_indices.append(len(full_text))  # Save start position of each page
            full_text += text + "\n"

        return self.clean_text(full_text), start_indices

    def clean_text(self, text: str) -> str:
        """Removes unwanted elements like page numbers and extra whitespace."""
        text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)  # Remove page numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove excessive whitespace
        return text

    def segment_sites(self, text: str) -> List[Tuple[str, int, int]]:
        """Splits the text into segments based on 'Site N°' and tracks start/end positions."""
        site_pattern = r'((?:SITE|Site)\s*[nN]°\s*\d+.*?)(?=(?:SITE|Site)\s*[nN]°\s*\d+|$)'
        matches = list(re.finditer(site_pattern, text, re.DOTALL))
        
        sites = []
        for i, match in enumerate(matches):
            site_text = match.group(1)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sites.append((site_text, start, end))

        return sites

    def extract_site_number(self, text: str) -> Optional[str]:
        """Extracts the site number from the text."""
        site_match = re.search(r'(?:SITE|Site)\s*[nN]°\s*(\d+)', text, re.IGNORECASE)
        return site_match.group(1) if site_match else None

    def extract_coordinates(self, text: str) -> Optional[str]:
        """Extracts geographic coordinates if available."""
        coord_match = re.search(r'feuille\s+de\s+([^,]+),\s*(\d+\.\d+)', text)
        return f"{coord_match.group(1)}: {coord_match.group(2)}" if coord_match else None

    def extract_references(self, text: str) -> List[Reference]:
        """Extracts references and footnotes from the text."""
        references = []
        footnote_pattern = r'(\d+)\s+(AAT\d*,.*?(?=\d+\s+AAT|\Z))'
        matches = re.finditer(footnote_pattern, text, re.DOTALL)

        for match in matches:
            ref_num, ref_text = match.groups()
            source_match = re.match(r'(AAT\d*,.*?):(.+)', ref_text.strip())
            if source_match:
                source, details = source_match.groups()
                references.append(Reference(source=source.strip(), page=ref_num, details=details.strip()))
            else:
                references.append(Reference(source="Unknown", page=ref_num, details=ref_text.strip()))

        return references

    def extract_images(self, pdf_path: str) -> Dict[int, List[str]]:
        """Extracts images from all pages and returns a mapping of page numbers to image file names."""
        doc = fitz.open(pdf_path)
        image_index = 1
        page_images = {}

        for page_num, page in enumerate(doc):
            images_on_page = []
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]

                if len(image_bytes) < self.size_threshold:
                    continue  # Ignore small images

                image_name = f"Fig {image_index}.{base_image['ext']}"
                image_path = os.path.join(self.image_dir, image_name)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                images_on_page.append(image_name)
                image_index += 1

            if images_on_page:
                page_images[page_num] = images_on_page  # Store images by page number

        return page_images

    def assign_images_to_sites(self, sites: List[Tuple[str, int, int]], start_indices: List[int], page_images: Dict[int, List[str]]) -> List[Dict]:
        """Assigns extracted images to the correct site based on their position in the text."""
        sites_data = []
        for site_text, start, end in sites:
            site_number = self.extract_site_number(site_text)
            if site_number:
                # Find corresponding page numbers
                site_pages = [i for i, pos in enumerate(start_indices) if start <= pos < end]
                
                # Assign images appearing within those page ranges
                site_images = []
                for page in site_pages:
                    if page in page_images:
                        site_images.extend(page_images[page])

                site_data = SiteData(
                    site_number=site_number,
                    location=self.extract_coordinates(site_text),
                    description=site_text,
                    references=self.extract_references(site_text),
                    coordinates=self.extract_coordinates(site_text),
                    images=site_images
                )
                sites_data.append(asdict(site_data))

        return sites_data

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Processes the entire PDF, correctly segmenting sites and assigning images."""
        print("Extracting full text...")
        full_text, start_indices = self.extract_full_text(pdf_path)

        print("Extracting images from PDF...")
        page_images = self.extract_images(pdf_path)  # Get images by page

        print("Segmenting sites...")
        site_segments = self.segment_sites(full_text)

        print("Assigning images to correct sites...")
        sites_data = self.assign_images_to_sites(site_segments, start_indices, page_images)

        return sites_data

    def create_faiss_index(self, sites_data: List[Dict]) -> None:
        """Creates and saves a FAISS index for similarity searching."""
        texts = [site["description"] for site in sites_data]
        embeddings = np.array([self.model.encode(text) for text in texts]).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, "faiss_index.bin")


def main():
    extractor = PDFSiteExtractor()
    pdf_path = "Volume II catalogue_removed.pdf"
    output_json = "structured_sites_data.json"

    print("Processing PDF and extracting site data...")
    sites_data = extractor.process_pdf(pdf_path)

    print("Creating FAISS index for similarity search...")
    extractor.create_faiss_index(sites_data)

    print("Saving structured data...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sites_data, f, indent=4, ensure_ascii=False)

    print(f"Processing complete. Data saved to {output_json}")


if __name__ == "__main__":
    main()
