import json
import numpy as np
import faiss
import torch
import os
from PIL import Image
from langdetect import detect
from transformers import AutoTokenizer, AutoModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class ArchaeologicalRAG:
    def __init__(self, json_path):
        """Initialise le système RAG avec FAISS et le modèle de génération."""
        self.data_store = []
        self.index = faiss.IndexFlatL2(384)  # 384 dimensions pour MiniLM

        # Modèle de vectorisation de texte
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Modèle de génération de texte
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key="AIzaSyDPNekCR-fsWRGFXRXIzBubcyUvY5KKOLQ",  # Remplacez par votre clé API
            temperature=0.3
        )

        # Charger et indexer les données JSON
        self.load_json_data(json_path)

    def load_json_data(self, json_path):
        """Charge les données JSON et les indexe dans FAISS"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            text = entry.get("description", "").strip()
            if not text:
                continue  # Ignorer les entrées sans description
            
            text_embedding = self._vectorize_text(text)

            # Stocker les données
            self.data_store.append({
                "site_number": entry.get("site_number", ""),
                "location": entry.get("location", ""),
                "description": text,
                "images": entry.get("images", [])  # Liste des images liées
            })

            # Ajouter l'embedding à FAISS
            self.index.add(np.array([text_embedding]))

    def _vectorize_text(self, text):
        """Convertit un texte en vecteur"""
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

    def detect_language(self, text):
        """Détecte la langue de la question (français ou anglais)"""
        try:
            lang = detect(text)
            return "fr" if lang == "fr" else "en"
        except:
            return "en"  # Par défaut, répondre en anglais si la détection échoue

    def display_images(self, images_list):
        """Affiche les images associées aux résultats"""
        for img_path in images_list:
            if os.path.exists(img_path):  # Vérifier si l'image existe avant de l'afficher
                img = Image.open(img_path)
                img.show()  # Ouvre l'image dans le visualiseur d'images
            else:
                print(f"⚠️ Image non trouvée : {img_path}")

    def query(self, question, k=3):
        """Recherche dans FAISS et génère une réponse en affichant les images associées"""
        query_embedding = self._vectorize_text(question)

        # Recherche des k meilleurs résultats
        distances, indices = self.index.search(np.array([query_embedding]), k)

        # Vérifier si des résultats sont retournés
        valid_indices = [i for i in indices[0] if i >= 0 and i < len(self.data_store)]
        if not valid_indices:
            return "Aucune information archéologique spécifique trouvée sur ce sujet."

        # Construire un contexte détaillé
        context_entries = [self.data_store[i]["description"] for i in valid_indices]
        context = "\n\n".join(context_entries)

        # Récupération des images associées aux résultats extraits
        images_list = []
        for i in valid_indices:
            images = self.data_store[i].get("images", [])
            if images:
                images_list.extend(images)

        # Affichage des images récupérées
        if images_list:
            print("\n📷 Images associées aux résultats trouvés :")
            for img in images_list:
                print(f" - {img}")

            # Afficher les images dans une fenêtre
            self.display_images(images_list)

        # Détection de la langue de la question
        lang = self.detect_language(question)

        # Adapter le prompt pour inclure les images et répondre dans la bonne langue
        if lang == "fr":
            prompt_template = """
            [Expert en archéologie tunisienne]
            Vous êtes un expert en archéologie et histoire tunisienne. Répondez de manière détaillée et éducative en intégrant les informations du contexte.

            Contexte archéologique pertinent :
            {context}

            Images associées aux vestiges retrouvés :
            {images}

            Question posée : {question}

            Fournissez une réponse informative en détaillant les aspects historiques, archéologiques et culturels :
            """
        else:
            prompt_template = """
            [Expert in Tunisian archaeology]
            You are an expert in Tunisian archaeology and history. Answer in a detailed and educational manner by integrating information from the provided context.

            Relevant archaeological context:
            {context}

            Images associated with the discovered remains:
            {images}

            Question asked: {question}

            Provide an informative response detailing historical, archaeological, and cultural aspects:
            """

        prompt = PromptTemplate.from_template(prompt_template)

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(context=context, images=", ".join(images_list), question=question)

        # Ajouter la liste des images à la fin de la réponse
        if images_list:
            response += "\n\n📌 **Images associées à cette analyse :**\n"
            response += "\n".join([f"- {img}" for img in images_list])

        return response

# === Exécution principale ===
if __name__ == "__main__":
    rag = ArchaeologicalRAG("data.json")  # Remplacez par le chemin de votre fichier JSON


"""
    # Exemple de requête
    question_fr = "Quels sont les vestiges retrouvés sur le site n°012 CNSAMH de Sbiba et que révèlent-ils sur l’histoire du lieu ?"
    response_fr = rag.query(question_fr)
    print("\n🔎 Réponse experte (FR) :")
    print(response_fr)
"""

question_en = "What archaeological discoveries have been made in Sbiba, and what do they reveal about its ancient history?"
response_en = rag.query(question_en)
print("\n🔎 Expert answer (EN) :")
print(response_en)
