import os
import re
import json
import logging
import sqlite3
import asyncio
from urllib.parse import unquote
from langchain_community.vectorstores import FAISS
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import faiss
import torch
from langdetect import detect
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import edge_tts
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# To avoid conflict with the faiss library, we alias the Langchain FAISS:
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
import cv2
import easyocr
# ------------------------------------------------------------------------------
# Basic setup and unified app instance
# ------------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates", static_folder="static")
working_dir = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore", category=UserWarning, message="Convert_system_message_to_human will be deprecated!")

UPLOAD_FOLDER = "static/uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists


# Initialize OCR Model
reader = easyocr.Reader(['en'])

# Initialize Embeddings & FAISS
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_db_path = "vector_db_dirtr"

try:
    vectordb = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded successfully.")
except Exception as e:
    print(f"Failed to load FAISS vector store: {e}")
    vectordb = None

# ------------------------------------------------------------------------------
# 1. Historical Region Q&A (Section)
# ------------------------------------------------------------------------------
def setup_vectorstore():
    persist_directory = os.path.join(working_dir, "vector_db_dir")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = LangchainFAISS.load_local(
        persist_directory, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

historical_faiss_index = setup_vectorstore()
top_k = 1

def get_retrieved_context(query):
    docs = historical_faiss_index.similarity_search(query, k=top_k)
    if docs:
        context = "Dictionary to followa    :\n" + "\n".join([doc.page_content for doc in docs])
    else:
        context = "Not enough relevant data found."
    return context

# Use a dedicated API key for this section.
HISTORICAL_GOOGLE_API_KEY = 'AIzaSyB2rdT1ZfKXqwVlePKeXlcUxltduC9psDU'
genai.configure(api_key=HISTORICAL_GOOGLE_API_KEY)
historical_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=HISTORICAL_GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

historical_template = """
You are a language model expert in describing the historical region of Sbiba.
Limit yourself to the data provided in the context.

User: {text}
Context: {context}
"""
historical_prompt = PromptTemplate.from_template(historical_template)
historical_chain = LLMChain(llm=historical_model, prompt=historical_prompt)

# Endpoints for Historical Q&A
@app.route("/historical")
def historical_index():
    # Render a dedicated page (e.g. historical_index.html)
    return render_template("historical_index.html")



@app.route("/historical_chat", methods=["POST"])
def historical_chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message requis"}), 400
    try:
        context = get_retrieved_context(user_message)
        response = historical_chain.invoke({"text": user_message, "context": context})
        return jsonify({"response": response["text"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------------------
# 2. Archaeological ChatBot with TTS (Section)
# ------------------------------------------------------------------------------
class ArchaeologicalChatBot:
    def __init__(self, json_paths, thesis_index_file="faiss_index_volume_1.bin"):
        self.data_store = []
        self.chat_history = []
        self.index = None
        
        # Initialize text embedding tools.
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Configure the LLM (using a separate API key).
        google_api_key = "AIzaSyDPNekCR-fsWRGFXRXIzBubcyUvY5KKOLQ"
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )
        
        self.thesis_entries = []
        self.arch_entries = []
        for path in json_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logging.info("Loaded %d entries from %s.", len(data), path)
            except Exception as e:
                logging.error("Error loading JSON from %s: %s", path, e)
                continue
            if "volume_1" in os.path.basename(path).lower():
                for entry in data:
                    description = entry.get("description", "") or entry.get("Text", "")
                    self.thesis_entries.append({
                        "site_number": entry.get("site_number", ""),
                        "location": entry.get("location", ""),
                        "description": description,
                        "images": entry.get("images", [])
                    })
            else:
                for entry in data:
                    self.arch_entries.append({
                        "site_number": entry.get("site_number", ""),
                        "location": entry.get("location", ""),
                        "description": entry.get("description", ""),
                        "images": entry.get("images", [])
                    })
        
        self.build_faiss_index(thesis_index_file)
        self.setup_database()
        
    def build_faiss_index(self, thesis_index_file):
        if self.thesis_entries:
            if os.path.exists(thesis_index_file):
                try:
                    index_thesis = faiss.read_index(thesis_index_file)
                    num_thesis = index_thesis.ntotal
                    thesis_embeddings = []
                    for i in range(num_thesis):
                        thesis_embeddings.append(index_thesis.reconstruct(i))
                    thesis_embeddings = np.array(thesis_embeddings, dtype=np.float32)
                    logging.info("Loaded %d thesis embeddings from %s.", num_thesis, thesis_index_file)
                except Exception as e:
                    logging.error("Error loading thesis FAISS index: %s. Recomputing embeddings.", e)
                    thesis_embeddings = np.array(
                        [self._vectorize_text(e["description"]) for e in self.thesis_entries],
                        dtype=np.float32
                    )
            else:
                thesis_embeddings = np.array(
                    [self._vectorize_text(e["description"]) for e in self.thesis_entries],
                    dtype=np.float32
                )
                logging.info("Prebuilt thesis index not found; computed %d embeddings.", len(self.thesis_entries))
        else:
            thesis_embeddings = np.empty((0, 384), dtype=np.float32)
        
        if self.arch_entries:
            arch_embeddings = np.array(
                [self._vectorize_text(e["description"]) for e in self.arch_entries],
                dtype=np.float32
            )
        else:
            arch_embeddings = np.empty((0, 384), dtype=np.float32)
        
        if thesis_embeddings.shape[0] == 0 and arch_embeddings.shape[0] == 0:
            logging.warning("No embeddings computed; FAISS index not created.")
            self.index = None
            return
        
        combined_embeddings = np.concatenate([thesis_embeddings, arch_embeddings], axis=0)
        self.data_store = self.thesis_entries + self.arch_entries
        d = combined_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(combined_embeddings)
        logging.info("Combined FAISS index built with %d entries.", combined_embeddings.shape[0])
        
    def setup_database(self):
        self.conn = sqlite3.connect("file:memdb1?mode=memory&cache=shared",
                                    uri=True, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                site_number TEXT,
                location TEXT,
                description TEXT,
                images TEXT
            )
        ''')
        for entry in self.data_store:
            images_json = json.dumps(entry.get("images", []))
            self.cursor.execute('''
                INSERT INTO sites (site_number, location, description, images)
                VALUES (?, ?, ?, ?)
            ''', (entry.get("site_number", ""),
                  entry.get("location", ""),
                  entry.get("description", ""),
                  images_json))
        self.conn.commit()
        logging.info("SQLite database created with %d entries.", len(self.data_store))
        
    def _vectorize_text(self, text):
        try:
            inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
            return vector
        except Exception as e:
            logging.error("Error vectorizing text: %s", e)
            return np.zeros(384, dtype=np.float32)
        
    def detect_language(self, text):
        try:
            return "fr" if detect(text) == "fr" else "en"
        except Exception as e:
            logging.warning("Language detection error: %s", e)
            return "en"
        
    def generate_response(self, question, context, lang):
        if lang == "fr":
            prompt_template = (
                "Vous êtes un expert en archéologie pour le site de Sbiba. "
                "Utilisez les informations ci-dessous pour répondre de manière claire et concise à la question de l'utilisateur.\n\n"
                "Informations:\n{context}\n\n"
                "Question: {question}\n\n"
                "Réponse:"
            )
        else:
            prompt_template = (
                "You are an archaeology expert for the Sbiba site. "
                "Using the information below, provide a clear and concise answer to the user's question.\n\n"
                "Information:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        try:
            answer = chain.run(context=context, question=question)
            return answer.strip()
        except Exception as e:
            logging.error("Error generating response: %s", e)
            return ("I'm sorry, I encountered an error processing your request."
                    if lang == "en" else "Désolé, une erreur est survenue lors du traitement de votre demande.")
        
    def query(self, question, k=3):
        if not isinstance(question, str) or not question.strip():
            return {"response": "Invalid question provided.", "images": []}
        
        lang = self.detect_language(question)
        greetings = ["hello", "hi", "hey", "bonjour", "salut"]
        if any(question.lower().strip().startswith(g) for g in greetings):
            resp = {
                "en": "Hello! I'm here to help with archaeological information about Sbiba. What would you like to know?",
                "fr": "Bonjour ! Je suis là pour fournir des informations archéologiques sur Sbiba. Que souhaitez-vous savoir ?"
            }
            answer = resp.get(lang, resp["en"])
            self.chat_history += [f"User: {question}", f"Bot: {answer}"]
            return {"response": answer, "images": []}
        
        m = re.search(r"site(?:\s*number)?\s*[:#]?\s*(\d+)", question, re.IGNORECASE)
        if m:
            site_num = m.group(1)
            padded = f"{int(site_num):03d}"
            self.cursor.execute(
                "SELECT site_number, location, description, images FROM sites WHERE site_number=? OR site_number=?",
                (site_num, padded)
            )
            row = self.cursor.fetchone()
            if row:
                site_number, location, description, imgs = row
                imgs = json.loads(imgs)
                answer = f"Site Number: {site_number}\nLocation: {location}\nDescription: {description}"
                self.chat_history += [f"User: {question}", f"Bot: {answer}"]
                return {"response": answer,
                        "images": [{"url": img, "caption": f"Site {site_number} - {location}"} for img in imgs]}
            else:
                msg = (f"I don't have any information about site number {site_num}."
                       if lang == "en" else f"Je n'ai aucune information sur le site numéro {site_num}.")
                self.chat_history += [f"User: {question}", f"Bot: {msg}"]
                return {"response": msg, "images": []}
        
        if self.index is None:
            msg = ("Sorry, I couldn't build a knowledge index." if lang == "en" 
                   else "Désolé, je n'ai pas pu construire un index de connaissance.")
            return {"response": msg, "images": []}
        
        query_emb = self._vectorize_text(question)
        distances, indices = self.index.search(np.array([query_emb], dtype=np.float32), k)
        valid = [i for i in indices[0] if 0 <= i < len(self.data_store)]
        if not valid:
            msg = ("Sorry, I couldn't find any relevant archaeological information."
                   if lang == "en" else "Désolé, je n'ai trouvé aucune information archéologique pertinente.")
            self.chat_history += [f"User: {question}", f"Bot: {msg}"]
            return {"response": msg, "images": []}
        
        context_parts = []
        imgs_info = []
        for i in valid:
            entry = self.data_store[i]
            context_parts.append(
                f"Site Number: {entry.get('site_number', 'N/A')}\n"
                f"Location: {entry.get('location', 'N/A')}\n"
                f"Description: {entry.get('description', '')}"
            )
            for img in entry.get("images", []):
                imgs_info.append({"url": img, "caption": f"Site {entry.get('site_number', 'N/A')} - {entry.get('location', 'N/A')}"})
        context = "\n\n".join(context_parts)
        answer = self.generate_response(question, context, lang)
        self.chat_history += [f"User: {question}", f"Bot: {answer}"]
        return {"response": answer, "images": imgs_info}

# Text-to-Speech (TTS) helpers.
voices = {
    "en": {
        "male": {
            "serious": ["en-US-EricNeural"]
        },
        "female": {
            "serious": ["en-US-MichelleNeural"]
        }
    },
    "fr": {
        "male": {
            "serious": ["fr-FR-HenriNeural"]
        }
    }
}

def select_voice(language, gender="male", style="serious"):
    if language not in voices:
        language = "en"
    voice_options = voices[language][gender].get(style)
    if not voice_options:
        voice_options = voices[language][gender].get("serious")
    if isinstance(voice_options, list) and len(voice_options) > 0:
        return voice_options[0]
    return voice_options

async def text_to_speech(text, language, output_file="static/audio_output.mp3", style="serious"):
    try:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        voice = select_voice(language, style=style)
        logging.info(f"Selected voice: {voice}")
        tts = edge_tts.Communicate(text=text, voice=voice)
        await tts.save(output_file)
        logging.info(f"Audio saved to {output_file}")
    except Exception as e:
        logging.error(f"TTS conversion error: {e}")

# Instantiate the archaeological chatbot with its data sources.
bot = ArchaeologicalChatBot(["data.json", "structured_data_volume_1.json"])

# Endpoints for the Archaeological ChatBot
@app.route("/chatbot")
def chatbot_home():
    return render_template("chat.html")

@app.route("/query", methods=["POST"])
def query_archaeology():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "A question is required"}), 400

    result = bot.query(question)

    # Create unique filenames to avoid collisions
    unique_id = uuid.uuid4().hex
    audio_filename = f"audio_output_{unique_id}.mp3"
    audio_filename_fr = f"audio_output_fr_{unique_id}.mp3"
    audio_path = os.path.join("static", audio_filename)
    audio_path_fr = os.path.join("static", audio_filename_fr)

    lang = bot.detect_language(result["response"])
    
    try:
        asyncio.run(text_to_speech(result["response"], lang, output_file=audio_path))
        result["audio_url"] = f"/static/{audio_filename}"
    except Exception as e:
        logging.error(f"Error generating TTS audio: {e}")
        result["audio_url"] = ""

    try:
        asyncio.run(text_to_speech(result["response"], "fr", output_file=audio_path_fr))
        result["audio_url_fr"] = f"/static/{audio_filename_fr}"
    except Exception as e:
        logging.error(f"Error generating French TTS audio: {e}")
        result["audio_url_fr"] = ""
    
    return jsonify(result)
@app.route('/extracted_images/<path:filename>')
def serve_image(filename):
    for directory in ['extracted_images', 'extracted_images_volume_1']:
        try:
            safe_filename = unquote(filename)
            return send_from_directory(directory, safe_filename)
        except Exception:
            continue
    logging.error("Image %s not found in any directory", filename)
    return jsonify({"error": "Image not found"}), 404

@app.route("/clear_history", methods=["POST"])
def clear_history():
    bot.chat_history = []
    return jsonify({"message": "Conversation history cleared."})

# ------------------------------------------------------------------------------
# 3. Unified Home Page
# ------------------------------------------------------------------------------
@app.route("/")
def home():
    # A simple landing page with links to both functionalities.
    return render_template("index.html")

@app.route("/ar")
def ar():
    return render_template("ar.html")
from flask import Flask, Response, send_file, request




@app.route('/audio')
def stream_audio():
    file_path = "static/audio_output.mp3"
    
    def generate():
        with open(file_path, "rb") as f:
            while chunk := f.read(1024):  # Read file in chunks
                yield chunk

    return Response(generate(), mimetype="audio/mpeg")
GOOGLE_API_KEY = 'AIzaSyB2rdT1ZfKXqwVlePKeXlcUxltduC9psDU'
if not GOOGLE_API_KEY:
    raise ValueError("Google API key is missing!")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)
import os
import cv2
import easyocr
import warnings
from flask import Flask, request, render_template, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Convert_system_message_to_human will be deprecated!")


UPLOAD_FOLDER = "static/uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

# Initialize OCR Model
reader = easyocr.Reader(['en'])

# Initialize Embeddings & FAISS
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_db_path = "vector_db_dirtr"

try:
    vectordb = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded successfully.")
except Exception as e:
    print(f"Failed to load FAISS vector store: {e}")
    vectordb = None

# Initialize Google Generative AI Model
GOOGLE_API_KEY = 'AIzaSyB2rdT1ZfKXqwVlePKeXlcUxltduC9psDU'
if not GOOGLE_API_KEY:
    raise ValueError("Google API key is missing!")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

# Define Prompt Template
templatet = """
You are a skilled language model adept at deciphering Latin mosaic inscriptions. Given an input of Latin words, your task is to:
**Translate the Text:** Translate the transcribed Latin text into English, providing a clear and accurate interpretation using this context: 
Text to translate: {text}
Context: {context}
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(templatet)

# Function to retrieve context using FAISS
def get_retrieved_contextt(query, top_k=2):
    if not vectordb:
        return "FAISS vector store not loaded."
    
    try:
        docs = vectordb.similarity_search(query, k=top_k)
        context = "\n".join(doc.page_content for doc in docs) if docs else "Not enough relevant data found."
    except Exception as e:
        context = f"Error retrieving context: {e}"
    return context

# Function to process an uploaded image and perform OCR
def process_image(image_path):
    img = cv2.imread(image_path, 0)
    results = reader.readtext(img)
    score_threshold = 0.5
    extracted_text = " ".join(entry[1] for entry in results if entry[2] > score_threshold)
    return extracted_text


# Flask Route: OCR Processing
@app.route("/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Perform OCR on image
    extracted_text = process_image(img_path)
    if not extracted_text:
        return jsonify({"error": "No text found in image"}), 400

    return jsonify({"text": extracted_text})

# Flask Route: Translate Extracted Text
@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    extracted_text = data["text"]

    # Retrieve relevant context using FAISS
    context = get_retrieved_contextt(extracted_text)

    # Generate translation using Gemini AI
    try:
        from langchain.chains import LLMChain
        qa_chain = LLMChain(llm=model, prompt=QA_CHAIN_PROMPT)
        result = qa_chain.invoke({"text": extracted_text, "context": context})
    except Exception as e:
        return jsonify({"error": f"Translation failed: {e}"}), 500

    return jsonify({
        "extracted_text": extracted_text,
        "translation": result,
        "context": context
    })

@app.route("/translation")
def translation():
    return render_template("translation.html")

with open("quiz.json", "r", encoding="utf-8") as f:
    quiz_data = json.load(f)["quiz"]
import random
# Shuffle questions to make it more engaging
random.shuffle(quiz_data)

@app.route("/quiz")
def quiz():
    question = random.choice(quiz_data)
    return render_template("quiz.html", question=question)

@app.route("/check_answer", methods=["POST"])
def check_answer():
    data = request.json
    selected_option = data["selected_option"]
    correct_answer = data["correct_answer"]
    justification = data["justification"]

    if selected_option == correct_answer:
        return jsonify({"correct": True, "justification": justification})
    else:
        return jsonify({"correct": False, "justification": justification})
if __name__ == '__main__':
    app.run(debug=True)