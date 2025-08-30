import streamlit as st
import io
import re
import os
import json
import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
from collections import Counter
import pytesseract

# YouTube processing
import yt_dlp
import whisper
from pydub import AudioSegment

# NLP and AI
import evaluate
from bert_score import score as bert_score
import google.generativeai as genai
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
import spacy
# Standard library imports
import io
import re
import os
import json
import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
from collections import Counter
import pytesseract

# YouTube processing
import yt_dlp
import whisper
from pydub import AudioSegment

# NLP and AI
import evaluate
from bert_score import score as bert_score
import google.generativeai as genai
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
import spacy

# Machine Learning frameworks - ADD THESE IMPORTS
import torch
import torchvision
import torchaudio

# Transformers
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    LEDForConditionalGeneration, LEDTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss

# Additional imports
import tempfile
import requests
from pathlib import Path
import subprocess
import sys

# Transformers
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    LEDForConditionalGeneration, LEDTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
import faiss

# Additional imports
import tempfile
import requests
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Advanced Document Intelligence System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required resources
@st.cache_resource
def download_resources():
    # Download NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'mcqs' not in st.session_state:
    st.session_state.mcqs = None

# Download resources
download_resources()

# Text Extraction & Processing Functions
def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF file."""
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        rect = page.rect
        ignore_top = rect.height * 0.10
        ignore_bottom = rect.height * 0.85
        blocks = page.get_text("dict")["blocks"]
        page_text = []
        for block in blocks:
            if 'lines' in block:
                block_bbox = block["bbox"]
                if ignore_top < block_bbox[1] and block_bbox[3] < ignore_bottom:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_text.append(span["text"])
        full_text.append(" ".join(page_text))
    doc.close()
    return " ".join(full_text)

def clean_text(text):
    """Cleans extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def ocr_images_from_pdf(pdf_path, dpi=200):
    """Extracts text from images in PDF using OCR."""
    ocr_results = []
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        image_list = page.get_images(full=True)
        if not image_list:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            if text.strip():
                ocr_results.append((pno, text))
            continue
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            imgPIL = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(imgPIL)
            if text.strip():
                ocr_results.append((pno, text))
    doc.close()
    return ocr_results

def structured_extract_text(pdf_path, do_ocr_images=False):
    """Main text extraction function."""
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    
    # Detect equations
    lines = re.split(r'(?<=[\.\n])\s+', cleaned_text)
    clean_lines = []
    equations = []
    for line in lines:
        if not line.strip():
            continue
        non_alpha = sum(1 for c in line if not c.isalnum() and not c.isspace())
        ratio = non_alpha / max(1, len(line))
        if ratio > 0.3 or any(sym in line for sym in ['=', '∑', '∫', '\\frac']):
            equations.append(line.strip())
        else:
            clean_lines.append(line)
    final_clean_text = " ".join(clean_lines)

    ocr_texts = []
    if do_ocr_images:
        try:
            ocr_texts = ocr_images_from_pdf(pdf_path)
        except Exception as e:
            st.error(f"OCR failed: {e}")

    return {
        "text": final_clean_text,
        "equations": equations,
        "ocr_texts": ocr_texts
    }

# Language and entity functions
import streamlit as st
import os
import sys
import subprocess
import importlib

import streamlit as st
import os
import sys
import subprocess

# Check and install spaCy if needed
def setup_environment():
    """Set up the NLP environment"""
    # Try to import spaCy
    try:
        import spacy
        return True
    except ImportError:
        st.sidebar.warning("Installing spaCy...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "spacy"
            ], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                import importlib
                importlib.invalidate_caches()
                global spacy
                import spacy
                return True
            else:
                st.sidebar.error("Failed to install spaCy")
                return False
        except:
            return False

# Setup environment
spacy_available = setup_environment()

@st.cache_resource
def load_spacy_models():
    """Load spaCy models with fallbacks"""
    if not spacy_available:
        return None, None
    
    nlp_en, nlp_multi = None, None
    
    # Try to load English model
    try:
        nlp_en = spacy.load("en_core_web_sm")
    except OSError:
        try:
            # Download model
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                          capture_output=True, timeout=300)
            nlp_en = spacy.load("en_core_web_sm")
        except:
            try:
                nlp_en = spacy.blank("en")
            except:
                nlp_en = None
    
    # Try to load multi-language model  
    try:
        nlp_multi = spacy.load("xx_ent_wiki_sm")
    except OSError:
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "xx_ent_wiki_sm"],
                          capture_output=True, timeout=300)
            nlp_multi = spacy.load("xx_ent_wiki_sm")
        except:
            nlp_multi = nlp_en  # Fallback to English model
    
    return nlp_en, nlp_multi

# Load models
nlp_en, nlp_multi = load_spacy_models()
def detect_language(text):
    try:
        return detect(text[:1000])
    except:
        return 'en'

def extract_entities(text, lang='en'):
    if lang.startswith('en'):
        doc = nlp_en(text)
    else:
        doc = nlp_multi(text)
    return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

# Text chunking functions
def chunk_text_sentences(text, chunk_size=400, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        words = sentence.split()
        sentence_length = len(words)
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlapping_words = ' '.join(current_chunk).split()[-overlap:]
            current_chunk = [' '.join(overlapping_words)] if overlap > 0 else []
            current_length = len(overlapping_words)
        current_chunk.append(sentence)
        current_length += sentence_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def chunk_text_tokenwise(text, tokenizer, max_tokens=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    total_length = len(tokens)
    while start < total_length:
        end = min(start + max_tokens, total_length)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start = end - overlap
    return chunks

# YouTube Processing Functions
def download_youtube_audio(youtube_url, output_path="downloads"):
    os.makedirs(output_path, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # Enhanced bypass options
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        },
        'extract_flat': False,
        'ignoreerrors': True,
        'retries': 10,
        'fragment_retries': 10,
        'skip_unavailable_fragments': True,
        'keep_fragments': True,
        'no_check_certificate': True,
        'geo_bypass': True,
        'geo_bypass_country': 'US',
        'geo_bypass_ip_block': '0.0.0.0/0',
        'verbose': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'unknown_title')
            
            st.info(f"Attempting to download: {video_title}")
            
            # Try to download
            try:
                ydl.download([youtube_url])
            except Exception as download_error:
                st.warning(f"Standard download failed: {download_error}. Trying alternative method...")
                # Try with different format
                return download_youtube_audio_fallback(youtube_url, output_path)
            
            # Find the downloaded file
            expected_filename = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            
            if os.path.exists(expected_filename):
                return expected_filename, video_title
            else:
                # Look for any audio file in the directory
                for file in os.listdir(output_path):
                    if file.endswith(('.mp3', '.m4a', '.webm')):
                        return os.path.join(output_path, file), video_title
                
                return None, video_title
                
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None, None

def transcribe_audio(audio_path, model_size="base"):
    st.info(f"Loading Whisper {model_size} model...")
    
    try:
        model = whisper.load_model(model_size)
        st.info("Transcribing audio...")
        
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = model.transcribe(audio_path)
        return result["text"]
        
    except FileNotFoundError as e:
        st.error(f"Audio file not found: {e}")
        return ""
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        
        # Try alternative transcription method
        return transcribe_audio_fallback(audio_path)

def split_transcript(transcript, max_chunk_duration=10):
    words = transcript.split()
    chunk_size = max_chunk_duration * 100
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= chunk_size and '.' in word:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Summarization & QA Functions
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_summarization_model(model_choice="bart"):
    model_map = {
        "bart": ("facebook/bart-large-cnn", BartTokenizer, BartForConditionalGeneration),
        "pegasus": ("google/pegasus-xsum", PegasusTokenizer, PegasusForConditionalGeneration),
        "led": ("allenai/led-base-16384", LEDTokenizer, LEDForConditionalGeneration)
    }
    if model_choice not in model_map:
        model_choice = "bart"
    model_name, tokenizer_class, model_class = model_map[model_choice]
    st.info(f"Loading {model_choice} model...")
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name).to(device)
    return model, tokenizer, model_choice

def summarize_text(text, model, tokenizer, max_input_length=1024, max_summary_length=150):
    if not text.strip():
        return ""
    inputs = tokenizer(text, max_length=max_input_length, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=max_summary_length,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def hierarchical_summarization(text, model, tokenizer, model_choice="bart", target_words=None, compression_ratio=None):
    original_word_count = len(text.split())
    if target_words:
        max_summary_length = target_words
    elif compression_ratio:
        max_summary_length = max(50, int(original_word_count * compression_ratio))
    else:
        max_summary_length = 150
    if model_choice == "led":
        chunks = chunk_text_sentences(text, chunk_size=1000, overlap=100)
    else:
        chunks = chunk_text_sentences(text, chunk_size=400, overlap=50)
    st.info(f"Document split into {len(chunks)} chunks")
    chunk_summaries = []
    for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
        try:
            summary = summarize_text(chunk, model, tokenizer, max_summary_length=max_summary_length//2)
            chunk_summaries.append(summary)
        except Exception as e:
            st.error(f"Error summarizing chunk {i}: {e}")
            chunk_summaries.append("")
    combined_summary = " ".join(chunk_summaries)
    final_summary = summarize_text(combined_summary, model, tokenizer, max_summary_length=max_summary_length)
    return {
        "chunks": chunks,
        "chunk_summaries": chunk_summaries,
        "final_summary": final_summary,
        "word_count": original_word_count,
        "summary_length": len(final_summary.split())
    }

# QA System
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2').to(device)

embedder = load_embedder()

@st.cache_resource
def init_qa_system():
    return pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        tokenizer="distilbert-base-uncased-distilled-squad",
        device=0 if device == "cuda" else -1
    )

def build_search_index(chunks):
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    embeddings_np = chunk_embeddings.cpu().numpy()
    embeddings_np = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np.astype('float32'))
    return index, embeddings_np

def answer_question(question, chunks, index, qa_pipeline, top_k=3):
    if index is None or not chunks:
        return {
            'answer': 'No content available to answer questions',
            'score': 0.0,
            'relevant_chunks': []
        }
    
    try:
        question_embedding = embedder.encode([question], convert_to_tensor=True)
        question_embedding = question_embedding.cpu().numpy()
        
        # Handle 1D array
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.reshape(1, -1)
        
        # Normalize question embedding
        question_norm = np.linalg.norm(question_embedding, axis=1, keepdims=True)
        question_norm = np.where(question_norm == 0, 1, question_norm)
        question_embedding = question_embedding / question_norm
        
        distances, indices = index.search(question_embedding.astype('float32'), top_k)
        relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        context = "\n".join(relevant_chunks)
        
        result = qa_pipeline({'question': question, 'context': context})
        return {
            'answer': result['answer'],
            'score': result['score'],
            'relevant_chunks': relevant_chunks
        }
        
    except Exception as e:
        st.error(f"Error answering question: {e}")
        return {
            'answer': 'Error processing question',
            'score': 0.0,
            'relevant_chunks': []
        }

# Evaluation & Visualization Functions
def evaluate_summary(reference, generated):
    results = {}
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(predictions=[generated], references=[reference], use_stemmer=True)
    results['rouge'] = rouge_scores
    P, R, F1 = bert_score([generated], [reference], lang='en', verbose=False)
    results['bert_score'] = {'precision': P.mean().item(), 'recall': R.mean().item(), 'f1': F1.mean().item()}
    results['length_ratio'] = len(generated.split()) / max(1, len(reference.split()))
    return results

def create_wordcloud(text, title="Word Cloud"):
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig
def extract_entities(text, lang='en'):
    """
    Extract entities from text with debugging
    """
    print(f"Extracting entities from text: {text[:100]}...")  # Debug
    
    # If spaCy models aren't available, return empty list
    if nlp_en is None or not hasattr(nlp_en, 'pipe_names'):
        print("spaCy model not available")  # Debug
        return []
    
    try:
        if lang.startswith('en'):
            doc = nlp_en(text)
        else:
            doc = nlp_multi(text) if nlp_multi else nlp_en(text)
        
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        print(f"Found {len(entities)} entities: {entities}")  # Debug
        return entities
        
    except Exception as e:
        print(f"Entity extraction failed: {e}")  # Debug
        # Simple fallback: extract capitalized words as potential entities
        try:
            import re
            # Simple regex to find potential entities (capitalized words)
            potential_entities = re.findall(r'\b[A-Z][a-z]+\b(?:\s+[A-Z][a-z]+\b)*', text)
            fallback_entities = [{'text': ent, 'label': 'UNKNOWN'} for ent in potential_entities[:10]]
            print(f"Using fallback entities: {fallback_entities}")  # Debug
            return fallback_entities
        except Exception as fallback_error:
            print(f"Fallback entity extraction also failed: {fallback_error}")  # Debug
            return []
            
def plot_entity_types(entities):
    entity_counts = Counter([entity['label'] for entity in entities])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(entity_counts.keys()), list(entity_counts.values()))
    ax.set_xlabel('Count')
    ax.set_title('Named Entity Types in Document')
    plt.tight_layout()
    return fig

def save_report(content, filename="analysis_report", title="Document Analysis Report"):
    """
    Save report content to a file with multiple fallback options
    Returns the file path and MIME type
    """
    try:
        # Try PDF first
        try:
            pdf_path = f"{filename}.pdf"
            c = canvas.Canvas(pdf_path, pagesize=letter)
            width, height = letter
            margin = 50
            y_position = height - margin
            
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, y_position, title)
            y_position -= 30
            
            c.setFont("Helvetica", 12)
            lines = textwrap.wrap(content, width=80)
            
            for line in lines:
                if y_position < margin:
                    c.showPage()
                    y_position = height - margin
                    c.setFont("Helvetica", 12)
                c.drawString(margin, y_position, line)
                y_position -= 15
            
            c.save()
            
            if os.path.exists(pdf_path):
                return pdf_path, "application/pdf"
        except Exception as e:
            print(f"PDF creation failed: {e}")
        
        # Fallback to text file
        try:
            txt_path = f"{filename}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            if os.path.exists(txt_path):
                return txt_path, "text/plain"
        except Exception as e:
            print(f"Text file creation failed: {e}")
        
        # Final fallback: temporary file
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            return temp_path, "text/plain"
        except Exception as e:
            print(f"Temp file creation failed: {e}")
            raise Exception("All file creation methods failed")
            
    except Exception as e:
        # Ultimate fallback: return None and handle in calling code
        print(f"Error saving report: {e}")
        return None, None

def save_to_pdf(content, filename="document_analysis.pdf", title="Document Analysis Report"):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        margin = 50
        y_position = height - margin
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y_position, title)
        y_position -= 30
        
        c.setFont("Helvetica", 12)
        lines = textwrap.wrap(content, width=80)
        
        for line in lines:
            if y_position < margin:
                c.showPage()
                y_position = height - margin
                c.setFont("Helvetica", 12)
            c.drawString(margin, y_position, line)
            y_position -= 15
        
        c.save()
        
        # Check if file was actually created
        if os.path.exists(filename):
            return filename
        else:
            # Fallback: save as text file
            txt_filename = filename.replace('.pdf', '.txt')
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return txt_filename
            
    except Exception as e:
        # Fallback: create a simple text file
        fallback_filename = "analysis_report.txt"
        with open(fallback_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return fallback_filename

def create_download_button(file_path, mime_type, label="Download Report"):
    """Create a download button for a file"""
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                st.download_button(
                    label=label,
                    data=f,
                    file_name=os.path.basename(file_path),
                    mime=mime_type
                )
            return True
        except Exception as e:
            st.error(f"Error creating download button: {e}")
            return False
    else:
        st.warning("Report file could not be generated for download")
        return False

def save_models(model, tokenizer, save_path="saved_models"):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return save_path

# Gemini API Functions (MCQ Generation)
def setup_gemini_automatically():
    try:
        # Get API key from user
        if 'gemini_api_key' not in st.session_state:
            return False
            
        API_KEY = st.session_state.gemini_api_key
        if not API_KEY:
            st.error("Please enter your Gemini API key in the sidebar")
            return False
            
        genai.configure(api_key=API_KEY)
        models = list(genai.list_models())
        if any('generateContent' in model.supported_generation_methods for model in models):
            st.success("Gemini API configured successfully!")
            return True
        return False
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        return False

def generate_mcqs_with_gemini(context, num_questions=5):
    if not setup_gemini_automatically():
        return []
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""CONTEXT: {context[:4000]}
        Generate {num_questions} multiple choice questions based on the context.
        Each question should have 4 options (A, B, C, D), correct answer, and explanation.
        Return ONLY JSON format: [{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "correct_answer": "A", "explanation": "..."}}]"""
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        mcqs = json.loads(response_text)
        if not isinstance(mcqs, list):
            raise ValueError("Expected a list of MCQs")
        for mcq in mcqs:
            if not all(key in mcq for key in ['question', 'options', 'correct_answer', 'explanation']):
                raise ValueError("Invalid MCQ format")
        return mcqs[:num_questions]
    except Exception as e:
        st.error(f"Error generating MCQs: {e}")
        return []

def save_mcqs_to_file(mcqs, filename):
    try:
        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write("Generated Multiple Choice Questions\n")
            f.write("=" * 50 + "\n\n")
            for i, mcq in enumerate(mcqs, 1):
                f.write(f"{i}. {mcq['question']}\n")
                for option, text in mcq['options'].items():
                    f.write(f"   {option}. {text}\n")
                f.write(f"\n   Correct Answer: {mcq['correct_answer']}\n")
                f.write(f"   Explanation: {mcq['explanation']}\n")
                f.write("-" * 50 + "\n\n")
        return f"{filename}.txt"
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def process_pdf_document(uploaded_file, do_ocr, model_choice, target_words, compression_ratio):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
    
    try:
        extracted_data = structured_extract_text(pdf_path, do_ocr_images=do_ocr)
        lang = detect_language(extracted_data['text'])
        entities = extract_entities(extracted_data['text'], lang)
        
        model, tokenizer, model_choice = load_summarization_model(model_choice)
        summary_results = hierarchical_summarization(
            extracted_data['text'], model, tokenizer, model_choice,
            target_words=target_words, compression_ratio=compression_ratio
        )
        
        qa_pipeline = init_qa_system()
        search_index, embeddings = build_search_index(summary_results['chunks'])
        
        # Create visualizations
        wordcloud_fig = create_wordcloud(extracted_data['text'], "Document Word Cloud")
        entities_fig = plot_entity_types(entities)
        
        final_output = f"""DOCUMENT ANALYSIS REPORT
Document: {uploaded_file.name}
Language: {lang}
Original words: {summary_results['word_count']}
Summary words: {summary_results['summary_length']}
Compression ratio: {summary_results['word_count']/summary_results['summary_length']:.1f}:1

SUMMARY:
{summary_results['final_summary']}

KEY ENTITIES:
{', '.join([e['text'] for e in entities[:20]])}"""
        
        # Save report using robust method
        report_path, mime_type = save_report(final_output, "analysis_report")
        
        # Save models
        model_path = save_models(model, tokenizer)
        
        return {
            'extracted_data': extracted_data,
            'summary_results': summary_results,
            'entities': entities,
            'qa_system': (search_index, embeddings, qa_pipeline),
            'wordcloud_fig': wordcloud_fig,
            'entities_fig': entities_fig,
            'report_path': report_path,
            'report_mime_type': mime_type,
            'model_path': model_path
        }
    except Exception as e:
        st.error(f"Error processing document: {e}")
        # Return at least some data even if processing failed
        return {
            'error': str(e),
            'report_path': None,
            'report_mime_type': None
        }
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
        except:
            pass

def process_youtube_video(youtube_url, model_size, chunk_duration, model_choice, target_words):
    st.info("Downloading YouTube audio...")
    audio_path, video_title = download_youtube_audio(youtube_url)
    if not audio_path:
        return None
    
    st.info(f"Downloaded: {video_title}")
    transcript = transcribe_audio(audio_path, model_size)
    st.info(f"Transcription complete: {len(transcript.split())} words")
    transcript_chunks = split_transcript(transcript, chunk_duration)
    st.info(f"Split into {len(transcript_chunks)} chunks")

    model, tokenizer, _ = load_summarization_model(model_choice)
    chunk_summaries = []
    for i, chunk in enumerate(tqdm(transcript_chunks, desc="Summarizing chunks")):
        try:
            summary = summarize_text(chunk, model, tokenizer, max_summary_length=target_words//len(transcript_chunks))
            chunk_summaries.append(summary)
        except Exception as e:
            st.error(f"Error summarizing chunk {i}: {e}")
            chunk_summaries.append("")

    combined_summary = " ".join(chunk_summaries)
    final_summary = summarize_text(combined_summary, model, tokenizer, max_summary_length=target_words)
    lang = detect_language(final_summary)
    entities = extract_entities(final_summary, lang)
    qa_pipeline = init_qa_system()
    search_index, embeddings = build_search_index(transcript_chunks)

    final_output = f"""YOUTUBE VIDEO ANALYSIS REPORT
Video Title: {video_title}
Video URL: {youtube_url}
Transcript words: {len(transcript.split())}
Summary words: {len(final_summary.split())}
Language: {lang}

SUMMARY:
{final_summary}

KEY ENTITIES:
{', '.join([e['text'] for e in entities[:15]])}"""

    pdf_path = save_to_pdf(final_output, "youtube_analysis_report.pdf")
    
    # Create visualizations
    wordcloud_fig = create_wordcloud(transcript, "Transcript Word Cloud")
    entities_fig = plot_entity_types(entities)
    
    return {
        'video_data': {
            'video_title': video_title,
            'transcript': transcript,
            'chunks': transcript_chunks,
            'audio_path': audio_path
        },
        'summary': final_summary,
        'entities': entities,
        'qa_system': (search_index, embeddings, qa_pipeline),
        'wordcloud_fig': wordcloud_fig,
        'entities_fig': entities_fig,
        'pdf_path': pdf_path
    }

# Streamlit UI
def main():
    st.title("📚 Advanced Document Intelligence System")
    st.markdown("""
    This system processes PDF documents and YouTube videos to extract text, generate summaries,
    answer questions, and create multiple-choice questions based on the content.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Gemini API key input
        gemini_api_key = st.text_input("Gemini API Key", type="password", 
                                      help="Get your API key from https://aistudio.google.com/app/apikey")
        if gemini_api_key:
            st.session_state.gemini_api_key = gemini_api_key
        
        st.divider()
        
        # Model selection
        model_choice = st.selectbox(
            "Summarization Model",
            ["bart", "pegasus", "led"],
            index=0,
            help="Choose the model for summarization"
        )
        
        # Length control
        length_control = st.radio(
            "Length Control",
            ["Word Count", "Compression Ratio"],
            index=0
        )
        
        if length_control == "Word Count":
            target_words = st.slider("Target Word Count", 50, 500, 150)
            compression_ratio = None
        else:
            compression_ratio = st.slider("Compression Ratio", 0.05, 0.5, 0.2)
            target_words = None
            
        st.divider()
        
        # Processing options
        do_ocr = st.checkbox("Perform OCR on images", value=False)
        
        st.divider()
        
        # YouTube options
        st.subheader("YouTube Options")
        model_size = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=1
        )
        chunk_duration = st.slider("Chunk Duration (minutes)", 1, 30, 10)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["PDF Processing", "YouTube Processing", "QA System", "MCQ Generation"])
    
    with tab1:
        st.header("PDF Document Processing")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    results = process_pdf_document(
                        uploaded_file, do_ocr, model_choice, 
                        target_words, compression_ratio
                    )
                    
                    if results and 'error' not in results:
                        st.session_state.results = results
                        
                        # Display results
                        st.success("PDF processing completed!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Document Information")
                            st.write(f"**Original words:** {results['summary_results']['word_count']}")
                            st.write(f"**Summary words:** {results['summary_results']['summary_length']}")
                            st.write(f"**Compression ratio:** {results['summary_results']['word_count']/results['summary_results']['summary_length']:.1f}:1")
                            st.write(f"**Language:** {detect_language(results['extracted_data']['text'])}")
                            
                            st.subheader("Summary")
                            st.write(results['summary_results']['final_summary'])
                        
                        with col2:
                            st.pyplot(results['wordcloud_fig'])
                            st.pyplot(results['entities_fig'])
                        
                        # Download buttons
                        if 'report_path' in results and results['report_path']:
                            create_download_button(
                                results['report_path'], 
                                results.get('report_mime_type', 'text/plain'),
                                "Download Report"
                            )
                    elif results and 'error' in results:
                        st.error(f"Processing failed: {results['error']}")
                    else:
                        st.error("Processing failed with unknown error")
    
    with tab2:
        st.header("YouTube Video Processing")
        youtube_url = st.text_input("Enter YouTube URL")
        
        if youtube_url:
            if st.button("Process YouTube Video", type="primary"):
                with st.spinner("Processing YouTube video..."):
                    results = process_youtube_video(
                        youtube_url, model_size, chunk_duration,
                        model_choice, target_words
                    )
                    
                    if results and 'error' not in results:
                        st.session_state.results = results
                        
                        # Display results
                        st.success("YouTube video processing completed!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Video Information")
                            st.write(f"**Title:** {results['video_data']['video_title']}")
                            st.write(f"**Transcript words:** {len(results['video_data']['transcript'].split())}")
                            st.write(f"**Summary words:** {len(results['summary'].split())}")
                            st.write(f"**Language:** {detect_language(results['summary'])}")
                            
                            st.subheader("Summary")
                            st.write(results['summary'])
                        
                        with col2:
                            st.pyplot(results['wordcloud_fig'])
                            st.pyplot(results['entities_fig'])
                        
                        # Download buttons
                        if 'report_path' in results and results['report_path']:
                            create_download_button(
                                results['report_path'], 
                                results.get('report_mime_type', 'text/plain'),
                                "Download Report"
                            )
                    elif results and 'error' in results:
                        st.error(f"Processing failed: {results['error']}")
                    else:
                        st.error("Processing failed with unknown error")
    
    with tab3:
        st.header("Question Answering System")
        
        if st.session_state.results is None or 'error' in st.session_state.results:
            st.info("Please process a document or YouTube video first.")
        else:
            # Get context based on processing type
            if 'summary_results' in st.session_state.results:
                chunks = st.session_state.results['summary_results']['chunks']
                context = " ".join(chunks)
            else:
                context = st.session_state.results.get('video_data', {}).get('transcript', '')
                chunks = [context]
            
            # Initialize QA system if not already done
            if st.session_state.qa_system is None:
                qa_pipeline = init_qa_system()
                search_index, embeddings = build_search_index(chunks)
                st.session_state.qa_system = (search_index, embeddings, qa_pipeline)
            
            search_index, embeddings, qa_pipeline = st.session_state.qa_system
            
            # Question input
            question = st.text_input("Enter your question")
            
            use_gemini = st.checkbox("Use Gemini API for QA", value=False)
            
            if question:
                if use_gemini:
                    if not setup_gemini_automatically():
                        st.error("Please configure Gemini API first")
                    else:
                        try:
                            answer = ask_gemini(question, context)
                            st.write(f"**Answer (Gemini API):** {answer}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    result = answer_question(question, chunks, search_index, qa_pipeline)
                    st.write(f"**Answer (Local QA):** {result['answer']}")
                    st.write(f"**Confidence:** {result['score']:.3f}")
                    
                    with st.expander("Show relevant context"):
                        for i, chunk in enumerate(result['relevant_chunks']):
                            st.write(f"**Chunk {i+1}:** {chunk}")
    
    with tab4:
        st.header("MCQ Generation with Gemini")
        
        if st.session_state.results is None or 'error' in st.session_state.results:
            st.info("Please process a document or YouTube video first.")
        elif not st.session_state.gemini_api_key:
            st.info("Please enter your Gemini API key in the sidebar.")
        else:
            # Get context based on processing type
            if 'extracted_data' in st.session_state.results:
                context = st.session_state.results['extracted_data']['text']
                source = "PDF document"
            else:
                context = st.session_state.results['video_data']['transcript']
                source = "YouTube transcript"
            
            num_questions = st.slider("Number of MCQs to generate", 1, 20, 5)
            
            if st.button("Generate MCQs", type="primary"):
                with st.spinner("Generating MCQs..."):
                    mcqs = generate_mcqs_with_gemini(context, num_questions)
                    
                    if mcqs:
                        st.session_state.mcqs = mcqs
                        st.success(f"Generated {len(mcqs)} MCQs from {source}!")
                        
                        for i, mcq in enumerate(mcqs, 1):
                            with st.expander(f"Question {i}"):
                                st.write(f"**{mcq['question']}**")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    for option, text in mcq['options'].items():
                                        st.write(f"{option}. {text}")
                                
                                with col2:
                                    st.success(f"**Correct Answer:** {mcq['correct_answer']}")
                                    st.info(f"**Explanation:** {mcq['explanation']}")
                        
                        # Download button
                        filename = st.text_input("Filename for MCQs", value="generated_mcqs")
                        if st.button("Save MCQs to file"):
                            file_path = save_mcqs_to_file(mcqs, filename)
                            if file_path:
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        label="Download MCQs",
                                        data=f,
                                        file_name=f"{filename}.txt",
                                        mime="text/plain"
                                    )
                    else:
                        st.error("Failed to generate MCQs")
def ask_gemini(question, context):
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"Context: {context[:3000]}\nQuestion: {question}\nAnswer based on context. If not found, say so."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":

    main()














