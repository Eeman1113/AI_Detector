import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.hub
import re
import os

# --- Configuration ---
MODEL1_PATH = "modernbert.bin" # Make sure this file is in the same directory or provide the full path
MODEL2_URL = "https://huggingface.co/mihalykiss/modernbert_2/resolve/main/Model_groups_3class_seed12"
MODEL3_URL = "https://huggingface.co/mihalykiss/modernbert_2/resolve/main/Model_groups_3class_seed22"
BASE_MODEL = "answerdotai/ModernBERT-base"
NUM_LABELS = 41

# --- Device Setup ---
@st.cache_resource
def get_device():
    """Gets the appropriate torch device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEVICE = get_device()

# --- Model and Tokenizer Loading (Cached) ---
@st.cache_resource
def load_tokenizer(model_name):
    """Loads the tokenizer."""
    st.info(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    st.info("Tokenizer loaded.")
    return tokenizer

@st.cache_resource
def load_model(model_path_or_url, base_model, num_labels, is_url=False, _device=DEVICE):
    """Loads a sequence classification model from local path or URL."""
    model_name = os.path.basename(model_path_or_url) if not is_url else model_path_or_url.split('/')[-1]
    st.info(f"Loading model structure: {base_model}...")
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
    st.info(f"Loading model weights: {model_name}...")
    try:
        if is_url:
            state_dict = torch.hub.load_state_dict_from_url(model_path_or_url, map_location=_device, progress=True)
        else:
            if not os.path.exists(model_path_or_url):
                 st.error(f"Model file not found at {model_path_or_url}. Please ensure it's in the correct location.")
                 st.stop() # Stop execution if local model is missing
            state_dict = torch.load(model_path_or_url, map_location=_device)
        model.load_state_dict(state_dict)
        model.to(_device).eval()
        st.info(f"Model {model_name} loaded and moved to {_device}.")
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        st.stop() # Stop execution on model loading error


TOKENIZER = load_tokenizer(BASE_MODEL)
MODEL_1 = load_model(MODEL1_PATH, BASE_MODEL, NUM_LABELS, is_url=False, _device=DEVICE)
MODEL_2 = load_model(MODEL2_URL, BASE_MODEL, NUM_LABELS, is_url=True, _device=DEVICE)
MODEL_3 = load_model(MODEL3_URL, BASE_MODEL, NUM_LABELS, is_url=True, _device=DEVICE)


# --- Label Mapping ---
LABEL_MAPPING = {
    0: '13B', 1: '30B', 2: '65B', 3: '7B', 4: 'GLM130B', 5: 'bloom_7b',
    6: 'bloomz', 7: 'cohere', 8: 'davinci', 9: 'dolly', 10: 'dolly-v2-12b',
    11: 'flan_t5_base', 12: 'flan_t5_large', 13: 'flan_t5_small',
    14: 'flan_t5_xl', 15: 'flan_t5_xxl', 16: 'gemma-7b-it', 17: 'gemma2-9b-it',
    18: 'gpt-3.5-turbo', 19: 'gpt-35', 20: 'gpt4', 21: 'gpt4o',
    22: 'gpt_j', 23: 'gpt_neox', 24: 'human', 25: 'llama3-70b', 26: 'llama3-8b',
    27: 'mixtral-8x7b', 28: 'opt_1.3b', 29: 'opt_125m', 30: 'opt_13b',
    31: 'opt_2.7b', 32: 'opt_30b', 33: 'opt_350m', 34: 'opt_6.7b',
    35: 'opt_iml_30b', 36: 'opt_iml_max_1.3b', 37: 't0_11b', 38: 't0_3b',
    39: 'text-davinci-002', 40: 'text-davinci-003'
}
HUMAN_LABEL_INDEX = 24 # Assuming 'human' is always index 24

# --- Text Processing Functions ---
def clean_text(text):
    """Cleans the input text using regex."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text) # Handle hyphenated words broken by newline
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text) # Replace single newlines with spaces
    text = text.strip()
    return text

def classify_text(text, tokenizer, model_1, model_2, model_3, device, label_mapping, human_label_index):
    """Classifies the text using the ensemble of models."""
    if not text or not text.strip():
        return None # Indicate no classification needed for empty text

    cleaned_text = clean_text(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=tokenizer.model_max_length).to(device)

    with torch.no_grad():
        try:
            logits_1 = model_1(**inputs).logits
            logits_2 = model_2(**inputs).logits
            logits_3 = model_3(**inputs).logits

            softmax_1 = torch.softmax(logits_1, dim=1)
            softmax_2 = torch.softmax(logits_2, dim=1)
            softmax_3 = torch.softmax(logits_3, dim=1)

            # Ensemble by averaging probabilities
            averaged_probabilities = (softmax_1 + softmax_2 + softmax_3) / 3
            probabilities = averaged_probabilities[0].cpu() # Move to CPU for numpy/python processing

            # Separate human vs AI probability
            human_prob = probabilities[human_label_index].item() * 100

            # Calculate AI probability (sum of all non-human labels)
            ai_probs = probabilities.clone()
            ai_probs[human_label_index] = 0 # Zero out human probability for AI calculation
            ai_total_prob = ai_probs.sum().item() * 100

            # Find the most likely AI model
            ai_argmax_index = torch.argmax(ai_probs).item()
            ai_argmax_model = label_mapping.get(ai_argmax_index, "Unknown AI")

            # Determine final classification
            if human_prob > ai_total_prob:
                return {"is_human": True, "probability": human_prob, "model": "Human"}
            else:
                return {"is_human": False, "probability": ai_total_prob, "model": ai_argmax_model}
        except Exception as e:
             st.error(f"Error during model inference: {e}")
             return {"error": True}


# --- Streamlit UI ---
st.set_page_config(page_title="AI Text Detector", layout="centered")

# Inject Custom CSS for highlighting
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    body, .stTextArea textarea, .stMarkdown, .stButton button {
        font-family: 'Roboto Mono', sans-serif !important;
    }
    .stTextArea textarea {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        font-size: 16px; /* Adjusted for better fit */
        padding: 15px;
        background-color: #f0fff0; /* Light green background */
    }
     .stButton button {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        padding: 10px 24px;
        width: 100%;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
     }
     .stButton button:hover {
        background-color: #45a049;
        color: white;
        border-color: #45a049;
     }

    .result-box {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        font-size: 18px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        background-color: #f9f9f9;
        box-shadow: 0px 0px 5px rgba(0,0,0,0.1);
    }

    .highlight-human {
        color: #4CAF50 !important; /* Use !important to override potential conflicts */
        font-weight: bold;
        background: rgba(76, 175, 80, 0.2);
        padding: 5px 8px; /* Added padding */
        border-radius: 8px;
        display: inline-block; /* Ensures padding and background apply correctly */
    }

    .highlight-ai {
        color: #FF5733 !important; /* Use !important */
        font-weight: bold;
        background: rgba(255, 87, 51, 0.2);
        padding: 5px 8px; /* Added padding */
        border-radius: 8px;
        display: inline-block; /* Ensures padding and background apply correctly */
    }

    .footer {
        text-align: center;
        margin-top: 50px;
        font-weight: bold;
        font-size: 16px; /* Adjusted size */
        color: #555; /* Slightly muted color */
    }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ AI Text Detector 🤖")
# st.markdown(description) # Add description if you have one

# --- Input Area ---
input_text = st.text_area(
    label="Enter text to analyze:",
    placeholder="Type or paste your content here...",
    height=200,
    key="text_input"
)

# --- Analyze Button and Output ---
analyze_button = st.button("Analyze Text", key="analyze_button")
result_placeholder = st.empty() # Create a placeholder for the result output

if analyze_button:
    if input_text and input_text.strip():
        with st.spinner('Analyzing text... This might take a moment.'):
            # --- Perform Classification ---
            classification_result = classify_text(
                input_text,
                TOKENIZER,
                MODEL_1,
                MODEL_2,
                MODEL_3,
                DEVICE,
                LABEL_MAPPING,
                HUMAN_LABEL_INDEX
            )

        # --- Display Result ---
        if classification_result and not classification_result.get("error"):
            if classification_result["is_human"]:
                prob = classification_result['probability']
                result_html = (
                    f"<div class='result-box'>"
                    f"<b>The text is</b> <span class='highlight-human'><b>{prob:.2f}%</b> likely <b>Human written</b>.</span>"
                    f"</div>"
                )
            else:
                prob = classification_result['probability']
                model_name = classification_result['model']
                result_html = (
                    f"<div class='result-box'>"
                    f"<b>The text is</b> <span class='highlight-ai'><b>{prob:.2f}%</b> likely <b>AI generated</b>.</span><br><br>"
                    f"<b>Identified AI Model: {model_name}</b>"
                    f"</div>"
                )
            result_placeholder.markdown(result_html, unsafe_allow_html=True)
        elif classification_result and classification_result.get("error"):
             result_placeholder.error("An error occurred during analysis. Please try again.")
        else:
             result_placeholder.warning("Please enter some text to analyze.") # Should not happen if button clicked with text, but good practice

    else:
        result_placeholder.warning("Please enter some text to analyze.")

# --- Footer ---
st.markdown("<div class='footer'>**Developed by Eeman Majumder**</div>", unsafe_allow_html=True)
