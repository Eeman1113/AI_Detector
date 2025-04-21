import streamlit as st
import torch
import torch.hub
import re
import os

# --- Set Page Config First ---
st.set_page_config(page_title="AI Text Detector", layout="centered")

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

# --- Inject Custom CSS for highlighting ---
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

DEVICE = get_device()

# Now, we can safely continue with the rest of the code

# --- Model and Tokenizer Loading (Cached) ---
@st.cache_resource
def load_tokenizer(model_name):
    """Loads the tokenizer."""
    st.info(f"Loading tokenizer: {model_name}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    st.info("Tokenizer loaded.")
    return tokenizer

@st.cache_resource
def load_model(model_path_or_url, base_model, num_labels, is_url=False, _device=DEVICE):
    """Loads a sequence classification model from local path or URL."""
    from transformers import AutoModelForSequenceClassification
    model_name = os.path.basename(model_path_or_url) if not is_url else model_path_or_url.split('/')[-1]
    st.info(f"Loading model structure: {base_model}...")
    # Load the base model architecture with the desired number of labels.
    # The classification head will be randomly initialized initially.
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
    st.info(f"Loading model weights: {model_name}...")
    try:
        if is_url:
            # Load state dict from URL (usually safer as HF handles download/caching)
            state_dict = torch.hub.load_state_dict_from_url(model_path_or_url, map_location=_device, progress=True)
        else:
            # Load state dict from local file
            if not os.path.exists(model_path_or_url):
                 st.error(f"Model file not found at {model_path_or_url}. Please ensure it's in the correct location.")
                 st.stop() # Stop execution if local model is missing

            # --- FIX APPLIED HERE ---
            # Load state dict from local path.
            # Set weights_only=False because the .bin file likely contains more than just weights
            # and PyTorch 2.6+ defaults to weights_only=True for security.
            # WARNING: Only use weights_only=False if you TRUST the source of the .bin file,
            # as it can execute arbitrary code.
            st.warning(f"Loading '{model_name}' with weights_only=False. Ensure this file is from a trusted source.")
            state_dict = torch.load(model_path_or_url, map_location=_device, weights_only=False)
            # --- END FIX ---

        # Load the state dictionary into the model structure.
        # This should overwrite the randomly initialized classification head
        # if the state_dict contains the trained classifier weights.
        # The warning "Some weights were not initialized..." might still appear
        # but is often ignorable if loading succeeds without key errors.
        model.load_state_dict(state_dict)
        model.to(_device).eval() # Set model to evaluation mode
        st.info(f"Model {model_name} loaded and moved to {_device}.")
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        # Display the full traceback for debugging if needed
        # import traceback
        # st.error(traceback.format_exc())
        st.stop() # Stop execution on model loading error

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
    if not isinstance(text, str): # Basic type check
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Improved handling for hyphenated words broken by newline: handles potential space after hyphen
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text) # Replace single newlines with spaces
    text = text.strip()
    return text

def classify_text(text, tokenizer, model_1, model_2, model_3, device, label_mapping, human_label_index):
    """Classifies the text using the ensemble of models."""
    # Ensure models are loaded before proceeding
    if not all([model_1, model_2, model_3, tokenizer]):
         st.error("One or more models/tokenizer failed to load. Cannot classify.")
         return {"error": True, "message": "Model loading failed."}

    cleaned_text = clean_text(text)
    if not cleaned_text: # Check after cleaning
        # Don't show a warning here, just return None or an indicator for no text
        # st.warning("Please enter some text to analyze.")
        return None # Indicate no classification needed for empty/whitespace text

    try:
        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True, # Pad to max_length or model max length
            max_length=tokenizer.model_max_length # Ensure consistent length
        ).to(device)

        with torch.no_grad():
            logits_1 = model_1(**inputs).logits
            logits_2 = model_2(**inputs).logits
            logits_3 = model_3(**inputs).logits

            softmax_1 = torch.softmax(logits_1, dim=1)
            softmax_2 = torch.softmax(logits_2, dim=1)
            softmax_3 = torch.softmax(logits_3, dim=1)

            # Ensemble by averaging probabilities
            averaged_probabilities = (softmax_1 + softmax_2 + softmax_3) / 3
            probabilities = averaged_probabilities[0].cpu() # Move to CPU for numpy/python processing

            # Ensure human_label_index is valid
            if not (0 <= human_label_index < len(probabilities)):
                 st.error(f"Internal Error: Invalid human_label_index ({human_label_index}) for probability tensor size ({len(probabilities)}).")
                 return {"error": True, "message": "Configuration error."}

            # Separate human vs AI probability
            human_prob = probabilities[human_label_index].item() * 100

            # Calculate AI probability (sum of all non-human labels)
            # Create a mask to exclude the human label index
            mask = torch.ones_like(probabilities, dtype=torch.bool)
            mask[human_label_index] = False
            ai_total_prob = probabilities[mask].sum().item() * 100

            # If total prob doesn't sum roughly to 100, something might be off, but proceed.
            # Note: Due to potential floating point inaccuracies or model quirks,
            # human_prob + ai_total_prob might not be *exactly* 100.

            # Find the most likely AI model among the non-human labels
            # Create a temporary tensor with human prob zeroed out to find AI max
            ai_probs_only = probabilities.clone()
            ai_probs_only[human_label_index] = -float('inf') # Set human prob to neg infinity to ensure it's not chosen as max AI
            ai_argmax_index = torch.argmax(ai_probs_only).item()
            ai_argmax_model = label_mapping.get(ai_argmax_index, f"Unknown AI (Index {ai_argmax_index})")

            # Determine final classification
            # Use a small tolerance for comparison if needed, but direct comparison is usually fine
            if human_prob >= ai_total_prob:
                return {"is_human": True, "probability": human_prob, "model": "Human"}
            else:
                # Return the total AI probability, but name the single most likely AI model
                return {"is_human": False, "probability": ai_total_prob, "model": ai_argmax_model}

    except Exception as e:
        st.error(f"Error during model inference: {e}")
        # import traceback
        # st.error(traceback.format_exc()) # Uncomment for detailed traceback during debugging
        return {"error": True, "message": f"Inference failed: {e}"}

# Main UI section
st.title("🕵️ AI Text Detector 🤖")

# Load models and tokenizer
TOKENIZER = load_tokenizer(BASE_MODEL)
MODEL_1 = load_model(MODEL1_PATH, BASE_MODEL, NUM_LABELS, is_url=False, _device=DEVICE)
MODEL_2 = load_model(MODEL2_URL, BASE_MODEL, NUM_LABELS, is_url=True, _device=DEVICE)
MODEL_3 = load_model(MODEL3_URL, BASE_MODEL, NUM_LABELS, is_url=True, _device=DEVICE)

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
    # Check if input_text is not None and not just whitespace AFTER stripping
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
        if classification_result is None:
             # This case handles empty/whitespace input after cleaning
             result_placeholder.warning("Please enter some text to analyze.")
        elif classification_result.get("error"):
            error_message = classification_result.get("message", "An unknown error occurred during analysis.")
            result_placeholder.error(f"Analysis Error: {error_message}")
        elif classification_result["is_human"]:
            prob = classification_result['probability']
            result_html = (
                f"<div class='result-box'>"
                f"<b>The text is</b> <span class='highlight-human'><b>{prob:.2f}%</b> likely <b>Human written</b>.</span>"
                f"</div>"
            )
            result_placeholder.markdown(result_html, unsafe_allow_html=True)
        else: # AI generated
            prob = classification_result['probability']
            model_name = classification_result['model']
            result_html = (
                f"<div class='result-box'>"
                f"<b>The text is</b> <span class='highlight-ai'><b>{prob:.2f}%</b> likely <b>AI generated</b>.</span><br><br>"
                f"<b>Most Likely AI Model: {model_name}</b>" # Changed wording slightly
                f"</div>"
            )
            result_placeholder.markdown(result_html, unsafe_allow_html=True)

    else: # Handles case where input_text is None or empty string before stripping
        result_placeholder.warning("Please enter some text to analyze.")

# --- Footer ---
st.markdown("<div class='footer'>**Developed by Eeman Majumder**</div>", unsafe_allow_html=True)