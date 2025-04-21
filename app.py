import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
model1_path = "modernbert.bin"
model2_path = "https://huggingface.co/mihalykiss/modernbert_2/resolve/main/Model_groups_3class_seed12"
model3_path = "https://huggingface.co/mihalykiss/modernbert_2/resolve/main/Model_groups_3class_seed22"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

model_1 = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=41)
model_1.load_state_dict(torch.load(model1_path, map_location=device))
model_1.to(device).eval()

model_2 = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=41)
model_2.load_state_dict(torch.hub.load_state_dict_from_url(model2_path, map_location=device))
model_2.to(device).eval()

model_3 = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=41)
model_3.load_state_dict(torch.hub.load_state_dict_from_url(model3_path, map_location=device))
model_3.to(device).eval()


label_mapping = {
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

def clean_text(text):

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
 
    text = re.sub(r"\n\s*\n+", "\n\n", text)  
    
    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  

    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  

    text = text.strip()
    
    return text

def classify_text(text):
    cleaned_text = clean_text(text)
    if not text.strip():
        result_message = (
            f"---- \n"
        )
        return result_message

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits_1 = model_1(**inputs).logits
        logits_2 = model_2(**inputs).logits
        logits_3 = model_3(**inputs).logits

        softmax_1 = torch.softmax(logits_1, dim=1)
        softmax_2 = torch.softmax(logits_2, dim=1)
        softmax_3 = torch.softmax(logits_3, dim=1)

        averaged_probabilities = (softmax_1 + softmax_2 + softmax_3) / 3
        probabilities = averaged_probabilities[0]

    ai_probs = probabilities.clone()
    ai_probs[24] = 0
    ai_total_prob = ai_probs.sum().item() * 100
    human_prob = 100 - ai_total_prob

    ai_argmax_index = torch.argmax(ai_probs).item()
    ai_argmax_model = label_mapping[ai_argmax_index]

    if human_prob > ai_total_prob:
        result_message = (
            f"**The text is** <span class='highlight-human'>**{human_prob:.2f}%** likely <b>Human written</b>.</span>"
        )
    else:
        result_message = (
            f"**The text is** <span class='highlight-ai'>**{ai_total_prob:.2f}%** likely <b>AI generated</b>.</span>\n\n"
            f"**Identified AI Model: {ai_argmax_model}**"
        )

    return result_message





title = "AI Text Detector"

description = """"""
bottom_text = "**Developed by Eeman Majumder**"


# AI_texts = [
# "Camels are remarkable desert animals known for their unique adaptations to harsh, arid environments. Native to the Middle East, North Africa, and parts of Asia, camels have been essential to human life for centuries, serving as a mode of transportation, a source of food, and even a symbol of endurance and survival. There are two primary species of camels: the dromedary camel, which has a single hump and is commonly found in the Middle East and North Africa, and the Bactrian camel, which has two humps and is native to Central Asia. Their humps store fat, not water, as commonly believed, allowing them to survive long periods without food by metabolizing the stored fat for energy. Camels are highly adapted to desert life. They can go for weeks without water, and when they do drink, they can consume up to 40 gallons in one sitting. Their thick eyelashes, sealable nostrils, and wide, padded feet protect them from sand and help them walk easily on loose desert terrain.",
# "Wines are a fascinating reflection of culture, history, and craftsmanship. They embody a rich diversity shaped by the land, climate, and traditions where they are produced. From the bold reds of Bordeaux to the crisp whites of New Zealand, each bottle tells a unique story. What makes wine so special is its ability to connect people. Whether shared at a family dinner, a celebratory event, or a quiet evening with friends, wine enhances experiences and brings people together. The variety of flavors and aromas, influenced by grape type, fermentation techniques, and aging processes, make wine tasting a complex yet rewarding journey for the senses.",
# "I find artificial intelligence (AI) to be one of the most transformative and fascinating technologies of our time. Its potential spans a wide range of applications, from automating mundane tasks to revolutionizing industries like healthcare, education, and entertainment. AI has already made significant contributions in fields like language processing, image recognition, and decision-making systems, enabling innovations that were once purely science fiction. However, as powerful as AI can be, it also brings challenges and responsibilities. Ethical considerations, such as bias in data, transparency, and the potential for misuse, need to be carefully addressed to ensure fairness and accountability. The rise of generative AI has also sparked debates about creativity, originality, and intellectual property, making it essential to strike a balance between technological advancement and respecting human contributions."
# ]

# Human_texts = [
# "The present book is intended as a text in basic mathematics. As such, it can have multiple use: for a one-year course in the high schools during the third or fourth year (if possible the third, so that calculus can be taken during the fourth year); for a complementary reference in earlier high school grades (elementary algebra and geometry are covered); for a one-semester course at the college level, to review or to get a firm foundation in the basic mathematics necessary to go ahead in calculus, linear algebra, or other topics. Years ago, the colleges used to give courses in “ college algebra” and other subjects which should have been covered in high school. More recently, such courses have been thought unnecessary, but some experiences I have had show that they are just as necessary as ever. What is happening is that thecolleges are getting a wide variety of students from high schools, ranging from exceedingly well-prepared ones who have had a good first course in calculus, down to very poorly prepared ones.",
# "Fats are rich in energy, build body cells, support brain development of infants, help body processes, and facilitate the absorption and use of fat-soluble vitamins A, D, E, and K. The major component of lipids is glycerol and fatty acids. According to chemical properties, fatty acids can be divided into saturated and unsaturated fatty acids. Generally lipids containing saturated fatty acids are solid at room temperature and include animal fats (butter, lard, tallow, ghee) and tropical oils (palm,coconut, palm kernel). Saturated fats increase the risk of heart disease.",
# "To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., h Question, Answeri) in one token sequence. Throughout this work, a “sentence” can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A “sequence” refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together. We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary. The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence."]
iface = gr.Blocks(css="""
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    #text_input_box {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        font-size: 18px;
        padding: 15px;
        margin-bottom: 20px;
        width: 60%;
        box-sizing: border-box;
        margin: auto;
    }
    .form.svelte-633qhp {
        background: none;
        border: none;
        box-shadow: none;
    }

    #result_output_box {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        font-size: 18px;
        padding: 15px;
        margin-top: 20px;
        width: 40%;
        box-sizing: border-box;
        text-align: center;
        margin: auto;
    }

    @media (max-width: 768px) {
        #result_output_box {
            width: 100%;
        }
	#text_input_box{
	    width: 100%;
	}
    }

    body {
        font-family: 'Roboto Mono', sans-serif !important;
        padding: 20px;
        display: block;
        justify-content: center;
        align-items: center;
        height: 100vh;
        overflow-y: auto;
    }

    .gradio-container {
        border: 1px solid #4CAF50;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0px 0px 10px rgba(0,255,0,0.6);
        max-width: 600px;
        margin: auto;
	overflow-y: auto;
    }

    h1 {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 30px;
    }

    .highlight-human {
        color: #4CAF50;
        font-weight: bold;
        background: rgba(76, 175, 80, 0.2);
        padding: 5px;
        border-radius: 8px;
    }

    .highlight-ai {
        color: #FF5733;
        font-weight: bold;
        background: rgba(255, 87, 51, 0.2);
        padding: 5px;
        border-radius: 8px;
    }
    #bottom_text {
        text-align: center;
        margin-top: 50px;
        font-weight: bold;
        font-size: 20px;
    }
    .block.svelte-11xb1hd{
	background: none !important;
    }
""")

with iface:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    text_input = gr.Textbox(label="", placeholder="Type or paste your content here...", elem_id="text_input_box", lines=5)
    result_output = gr.Markdown("", elem_id="result_output_box")
    text_input.change(classify_text, inputs=text_input, outputs=result_output)
    # with gr.Tab("AI text examples"):
    #     gr.Examples(AI_texts, inputs=text_input)
    # with gr.Tab("Human text examples"):
    #     gr.Examples(Human_texts, inputs=text_input)
    gr.Markdown(bottom_text, elem_id="bottom_text")

iface.launch(share=True)

