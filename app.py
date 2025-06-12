# paraphrase_app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import T5Tokenizer

# Login (optional - only if using private models or rate limit bypass)
# from huggingface_hub import login
# login("your_token_here")

st.set_page_config(page_title="Paraphrasing Tool", layout="centered")
st.title("📝 Paraphrasing Tool using Hugging Face")

# Load model and tokenizer
@st.cache_resource
def load_pipeline():
    model_name = "Vamsi/T5_Paraphrase_Paws"
    tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

paraphrase = load_pipeline()

# Text input
input_text = st.text_area("Enter a sentence to paraphrase:", height=150)

# Paraphrase button
if st.button("Paraphrase"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        input_text_prepared = f"paraphrase: {input_text} </s>"
        outputs = paraphrase(input_text_prepared, max_length=100, num_beams=5, num_return_sequences=3, temperature=1.5)
        
        st.subheader("🔁 Paraphrased Outputs:")
        for i, out in enumerate(outputs, 1):
            st.markdown(f"**{i}.** {out['generated_text']}")