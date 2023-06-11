import torch
import json
import streamlit as st
from newsParser import getNewsTitleText

t5_tokenizer = None
bart_tokenizer = None
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
    T5_PATH = 't5-base'

    # model = T5ForConditionalGeneration.from_pretrained('t5-small')
    # tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # device = torch.device('cpu')

    t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
    t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH, model_max_length=1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except:
    print("Import T5 Error")

try:
    from transformers import BartForConditionalGeneration, BartTokenizer, BartModel
    BART_PATH = 'bart-large'

    # bart_model = BartForConditionalGeneration.from_pretrained(BART_PATH, output_past=True)
    # bart_tokenizer = BartTokenizer.from_pretrained(BART_PATH, model_max_length=1024, output_past=True)

    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', model_max_length=1024, output_past=True)
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', return_dict=True)

except:
    print("Import BART Error")


def t5TextSummerizer(text, num_words=200):
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    # print("original text preprocessed: \n", preprocess_text)
    tokenized_text = t5_tokenizer.encode(t5_prepared_Text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = t5_model.generate(tokenized_text,
                                    num_beams=14,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=int(num_words),
                                    early_stopping=True)
    output = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Summarized text: \n", output)
    return output


def bart_summarize(input_text, num_beams=14, num_words=200):
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = bart_model.generate(input_tokenized,
                                      num_beams=int(num_beams),
                                      no_repeat_ngram_size=3,
                                      length_penalty=2.0,
                                      min_length=30,
                                      max_length=int(num_words),
                                      early_stopping=True)
    output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
              summary_ids]
    return output[0]


def t5_summarize(input_text, num_beams=4, num_words=200):
    input_text = str(input_text).replace('\n', '')
    input_text = ' '.join(input_text.split())
    input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
    summary_task = torch.tensor([[21603, 10]]).to(device)
    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
    summary_ids = t5_model.generate(input_tokenized,
                                    num_beams=int(num_beams),
                                    no_repeat_ngram_size=3,
                                    length_penalty=2.0,
                                    min_length=30,
                                    max_length=int(num_words),
                                    early_stopping=True)
    output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]


st.title("Text Summarization")

option = st.selectbox("Select the option for Summarization: ", ["Text", 'URL'])

# print the selected topic
st.write("You have selected: ", option)

model = st.selectbox("Select the model for Summarization: ", ["T5", 'BART'])

if option == "Text":
    text = st.text_area(label="Input Text", value="", height=200)
    if st.button("Generate") and text != "":
        if model == "T5":
            summary = t5TextSummerizer(text)
            st.text_area(label="Summary", value=summary, height=200, key="text-t5")
        elif model == "BART":
            summary = bart_summarize(text)
            st.text_area(label="Summary", value=summary, height=200, key="text-bart")
        # summary = t5TextSummerizer(text)
        # st.text_area(label="Summary", value=summary, height=200)
elif option == "URL":
    url = st.text_input("Input URL", value="")
    print("url =", url)
    if st.button("Generate") and url != "":
        title, text = getNewsTitleText(url)
        if model == "T5":
            summary = t5TextSummerizer(text)
            st.text_area(label="Summary", value=summary, height=200, key="url-t5")
        elif model == "BART":
            summary = bart_summarize(text)
            st.text_area(label="Summary", value=summary, height=200, key="url-bart")

url = "http://timesofindia.indiatimes.com/world/china/chinese-expert-warns-of-troops-entering-kashmir/articleshow/59516912.cms"
