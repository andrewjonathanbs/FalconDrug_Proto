import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text
from gtts import gTTS
import langchain
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
from translate import Translator
from polyglot.text import Text, Word
language = "id"
translator1= Translator(to_lang="id")
translator2 = Translator(to_lang="en")

import requests

API_URL2 = "https://api-inference.huggingface.co/models/google/madlad400-10b-mt"
headers = {"Authorization": "Bearer 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"}

def query2(payload):
	response = requests.post(API_URL2, headers=headers, json=payload+'translate this to Indonesian language')
	return response.json()

HUGGINGFACEHUB_API_TOKEN = os.getenv('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')
repo_2 = "google/madlad400-10b-mt"
llm2 = HuggingFaceHub(huggingfacehub_api_token='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',
                     repo_id=repo_2,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})

template2 = """Question: {question}
Translate this into Indonesian language."""

prompt2 = PromptTemplate(template=template2, input_variables=["question"])
llm_chain2 = LLMChain(prompt=prompt2, llm=llm2, verbose=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv('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')
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token='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',
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})

template = """Question: {question}
Answer the question in Indonesian language based on this information:

You're an INDONESIAN AI that is created to give INDONESIAN people proper health and medicine information in INDONESIAN LANGUAGE."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

state=st.session_state

if 'text_received' not in state:
    state.text_received=[]

st.title('Falcon Drug Prototype')
st.write("This is the prototype of FalconDrug, a LLM created for telepharmacy services.")
st.write("Although the product is functional, it still needs more data to be a high quality telepharmacy model.")

st.image('og-image.jpg')

st.write("Please go easy and wait patiently for the LLM to load if it took a long time to load.")
st.write("Usually, it's fast but sometimes it load the responses slowly.")
st.write("The AI can understand both English and Indonesian, but answers in Indonesian.")

c1,c2=st.columns(2)
with c1:
    st.write("Feel free to ask me something about drugs.")
with c2:
    text=speech_to_text(language='id',use_container_width=True,just_once=True,key='STT')

if text:       
    state.text_received.append(text)
    for text in state.text_received:
        answer = llm_chain.run(text)
        answer = llm_chain2.run(answer)
        obj = gTTS(text=answer, lang=language, slow=False)
        obj.save('trans.mp3')
        audio_file = open('trans.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg',start_time=0)
