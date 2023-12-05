import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text
from gtts import gTTS
import langchain
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
import speech_recognition as sr
import translators as ts

HUGGINGFACEHUB_API_TOKEN = os.getenv('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')
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token='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',
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})

template = """Question: {question}
Answer the question based on this information:

You're a chatbot that is created to give people proper health and medicine information."""

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
	    translation = ts.google(text, from_language = 'id', to_language = 'en')
	    answer = llm_chain.run(translation)
	    trans_answer = ts.google(answer, from_language = 'en', to_language = 'id')
	    obj = gTTS(trans_answer, lang='id', slow=False)
	    obj.save('trans.mp3')
	    audio_file = open('trans.mp3', 'rb')
	    audio_bytes = audio_file.read()
	    st.audio(audio_bytes, format='audio/ogg',start_time=0)
