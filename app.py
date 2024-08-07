import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text
from gtts import gTTS
import langchain
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
import speech_recognition as sr
from translate import Translator

translator = Translator(from_lang="ID",to_lang="EN")
translator_1 = Translator(from_lang='EN',to_lang='ID')
language = 'id'

HUGGINGFACEHUB_API_TOKEN = os.getenv('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')
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
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

st.write("WARNING: In this prototype we're still using a free translation API, which function normally in a local setting...")
st.write("...but becomes extremely weird if used from Streamlit. So if you got weird responses, please try saying the question again.")
st.write("We apologize for the inconvenience.")

c1,c2=st.columns(2)
with c1:
    st.write("Feel free to ask me something about drugs.")
with c2:
    text=speech_to_text(language='id',use_container_width=True,just_once=True,key='STT')

if text:       
    state.text_received.append(text)
    for text in state.text_received:
	    translation = translator.translate(text)
	    answer = llm_chain.run(translation)
	    trans_answer = translator_1.translate(answer)
	    obj = gTTS(trans_answer, lang='id', slow=False)
	    obj.save('trans.mp3')
	    audio_file = open('trans.mp3', 'rb')
	    audio_bytes = audio_file.read()
	    st.audio(audio_bytes, format='audio/ogg',start_time=0)
