import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text
from gtts import gTTS
import langchain
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
from translate import Translator
language = "id"
translator1= Translator(to_lang="id")
translator2 = Translator(to_lang="en")

HUGGINGFACEHUB_API_TOKEN = os.getenv('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')
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token='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',
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})

template = """Question: {question}
Answer based on this context below.

You are a chatbot designed to perform telepharmacy services by answering questions about drug information. You will get input in Indonesian language and you have to reply in Indonesian language.
There's a possibilty for you to get input in english language, if so you can reply in english or Indonesia, depending on the user's preference.
"""

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
st.write("The AI can understand both English and Indonesian, but it responds with Indonesian language.")

c1,c2=st.columns(2)
with c1:
    st.write("Feel free to ask me something about drugs.")
with c2:
    text=speech_to_text(language='id',use_container_width=True,just_once=True,key='STT')

if text:       
    state.text_received.append(text)
    for text in state.text_received:
        question = translator1.translate(text)
        answer = llm_chain.run(question)
        trans_answer = translator2.translate(answer)
        obj = gTTS(text=trans_answer, lang=language, slow=False)
        obj.save('trans.mp3')
        audio_file = open('trans.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg',start_time=0)
