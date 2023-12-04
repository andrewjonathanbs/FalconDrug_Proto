import streamlit as st
from streamlit_mic_recorder import mic_recorder,speech_to_text
from gtts import gTTS
import langchain
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from translate import Translator
import os

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
translator = Translator(from_lang="ID",to_lang="EN")
translator_1 = Translator(from_lang='EN',to_lang='ID')
language = 'id'

state=st.session_state

if 'text_received' not in state:
    state.text_received=[]

st.title('Uji Coba Prototype FalconDrug')
st.write("Berikut adalah aplikasi prototype dari FalconDrug yang dapat digunakan untuk berbicara dengan AI terkait kefarmasian.")
st.write("Harap sabar menunggu respons karena aplikasi ini masih memiliki waktu loading yang cukup lama.")

st.image('og-image.jpg')

c1,c2=st.columns(2)
with c1:
    st.write("Silahkan berbicara denganku!")
with c2:
    text=speech_to_text(language='id',use_container_width=True,just_once=True,key='STT')

if text:       
    state.text_received.append(text)
    for text in state.text_received:
        question = translator.translate(text)
        answer = llm_chain.run(question)
        trans_answer = translator_1.translate(answer)
        obj = gTTS(text=trans_answer, lang=language, slow=False)
        obj.save('trans.mp3')
        audio_file = open('trans.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg',start_time=0)