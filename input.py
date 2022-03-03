import streamlit as st
import base64
import chatbot
import summary
import nltk
import json
import time
import requests
from streamlit_lottie import st_lottie
nltk.download('popular', quiet=True)
from PIL import Image
import graph
import tts
import text_scrape
from question_generator import QAGen
import streamlit as st
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from scipy.io import wavfile
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import json
import numpy as np
import webbrowser

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack('bg5.png')

m = st.sidebar.markdown("""
<style>
div.stButton > button:first-child {
    width : 95% ;
    align: center;
    border-radius : 10px ;
    height: 100px ;
    
}
</style>
""", unsafe_allow_html=True)

m = st.markdown("""
<style>
    st.form.submit_button{
    
        background-color : blue ;
    
    }
}
</style>
""", unsafe_allow_html=True)
image = Image.open('logoapp.png')
st.sidebar.image(image, width=360)
st.sidebar.title('Selection Menu')
options = st.sidebar.selectbox("Choose one of the following:",("main tools","lessons summury","Grammar correction"))

html_temp = """
     <div style="color:black;font-weight: bold;text-align:center;font-family:verdana;font-size:500%;">
     <span style="color: blue">S'</span>Cool
     </div>
     </div>
     """
st.markdown(html_temp, unsafe_allow_html=True)

def get_options(options):
    st.sidebar.write("\n")
    if (options == "main tools"):
        tools()


    elif (options == "lessons summury"):
        lessons_summury()

    elif (options == "Grammar correction"):
        grammar()



def tools():

    st.header('Paragraph input')
    st.markdown('<br>', unsafe_allow_html=True)


    choice = st.radio('', ['Enter Text', 'Get from URL'], key = "31234")
    st.markdown('<br>', unsafe_allow_html=True)
    is_url = False

    if choice == 'Enter Text' :
        with st.form(key='my_form'):
            file = st.text_area('Enter Paragraph :      (Tip : Scroll down after pressing submit 😉) ', height = 350, key="278")
            submit_button = st.form_submit_button(label='Submit')

    else:
        with st.form(key='url_form'):
            st.text('Enter a URL or submit Blank for a demo url !')
            myurl = st.text_input('Enter URL', key="278")
            submit_button = st.form_submit_button(label='Submit Form')
            if myurl != "":
                file = text_scrape.get_scraped_text(myurl)
                is_url = True
            else:
                st.text('Using Demo URL as you entered Nothing')
                file = text_scrape.get_scraped_text('https://stribny.name/blog/2020/10/how-to-extract-plain-text-from-an-html-page-in-python/')

    if is_url == True :
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.header("Full Text (Extracted from webpage)")
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(file)

    if file != "":

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.header("Tools")
        st.markdown('<br>', unsafe_allow_html=True)

        st.subheader("Choose the tool you want to use : ")
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        choice = st.radio('', ['Text to Speech','Summary', 'Paragraph Q/A chatbot', 'Keyword Graph'], key = "312")
        st.markdown('<br>', unsafe_allow_html=True)
        if choice == 'Summary' :
            st.markdown('<hr>', unsafe_allow_html=True)
            st.subheader("Summary Length")
            st.markdown('<br>', unsafe_allow_html=True)
            slider = st.slider('',
                                 min_value=10,
                                 max_value=90,
                                 value=30,
                                 step=10,
                                 key="1234"
                                 )
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader("I want my summary to be " + str(slider) + " % of my paragraph length" )

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<hr>', unsafe_allow_html=True)
            st.header("Summary")
            st.markdown('<br>', unsafe_allow_html=True)
            summary.para_summary(file, slider)



        elif choice == 'Paragraph Q/A chatbot':
            st.markdown('<hr>', unsafe_allow_html=True)
            st.header("Chatbot")
            st.markdown('<br>', unsafe_allow_html=True)
            image = Image.open('chatbot.png')
            st.image(image, width=360)
            chatbot.para_bot(file)

        elif choice == "Text to Speech" :
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<hr>', unsafe_allow_html=True)
            st.header("Paragraph Audio")
            st.markdown('<br>', unsafe_allow_html=True)
            tts.tts(file)

        else:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.header("Keyword Graph")
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('Total keywords to plot :')
            st.markdown('<br>', unsafe_allow_html=True)


            graph.plot_graph(file)


def grammar():
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model_name = "flexudy/t5-small-wav2vec2-grammar-fixer"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    with st.form(key='form'):
        file = st.text_area('Enter Paragraph :      (Tip : Scroll down after pressing submit 😉) ', height=350,
                            key="278")
        submit_button = st.form_submit_button(label='Submit')

    model_name = 'deep-learning-analytics/GrammarCorrector'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    def correct_grammar(input_text, num_return_sequences):
        batch = tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(
            torch_device)
        translated = model.generate(**batch, max_length=64, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    st.write(correct_grammar(file, num_return_sequences=1))


def lessons_summury():
    st.title(" Resume lessons audio")

    my_range = list(range(1, 11))
    option = st.sidebar.select_slider("Choose number of Questions(Between 1 and 10):", options=my_range, value=10)

    r = sr.Recognizer()

    def get_large_audio_transcription(path):
        sound = AudioSegment.from_wav(path)
        chunks = split_on_silence(sound,
                                  min_silence_len=500,
                                  silence_thresh=sound.dBFS - 14,
                                  keep_silence=500,
                                  )
        folder_name = "audio-chunks"
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        whole_text = ""
        for i, audio_chunk in enumerate(chunks, start=1):
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            with sr.AudioFile(chunk_filename) as source:
                audio_listened = r.record(source)
                try:
                    text = r.recognize_google(audio_listened)
                except sr.UnknownValueError as e:
                    print("Error:", str(e))
                else:
                    text = f"{text.capitalize()}. "
                    whole_text += text
        with open('data1.txt', 'w+') as f:
            f.write(whole_text)

        return whole_text

    def summarymaker(text):
        st.markdown('<hr>', unsafe_allow_html=True)
        st.subheader("Summary Length")
        st.markdown('<br>', unsafe_allow_html=True)
        slider = st.slider('',
                           min_value=10,
                           max_value=90,
                           value=30,
                           step=10,
                           key="1234"
                           )
        st.markdown('<br>', unsafe_allow_html=True)
        st.subheader("I want my summary to be " + str(slider) + " % of my paragraph length")

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.header("Summary")
        st.markdown('<br>', unsafe_allow_html=True)
        summary.para_summary(text, slider)

    def Questions(summary):
        qg = QAGen(summary)
        return qg

    uploaded_file = st.file_uploader("Choose a wav file")
    st.audio(uploaded_file, format='wav')

    try:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        text = get_large_audio_transcription(uploaded_file.name)
    except:
        st.error("Make sure to upload WAV file")
        st.stop()

    st.write("Summary:")
    # summary Generator

    summary1 = summarymaker(text)
    st.write(summary1)
    # question generator

    st.write("Questions:")
    questions = Questions(text)
    st.write(questions)

get_options(options)
