import streamlit as st 
import joblib,os
import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#from summarizer import Summarizer
import folium
import numpy as np
import nltk
from string import digits
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords #To Remove the StopWords like "the","in" ect
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pyttsx3
# initialisation 
engine = pyttsx3.init() 
from gtts import gTTS
from comtypes.client import CreateObject
engine = CreateObject("SAPI.SpVoice")
stream = CreateObject("SAPI.SpFileStream")


#st.text_input("Password:", value="", type="password")
#st.markdown(html,unsafe_allow_html=True,style = "background-color:Black")
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}.header{padding: 10px 16px; background: #111; color: #fff; position:fixed;top:0;text-align: justify;} .sticky { position: fixed; top: 0; width: 100%;}</style>', unsafe_allow_html=True)
#st.write('<style>body { margin: 0; font-family: font-family: Arial bold;font-size:25px, Arial bold, Arial bold;font-size: 30px;text-align: justify;} .header{padding: 10px 16px; background: #111; color: #fff; position:fixed;top:0;text-align: justify;} .sticky { position: fixed; top: 0; width: 100%;} </style><div class="header" id="myHeader">'+str(choice)+'</div>', unsafe_allow_html=True)
#st.write('<style>div.sidebar.row-sidebar.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


st.markdown("""
<style>
body {

    color: #111;
    background-color: #eaf4ff  ;
       
    
    etc. 
}
</style>

    
    """, unsafe_allow_html=True)

    # #;

    #111 for black background and fff for white letters
from spacy import displacy
HTML_WRAPPER = """div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
#HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

#summary pkgs
from gensim.summarization import summarize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

#Bi-grams
def ngrams(docx,n):
    #text = " ".join(docx)
    text1 = docx.lower()
    text2 = re.sub(r'[^a-zA-Z]'," ",text1)
    text3 = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(text2) if word not in stopwords.words("english") and len(word) > 2])
    words = nltk.word_tokenize(text3)
    ngram = list(nltk.ngrams(words,n))
    return ngram    


    
#    WC_height = 1000
#    WC_width = 1500
#    WC_max_words = 200
    
    


#Bert
#def bert_summary(docx):
#    model = Summarizer()
#    result = model(docx, min_length=60)
#    full = ''.join(result)
#    return full

# Reading Time
#@st.cache
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime


#NLP
#@st.cache(allow_output_mutation=True)
#@st.cache
def analyze_text(text):
    return nlp(text)

#webscrapping pkgs
from bs4 import BeautifulSoup
from urllib.request import urlopen

@st.cache
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
    return fetched_text

#def talkToMe(audio, lgg = 'en'):
    #print(audio)
#    tts = gTTS(text = audio, lang = lgg)
#    tts.save('audio.mp3') 
#    return None


def main():
    #st.sidebar.title("About")
    if st.sidebar.button("About this app"):
        st.sidebar.info(
                "This is an auto summarizer app for text articles, extracting the most important sentences by using NLP algorithms. It helps us to save time in our busy schedules who prefer to read the summary of those articles before we decide to jump in for reading entire article."
                )
    
    #st.write('<style>body { margin: 0; font-family: font-family: Tangerine;font-size:48px, Helvetica, sans-serif;font-size: 30px;text-align: center;} .header{padding: 10px 16px; background: #eaf4ff; color: #111; position:fixed;top:0;text-align: center;} .sticky { position: center; top: 0; width: 100%;} </style><div class="header" id="myHeader">'+str('RESUNER')+'</div>', unsafe_allow_html=True)
    st.write('<style>body { margin: 0; font-family: font-family: Tangerine;font-size:48px, Helvetica, sans-serif;font-size: 30px;text-align: justify;} .header{padding: 10px 16px; background: #eaf4ff; color: #111; position:fixed;top:0;text-align: center;} .sticky { position: fixed; top: 0; width: 100%;} </style><div class="header" id="myHeader">'+str('Summary Generator and Entity Recognizer')+'</div>', unsafe_allow_html=True)
    #st.title("Summary Generator and Entity checker")
    activities = ["Summarize","Summarize for URL","NER Checker","NER for URL"]
    choice = st.radio("Select Activity",activities)
    if choice == 'Summarize':
        st.info(
                "Please paste your text into the left side box & click the 'Summarize!' to view the summary"
                )
        st.sidebar.subheader("Summarization")
        raw_text = st.sidebar.text_area("Enter Text Here")
        #summary_choice = st.selectbox("Summary Choice",["Gensim","Sumy Lex Rank"])
        if st.sidebar.button("Summarize!"):
            summary_result = sumy_summarizer(raw_text)
            estimatedTime_org = readingTime(raw_text)
            #text_length = st.slider("Length to Preview",50,100)
            st.info("Original Reading time - {} mins".format(estimatedTime_org))

            st.write(summary_result)
            estimatedTime_res = readingTime(summary_result)
            st.info("Summary Reading time - {} mins".format(estimatedTime_res))
            
            engine = pyttsx3.init(driverName='sapi5')
            #infile = "tanjil.txt"
           # f = open(infile, 'r')
            #theText = f.read()
            #f.close()

            #Saving part starts from here 
            tts = gTTS(text=summary_result, lang='en')
            #saved_file=talkToMe(summary_result , lgg ='en')  
            tts.save("saved_file.mp3")
            audio_file = open('saved_file.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3',start_time=0)
        st.sidebar.subheader("Visualizations")
        visualize = ["Select","WordCloud","Bigrams","Trigrams"]
        choice2 = st.sidebar.selectbox("Visualize",visualize)
        #if choice2 == "Only Summary":
            
        if choice2 == "WordCloud":
            c_text = raw_text
            #plt.figure(figsize=[70,50])
            maskArray = np.array(Image.open("C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/wordcloudsummy/mask/comment.png"))
            wordcloud = WordCloud(max_font_size=200,max_words=3000, margin=10, background_color='white', mask = maskArray, contour_width=3,contour_color='black',
                                  scale=3, relative_scaling = 0.5, width=1900, height=1900,random_state=1).generate(c_text)
            plt.imshow(wordcloud,interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
            
        if choice2 == "Bigrams":
            c_text=ngrams((raw_text),2)
            for i in range(0,len(c_text)):
                c_text[i] = " ".join(c_text[i])
            Bigram_Freq = nltk.FreqDist(c_text)
            maskArray = np.array(Image.open("C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/wordcloudsummy/mask/comment.png"))

            #bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
            #plt.figure(figsize = (50,25))
            bigram_wordcloud = WordCloud(max_font_size=150,max_words=2000, margin=10, background_color='white', mask = maskArray, contour_width=3,contour_color='black',
                                         scale=3, relative_scaling = 0.5, width=900, height=900,random_state=1).generate_from_frequencies(Bigram_Freq)
            #plt.figure(figsize = (50,25))
            plt.imshow(bigram_wordcloud,interpolation = 'bilinear')
            plt.axis("off")
#            maskArray = np.array(Image.open("C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/wordcloudsummy/cloud2.png"))
    #wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,stopwords=stopwords_wc, background_color='white', mask = maskArray).generate_from_frequencies(dict(words_freq))
#            wordCloud = WordCloud(max_font_size=150,max_words=2000, margin=10, background_color='white', mask = maskArray,
#                                  scale=3, relative_scaling = 0.5, width=900, height=900,random_state=1).generate_from_frequencies(c_text)
#            plt.title('Most frequently occurring bigrams connected by same colour and font size')
#            plt.imshow(wordCloud, interpolation='bilinear')
#            plt.axis("off")
            #return st.pyplot()
            st.pyplot()
            
        if choice2 == "Trigrams":
            c_text=ngrams((raw_text),3)
            for i in range(0,len(c_text)):
                c_text[i] = " ".join(c_text[i])
            trigram_Freq = nltk.FreqDist(c_text)
            maskArray = np.array(Image.open("C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/wordcloudsummy/mask/comment.png"))
            #bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
            #plt.figure(figsize = (50,25))
            trigram_wordcloud = WordCloud(max_font_size=150,max_words=200, margin=10, background_color='white', mask = maskArray, contour_width=3,contour_color='black',
                                          scale=3, relative_scaling = 0.5, width=900, height=900,random_state=1).generate_from_frequencies(trigram_Freq)
            #plt.figure(figsize = (50,25))
            plt.imshow(trigram_wordcloud,interpolation = 'bilinear')
            plt.axis("off")
            st.pyplot()


    #st.write('<style>body { margin: 0; font-family: Arial, Helvetica, sans-serif;} .header{padding: 10px 16px; background: #7f78d2; color: #f1f1f1; position:fixed;top:0;} .sticky { position: fixed; top: 0; width: 100%;} </style><div class="header" id="myHeader">'+str('Summarator')+'</div>', unsafe_allow_html=True)            
            
    if choice == 'NER Checker':
        st.info(
            "About NER Checker: Named-entity recognition (NER) automatically identifies names of people, places, products & organizations. The entities displayed here is PERSON, NORP (nationalities, religious and political groups), FAC (buildings, airports etc.), ORG (organizations), GPE (countries, cities etc.), LOC (mountain ranges, water bodies etc.), PRODUCT (products), EVENT (event names), WORK_OF_ART (books, song titles), LAW (legal document titles), LANGUAGE (named languages), DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL and CARDINAL"
            )

        st.sidebar.subheader("Entity Recognition")
        raw_text = st.sidebar.text_area("Enter Text Here","Type Here")
        if st.sidebar.button("Analyze!"):
            # NLP
            docx = analyze_text(raw_text)
            html = displacy.render(docx,style='ent')
            html = html.replace("\n\n","\n")
            #st.write(html,unsafe_allow_html=True)
            st.markdown(html,unsafe_allow_html=True)
        
    if choice == 'NER for URL':
        st.info(
            "About NER Checker: Named-entity recognition (NER) automatically identifies names of people, places, products & organizations. The entities displayed here is PERSON, NORP (nationalities, religious and political groups), FAC (buildings, airports etc.), ORG (organizations), GPE (countries, cities etc.), LOC (mountain ranges, water bodies etc.), PRODUCT (products), EVENT (event names), WORK_OF_ART (books, song titles), LAW (legal document titles), LANGUAGE (named languages), DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL and CARDINAL"
            )

        st.sidebar.subheader("Analyze text from URL")
        raw_url = st.sidebar.text_input("Enter URL Here","Type here")
        #text_preview_length = st.slider("Length to Preview",50,100)
        if st.sidebar.button("Analyze"):
            if raw_url != "Type here":
                result = get_text(raw_url)
         #       len_of_full_text = len(result)
         #       len_of_short_text = round(len(result)/text_preview_length)
         #       st.success("Length of Full Text::{}".format(len_of_full_text))
         #       st.success("Length of Short Text::{}".format(len_of_short_text))
         #       st.info(result[:len_of_short_text])
                #summarized_docx = sumy_summarizer(result)
                docx = analyze_text(result)
                html = displacy.render(docx,style="ent")
                html = html.replace("\n\n","\n")
                #st.write(HTML_WRAPPER1.format(html),unsafe_allow_html=True)
                st.markdown(html,unsafe_allow_html=True)

                
    if choice == 'Summarize for URL':
        st.info(
                "Please paste your url into the left side box & click the 'Summarize!' to view the summary"
                )

        st.sidebar.subheader("Summary from URL")
        raw_url = st.sidebar.text_input("Enter URL","Type here")
        #text_length = st.sidebar.slider("Length to Preview",50,100)
#        text_length = st.slider("Length to Preview",50,100)
        if st.sidebar.button("Summarize!"):
            if raw_url != "Type here":
                result = get_text(raw_url)
                estimatedTime_org_url = readingTime(result)
            #text_length = st.slider("Length to Preview",50,100)
                #st.info("Original Reading time - {} mins".format(estimatedTime_org_url))

                #len_of_full_text = len(result)
                #len_of_short_text = round(len(result)/text_length)
                #st.info("Length::Full Text::{}".format(len_of_full_text))
                #st.info("Length::Short Text::{}".format(len_of_short_text))
                #st.write(result[:len_of_short_text])
                summary_result_url = sumy_summarizer(result)
                st.write(summary_result_url)
                estimatedTime_res_url = readingTime(summary_result_url)
                st.info("Summary Reading time - {} mins".format(estimatedTime_res_url))
                engine = pyttsx3.init(driverName='sapi5')
            #infile = "tanjil.txt"
           # f = open(infile, 'r')
            #theText = f.read()
            #f.close()

            #Saving part starts from here 
                tts = gTTS(text=summary_result_url, lang='en')
                #saved_file2=talkToMe(summary_result_url , lgg ='en')  
                tts.save("saved_file3.mp3")
                audio_file2 = open('saved_file3.mp3', 'rb')
                audio_bytes2 = audio_file2.read()
                st.audio(audio_bytes2, format='audio/mp3',start_time=0)
        st.sidebar.subheader("Visualizations")
        visualize = ["Select","WordCloud","Bigrams","Trigrams"]
        choice2 = st.sidebar.selectbox("Visualize",visualize)
        #if choice2 == "Only Summary":
            
        if choice2 == "WordCloud":
            if raw_url != "Type here":
                result = get_text(raw_url)
                c_text = result
                #plt.figure(figsize=[70,50])
                maskArray = np.array(Image.open("C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/wordcloudsummy/mask/comment.png"))
                wordcloud = WordCloud(max_font_size=200,max_words=3000, margin=10, background_color='white', mask = maskArray, contour_width=3,contour_color='black',
                                      scale=3, relative_scaling = 0.5, width=1900, height=1900,random_state=1).generate(c_text)
                plt.imshow(wordcloud,interpolation='bilinear')
                plt.axis("off")
                st.pyplot()
        
        if choice2 == "Bigrams":
            if raw_url != "Type here":
                result = get_text(raw_url)
                c_text=ngrams((result),2)
                for i in range(0,len(c_text)):
                    c_text[i] = " ".join(c_text[i])
                Bigram_Freq_u = nltk.FreqDist(c_text)
                maskArray = np.array(Image.open("C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/wordcloudsummy/mask/comment.png"))
                    
                    #bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
                    #plt.figure(figsize = (50,25))
                bigram_wordcloud_u = WordCloud(max_font_size=150,max_words=2000, margin=10, background_color='white', mask = maskArray, contour_width=3,contour_color='steelblue',
                                               scale=3, relative_scaling = 0.5, width=900, height=900,random_state=1).generate_from_frequencies(Bigram_Freq_u)
                    #plt.figure(figsize = (50,25))
                plt.imshow(bigram_wordcloud_u,interpolation = 'bilinear')
                plt.axis("off")
                st.pyplot()
            
        if choice2 == "Trigrams":
            if raw_url != "Type here":
                result = get_text(raw_url)
                c_text=ngrams((result),3)
                for i in range(0,len(c_text)):
                    c_text[i] = " ".join(c_text[i])
                trigram_Freq_u = nltk.FreqDist(c_text)
                maskArray = np.array(Image.open("C:/Users/NAKKANA1/OneDrive - Novartis Pharma AG/Desktop/aws_study/streamlit/wordcloudsummy/mask/comment.png"))
                
            #bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
            #plt.figure(figsize = (50,25))
                trigram_wordcloud_u = WordCloud(max_font_size=150,max_words=200, margin=10, background_color='white', mask = maskArray, contour_width=3,contour_color='black',
                                              scale=3, relative_scaling = 0.5, width=900, height=900,random_state=1).generate_from_frequencies(trigram_Freq_u)
            #plt.figure(figsize = (50,25))
                plt.imshow(trigram_wordcloud_u,interpolation = 'bilinear')
                plt.axis("off")
                st.pyplot()


                
                
    
    st.sidebar.title("")
    st.sidebar.info(
        "Connect: naga_mayuri.nakka@novartis.com"
        )
    

if __name__ == '__main__':
    main()
    
