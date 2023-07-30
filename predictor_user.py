import streamlit as st

import pickle
import time
import pandas as pd
import numpy as np

import preprocess_predictor_user as prep

with open('data/rfc_new.pkl', 'rb') as file:
    rfc, tf_idf_vector = pickle.load(file)
    
def predictor_page():
    st.title("Sentiment Predictor", help="Aplikasi prediksi positif atau negatif pada suatu sentiment.")
    st.write("")
    
    with st.expander("", expanded=True):
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 13px; font-weight: bold;
             color: white; background-color: #00386b; 
             border-radius: 5px; padding: 9px 22px;">
             Input Sentiment :</span>
         ''', unsafe_allow_html = True)
        sample_review = st.text_area("Input Sentiment:", height=200, placeholder='''
        "Sangat membantu dalam pembelian tiket, namun masih harus diperbaiki kecepatannya."
        ''', label_visibility="collapsed")    
    st.write("")
    
    if st.button("Prediksi"):
        if sample_review:
            st.write("")
            
            with st.spinner('Memprediksi...'):
                st.empty()
                time.sleep(1)

            with st.chat_message("assistant"):
                st.write("Berdasarkan pola yang saya pelajari, sentiment dianggap:")
                

                # memproses sample_review
                sample_review1 = prep.casefolding(sample_review)
                sample_review2 = prep.cleansing(sample_review1)    
                sample_review3 = prep.stemming(sample_review2)   
                sample_review4 = prep.convert_slangword(sample_review3)
                sample_review5 = prep.remove_stopword(sample_review4)
                sample_review6 = prep.remove_unwanted_words(sample_review5)
                sample_review7 = prep.remove_short_words(sample_review6)
                sample_review8 = prep.tokenizing(sample_review7)
                # st.write(sample_review8)
                # temp = tf_idf_vector.transform(sample_review8)
                # st.write(temp)
                # predict = rfc.predict(temp)
                
                if not sample_review8:
                    # Hasil Positive karena tidak terdapat kata pada sample
                    sentiment_analysis = "&nbsp;&nbsp;&nbsp;&nbsp; Positive"
                    st.success(f'{sentiment_analysis}', icon="😄")
                else:
                    # Transformasi TF-IDF dan Prediksi RFC
                    temp = tf_idf_vector.transform(sample_review8)
                    # st.write(temp)
                    predict = rfc.predict(temp)

                    # menghitung jumlah kemunculan kata yang diprediksi
                    jumlah_negative = np.count_nonzero(predict == 'Negative')
                    jumlah_positive = np.count_nonzero(predict == 'Positive')

                    # st.write("Negative :", jumlah_negative, " // ", "Positive :", jumlah_positive)

                    # Negative Result
                    if jumlah_negative >= jumlah_positive:
                        sentiment_analysis = "&nbsp;&nbsp;&nbsp;&nbsp; Negative"
                        st.warning(f'{sentiment_analysis}', icon="😟")

                    # Positive Result
                    else:
                        sentiment_analysis = "&nbsp;&nbsp;&nbsp;&nbsp; Positive"
                        st.success(f'{sentiment_analysis}', icon="😄")
                    
        # Text Area Kosong
        else:
            st.write("")
            with st.error("Maaf, input sentiment dahulu."):
                time.sleep(2)
                st.empty()
            
hide_streamlit = """ <style> footer {visibility: hidden;} </style> """    
st.markdown(hide_streamlit, unsafe_allow_html=True)            