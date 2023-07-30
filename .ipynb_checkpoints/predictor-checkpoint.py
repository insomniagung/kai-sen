import streamlit as st
session = st.session_state
# import gzip
import pickle
import time
import pandas as pd

import preprocess_predictor as prep

with open('rfc_new.pkl', 'rb') as file:
    rfc, tf_idf_vector = pickle.load(file)
    
def predict_sentiment(sample_review):
    if session['opt_casefolding'] == 'Ya':
        sample_review1 = prep.casefolding(sample_review)
        session['txtcasefolding'] = sample_review1
    else:
        sample_review1 = sample_review
    
    if session['opt_cleansing'] == 'Ya':
        sample_review2 = prep.cleansing(sample_review1)
        session['txtcleansing'] = sample_review2
    else:
        sample_review2 = sample_review1
    
    sample_review3 = sample_review2
        
    if session['opt_stemming'] == 'Ya':    
        sample_review4 = prep.stemming(sample_review3)
        session['txtstemming'] = sample_review4
    else:
        sample_review4 = sample_review3
    
    if session['opt_convert_slangword'] == 'Ya':    
        sample_review5 = prep.convert_slangword(sample_review4)
        session['txtconvert_slangword'] = sample_review5
    else:
        sample_review5 = sample_review4
    
    if session['opt_remove_stopword'] == 'Ya':    
        sample_review6 = prep.remove_stopword(sample_review5)
        session['txtremove_stopword'] = sample_review6
    else:
        sample_review6 = sample_review5
    
    if session['opt_remove_unwanted_words'] == 'Ya':    
        sample_review7 = prep.remove_unwanted_words(sample_review6)
        session['txtremove_unwanted_words'] = sample_review7
    else:
        sample_review7 = sample_review6
    
    if session['opt_remove_short_words'] == 'Ya':    
        sample_review8 = prep.remove_short_words(sample_review7)
        session['txtremove_short_words'] = sample_review8
    else:
        sample_review8 = sample_review7
    
    sample_review9 = prep.tokenizing(sample_review8)
    
    predictor = sample_review9
    return predictor

def predictor_page():
    st.title("Sentiment Predictor", help="Aplikasi prediksi positif atau negatif pada suatu sentiment.")
    st.write("")
    # st.divider()
    
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
        session['sample_review'] = sample_review
    
    #SIDEBAR
    with st.sidebar.expander("Opsi Preprocessing ", expanded=True):
        st.write("")
        # Casefolding
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 12px;
             color: white; background-color: #3b5998; 
             border-radius: 20px; padding: 7px 13px;">
             <b>Casefolding?</b></span>
         ''', unsafe_allow_html = True, help="_Proses mengkonverikan teks (mis: huruf é menjadi e, huruf E menjadi e)_")
        opt_casefolding = st.radio("Casefolding?", options=('Tidak', 'Ya'), index=1, horizontal=True, label_visibility="collapsed")
        if opt_casefolding:
            session['opt_casefolding'] = opt_casefolding
        else:
            session['opt_casefolding'] = opt_casefolding
        
        st.write("")
        # Cleansing
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 12px;
             color: white; background-color: #317256; 
             border-radius: 20px; padding: 7px 13px;">
             <b>Cleansing?</b></span>
         ''', unsafe_allow_html = True, help="_Proses pembersihan dokumen pada kata atau karakter_")
        opt_cleansing = st.radio("Cleansing?", options=('Tidak', 'Ya'), index=1, horizontal=True, label_visibility="collapsed")
        if opt_cleansing:
            session['opt_cleansing'] = opt_cleansing
        else:
            session['opt_cleansing'] = opt_cleansing

        st.write("")
        # Stemming
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 12px;
             color: white; background-color: #e0474c; 
             border-radius: 20px; padding: 7px 13px;">
             <b>Stemming?</b></span>
         ''', unsafe_allow_html = True, help="_Memotong/menghapus kata imbuhan menjadi kata dasar tanpa menggunakan akar dasar kamus bahasa_")
        opt_stemming = st.radio("Stemming?", options=('Tidak', 'Ya'), index=1, horizontal=True,label_visibility="collapsed")
        if opt_stemming:
            session['opt_stemming'] = opt_stemming
        else:
            session['opt_stemming'] = opt_stemming
        
        st.write("")
        # Convert Slangword
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 12px;
             color: black; background-color: #ffe303; 
             border-radius: 20px; padding: 7px 13px;">
             <b>Convert Slangword?</b></span>
         ''', unsafe_allow_html = True, help="_Mengubah kata non-formal menjadi kata formal_")
        opt_convert_slangword = st.radio("Convert Slangword?", options=('Tidak', 'Ya'), index=1, horizontal=True, label_visibility="collapsed")
        if opt_convert_slangword:
            session['opt_convert_slangword'] = opt_convert_slangword
        else:
            session['opt_convert_slangword'] = opt_convert_slangword
        
        st.write("")
        # Remove Stopword
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 12px;
             color: white; background-color: #8250e5; 
             border-radius: 20px; padding: 7px 13px;">
             <b>Remove Stopword?</b></span>
         ''', unsafe_allow_html = True, help="_Menghapus seluruh kata yang dianggap tidak memberikan kontribusi seperti kata hubung 'yang', 'di', 'dan', 'dari'_")
        opt_remove_stopword = st.radio("Remove Stopword?", options=('Tidak', 'Ya'), index=1, horizontal=True, label_visibility="collapsed")
        if opt_remove_stopword:
            session['opt_remove_stopword'] = opt_remove_stopword
        else:
            session['opt_remove_stopword'] = opt_remove_stopword
            
        st.write("")
        # Remove Unwanted Words
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 12px;
             color: white; background-color: #76a633; 
             border-radius: 20px; padding: 7px 13px;">
             <b>Remove Unwanted Words?</b></span>
         ''', unsafe_allow_html = True, help="_Membuat dictionary kata-kata yang kurang dianggap membawa hasil signifikan untuk analisis sentimen, dimana kata-kata yang sama dalam ulasan akan dihapus dari ulasan_")
        opt_remove_unwanted_words = st.radio("Remove Unwanted Words?", options=('Tidak', 'Ya'), index=1, horizontal=True,label_visibility="collapsed")
        if opt_remove_unwanted_words:
            session['opt_remove_unwanted_words'] = opt_remove_unwanted_words
        else:
            session['opt_remove_unwanted_words'] = opt_remove_unwanted_words
            
        st.write("")
        # Remove Short Words
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 12px;
             color: white; background-color: #046cd4; 
             border-radius: 20px; padding: 7px 13px;">
             <b>Remove Short Words?</b></span>
         ''', unsafe_allow_html = True, help="_Mempertahankan kata-kata lebih dari 2 karakter atau menghapus yang kurang dari 3 karakter_")
        opt_remove_short_words = st.radio("Remove Short Words?", options=('Tidak', 'Ya'), index=1, horizontal=True,label_visibility="collapsed")
        if opt_remove_short_words:
            session['opt_remove_short_words'] = opt_remove_short_words
        else:
            session['opt_remove_short_words'] = opt_remove_short_words
    
    with st.sidebar:
        st.write("")
        st.write("")
        st.write("")
        
    st.write("")
    if st.button("Prediksi"):
        
        if sample_review:
            
            st.write("")
            with st.spinner('Memprediksi...'):
                st.empty()
                time.sleep(0.3)
            
            with st.chat_message("user"):
                st.write("Sentiment:")
                st.write(f'''
                        <p style="text-decoration: none;
                        font-family: 'Open Sans'; font-size: 13px;
                        color: white; background-color: #00386b; 
                        border-radius: 5px; padding: 9px 22px;">
                        <b> {session['sample_review']} </b></p>
                    ''', unsafe_allow_html = True)

            with st.chat_message("assistant"):
                st.write("Berdasarkan pola yang saya pelajari, sentiment dianggap:")
                
                result_predict = predict_sentiment(sample_review)

                polar = {
                    "kata positive": [],
                    "kata negative": [],
                }
                polar_neutral = []
                
                for x in result_predict:
                    tf = tf_idf_vector.transform([x])
                    temp = rfc.predict(tf)
                    if (temp[0] ==  "negative"):
                        polar["kata negative"].append(x)
                    elif (temp[0] ==  "positive"):
                        polar["kata positive"].append(x)
                    else:
                        polar_neutral.append(x)

                positive_count = len(polar["kata positive"])
                negative_count = len(polar["kata negative"])
                
                # with st.expander("Hasil Polar"):
                #     st.write(polar_neutral)
                #     st.write(polar)
                
                # Negative Result
                if (negative_count > positive_count):                    
                    sentiment_analysis = "&nbsp;&nbsp;&nbsp;&nbsp; Negative"

                    # st.write("")
                    st.warning(f'{sentiment_analysis}', icon="😟")
                    
                    st.write(f"Jumlah: ({positive_count} kata positif, {negative_count} kata negatif)")
                    df_polar = pd.DataFrame.from_dict(polar, orient="index").transpose()
                    st.dataframe(df_polar.fillna(''), hide_index=True, use_container_width=True)
                    
                    st.write("")

                    if (opt_casefolding == 'Ya' or opt_cleansing =='Ya' or opt_stemming =='Ya' 
                        or opt_convert_slangword =='Ya' or opt_remove_stopword =='Ya' 
                        or opt_remove_unwanted_words =='Ya' or opt_remove_short_words =='Ya'
                        # or opt_tokenizing =='Ya'
                       ) :

                        with st.expander("Tahap Preprocessing", expanded=True):
                            if opt_casefolding == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #3b5998;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Casefolding : <b>{session['txtcasefolding']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_cleansing == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #317256;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Cleansing : <b>{session['txtcleansing']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_stemming == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #e0474c;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Stemming : <b>{session['txtstemming']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_convert_slangword == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: black; background-color: #ffe303;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Convert Slangword : <b>{session['txtconvert_slangword']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_remove_stopword == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #8250e5;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Remove Stopword : <b>{session['txtremove_stopword']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_remove_unwanted_words == 'Ya':    
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #76a633;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Remove Unwanted Words : <b>{session['txtremove_unwanted_words']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_remove_short_words == 'Ya':    
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #046cd4;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Remove Short Words : <b>{session['txtremove_short_words']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()


                # Positive Result
                else:
                    sentiment_analysis = "&nbsp;&nbsp;&nbsp;&nbsp; Positive"
                    st.write("")
                    st.success(f'{sentiment_analysis}', icon="😄")
                    
                    st.write(f"Jumlah: ({positive_count} kata positif, {negative_count} kata negatif)")
                    df_polar = pd.DataFrame.from_dict(polar, orient='index').transpose()
                    st.dataframe(df_polar.fillna(''), hide_index=True, use_container_width=True)
                    
                    st.write("")

                    if (opt_casefolding == 'Ya' or opt_cleansing =='Ya' or opt_stemming =='Ya' 
                        or opt_convert_slangword =='Ya' or opt_remove_stopword =='Ya' 
                        or opt_remove_unwanted_words =='Ya' or opt_remove_short_words =='Ya' 
                        # or opt_tokenizing =='Ya'
                       ) :

                        with st.expander("Tahap Preprocessing", expanded=True):
                            if opt_casefolding == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #3b5998;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Casefolding : <b>{session['txtcasefolding']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_cleansing == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #317256;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Cleansing : <b>{session['txtcleansing']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            # if opt_lemmatize == 'Ya':
                            #     st.write(f'''
                            #          <p style="text-decoration: none;
                            #          font-family: 'Open Sans'; font-size: 13px;
                            #          color: white; background-color: #7b7d7b;
                            #          border-radius: 5px; padding: 9px 22px;">
                            #          Lemmatize : <b>{session['txtlemmatize']}</b></p>
                            #      ''', unsafe_allow_html = True)
                            # else:
                            #     st.empty()

                            if opt_stemming == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #e0474c;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Stemming : <b>{session['txtstemming']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_convert_slangword == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: black; background-color: #ffe303;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Convert Slangword : <b>{session['txtconvert_slangword']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_remove_stopword == 'Ya':
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #8250e5;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Remove Stopword : <b>{session['txtremove_stopword']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_remove_unwanted_words == 'Ya':    
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #76a633;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Remove Unwanted Words : <b>{session['txtremove_unwanted_words']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()

                            if opt_remove_short_words == 'Ya':    
                                st.write(f'''
                                     <p style="text-decoration: none;
                                     font-family: 'Open Sans'; font-size: 13px;
                                     color: white; background-color: #046cd4;
                                     border-radius: 5px; padding: 9px 22px;">
                                     Remove Short Words : <b>{session['txtremove_short_words']}</b></p>
                                 ''', unsafe_allow_html = True)
                            else:
                                st.empty()
                                
        # Kolom Kosong
        else:
            st.write("")
            with st.error("Mohon input sentiment terlebih dahulu."):
                time.sleep(2)
                st.empty()
            
hide_streamlit = """ <style> footer {visibility: hidden;} </style> """    
st.markdown(hide_streamlit, unsafe_allow_html=True)            
