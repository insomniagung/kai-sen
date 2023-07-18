import streamlit as st
# session = st.session_state
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import xlsxwriter
from io import BytesIO

import pandas as pd
import numpy as np
import re
from unidecode import unidecode

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize
from nlp_id import StopWord, Tokenizer

from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
# from pandas.core.common import random_state
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go
    
def analysis_page():
    @st.cache_data(show_spinner=False)
    def df():
        df = pd.read_csv("data/ulasan_tiket_kai_access.csv")
        return df    
    
    @st.cache_data(show_spinner=False)
    def process(df):
        
        # 1. Data Checking
        st.subheader("Data Checking :", anchor='data-checking')
        st.markdown("_Melakukan pengecekan data, seperti nilai null dan memisahkan fitur yang akan digunakan._")
        tabMD, tabMKF= st.tabs([
            "&nbsp;&nbsp;&nbsp; **Menampilkan DataFrame** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Memilah & Cek Kondisi Fitur** &nbsp;&nbsp;&nbsp;",
        ])

        with tabMD:
            df_total = len(df)
            
            with st.expander("Dataframe", expanded=True):
                st.write("Jumlah data: ", df_total)
                st.write("")
                st.dataframe(df, use_container_width=True)

        with tabMKF:
            df = df.drop(columns=df.columns.difference(['content']))

            value_null = df.content.isnull().sum()
            value_counts = df.content.str.contains('tiket', case=False).sum()
            with st.expander("Output Check Data", expanded=True):
                st.write("Jumlah baris fitur yang kosong : ", value_null)
                st.write("Jumlah baris fitur yang memiliki kata 'tiket' : ", value_counts)
                st.write("")
                st.dataframe(df.head(), use_container_width=True)
        st.divider()
    
        # 2. Data Cleaning
        def casefolding(text):
            text = text.lower()
            text = unidecode(text)
            return text

        def cleansing(text):
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        st.subheader("Data Cleaning :", anchor='data-cleaning')
        tabCasefolding, tabCleansing, tabHasilCleaning = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Casefolding** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Cleansing** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Hasil Cleaning** &nbsp;&nbsp;&nbsp;",
        ])

        with tabCasefolding:
            df['cleaning'] = df['content'].apply(casefolding)
            st.markdown("_Proses melakukan konversi teks. Mengubah huruf besar menjadi huruf kecil, dan mengubah huruf aksen ke bentuk tanpa aksen yang setara (mis: huruf é menjadi e, huruf E menjadi e)._")
            with st.expander("Output Casefolding", expanded=True):
                st.dataframe( df['cleaning'].head(11) , use_container_width=True)

        with tabCleansing:
            df['cleaning'] = df['cleaning'].apply(cleansing)
            st.markdown("_Proses membersihkan atau membuang noise (angka, tanda baca, emoji, multi spasi, dan baris enter)_")
            with st.expander("Output Cleansing", expanded=True):
                st.dataframe( df['cleaning'].head(11) , use_container_width=True)

        with tabHasilCleaning:
            # st.markdown("_Hasil Cleaning pada Content._")
            with st.expander("Output Cleaning", expanded=True):
                st.dataframe(df.head(11) , use_container_width=True)            

        st.divider()

        # 3. Data Normalize
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        def stemming(text):
            stemmed_text = stemmer.stem(text)
            return stemmed_text

        kbba_dictionary = pd.read_csv(
                    'https://raw.githubusercontent.com/insomniagung/kamus_kbba/main/kbba.txt', 
                    delimiter='\t', names=['slang', 'formal'], header=None, encoding='utf-8')

        slang_dict = dict(zip(kbba_dictionary['slang'], kbba_dictionary['formal']))
        def convert_slangword(text):
            words = text.split()
            normalized_words = [slang_dict[word] if word in slang_dict else word for word in words]
            normalized_text = ' '.join(normalized_words)
            return normalized_text

        st.subheader("Data Normalize :", anchor="data-normalize")
        tabStem, tabSlang, tabHasilNormalize = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Stemming** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Slang Word Normalization** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Hasil Normalize** &nbsp;&nbsp;&nbsp;",
        ])

        with tabStem:
            df['normalize'] = df['cleaning'].apply(stemming)
            st.markdown("_Proses menemukan kata dasar dengan menghilangkan semua imbuhan yang menyatu pada kata. Misalnya kata 'diperbaiki' akan diubah menjadi 'baik'._")
            with st.expander("Output Stemming", expanded=True):
                st.dataframe( df['normalize'].head(11) , use_container_width=True)

        with tabSlang:
            st.markdown("_Proses mengubah kata non-baku (slang) menjadi kata baku._")
            with st.expander("Kamus Kata Slang Word", expanded=True):
                st.dataframe(kbba_dictionary, use_container_width=True)

            df['normalize'] = df['normalize'].apply(convert_slangword)    
            with st.expander("Output Slangword Normalization", expanded=True):    
                st.dataframe(df['normalize'].head(11) , use_container_width=True)

        with tabHasilNormalize:
            with st.expander("Output Normalize", expanded=True):
                st.dataframe(df.head(11) , use_container_width=True)        

        st.divider() 

        # 4. Words Removal
        def remove_stopword(text):
            stopword = StopWord()
            text = stopword.remove_stopword(text)
            return text
        
        def remove_unwanted_words(text):
            unwanted_words = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 
                              'sep', 'oct', 'nov', 'dec', 'januari', 'februari', 'maret', 
                              'april', 'mei', 'juni', 'juli', 'agustus', 'september', 
                              'oktober', 'november', 'desember', 'gin'}
            word_tokens = word_tokenize(text)
            filtered_words = [word for word in word_tokens if word not in unwanted_words]
            filtered_text = ' '.join(filtered_words)
            return filtered_text

        def remove_short_words(text):
            return ' '.join([word for word in text.split() if len(word) >= 3])

        st.subheader("Words Removal :", anchor="words-removal")
        tabStopword, tabUnwanted, tabShortword, tabHasilWR = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Stopword** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Unwanted Word** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Short Word** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Hasil Words Removal** &nbsp;&nbsp;&nbsp;",
        ])

        with tabStopword:
            df['removal'] = df['normalize'].apply(remove_stopword)
            st.markdown("_Proses menghapus seluruh kata yang dianggap tidak memiliki makna. Seperti kata hubung 'yang', 'dan', 'dari'._")
            with st.expander("Output Stopwording", expanded=True):
                st.dataframe(df['removal'].head(11) , use_container_width=True)

        with tabUnwanted:
            st.markdown("_Proses membuat dictionary kata-kata yang kurang dianggap bermakna, lalu menghapus kata yang sama dari ulasan. Kata yang dianggap tidak bermakna yaitu seperti nama bulan dalam kalender._")
            df['removal'] = df['removal'].apply(remove_unwanted_words)
            with st.expander("Output Unwanted Word Removal", expanded=True):
                st.dataframe(df['removal'].head(11) , use_container_width=True)

        with tabShortword:
            df['removal'] = df['removal'].apply(remove_short_words)
            st.markdown("_Proses menghapus kata apapun yang kurang dari 3 karakter. Seperti kata 'di'._")
            with st.expander("Output Shortword Removal", expanded=True):
                st.dataframe(df['removal'].head(11) , use_container_width=True)

        with tabHasilWR:
            with st.expander("Output Words Removal", expanded=True):
                st.dataframe(df.head(11) , use_container_width=True)        

        st.divider()

        # 5. Tokenizing
        tokenizer = Tokenizer()
        def tokenizing(text):
            return tokenizer.tokenize(text)

        # Dictionary kata positif yang digunakan :
        df_positive = pd.read_csv(
            'https://raw.githubusercontent.com/SadamMahendra/ID-NegPos/main/positive.txt', sep='\t')
        
        list_positive = list(df_positive.iloc[::, 0])

        # Dictionary kata negatif yang digunakan :
        df_negative = pd.read_csv(
            'https://raw.githubusercontent.com/SadamMahendra/ID-NegPos/main/negative.txt', sep='\t')
        list_negative = list(df_negative.iloc[::, 0])

        def sentiment_analysis_dictionary_id(text):
            score = 0
            positive_words = []
            negative_words = []
            neutral_words = []

            for word in text:
                if (word in list_positive):
                    score += 1
                    positive_words.append(word)
                if (word in list_negative):
                    score -= 1
                    negative_words.append(word)
                if (word not in list_positive and word not in list_negative): 
                    neutral_words.append(word)

            polarity = ''
            if (score > 0):
                polarity = 'positive'
            elif (score < 0):
                polarity = 'negative'
            else:
                polarity = 'neutral'

            # return score, polarity
            result = {'positif': positive_words,'negatif':negative_words,'netral': neutral_words}
            return score, polarity, result, positive_words, negative_words

        st.subheader("Tokenizing :", anchor="tokenizing")
        tabSplitwords, tabLabeling, tabTFidf = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Split Words** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Labeling** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **TF-IDF** &nbsp;&nbsp;&nbsp;",
        ])

        with tabSplitwords:
            df['tokenizing'] = df['removal'].apply(tokenizing)
            st.markdown("_Proses pemisahan kata pada tiap ulasan._")
            with st.expander("Output Split Words", expanded=True):
                st.dataframe(df['tokenizing'].head(11) , use_container_width=True)

        with tabLabeling:
            df_positive_words = pd.DataFrame({'List Positive': list_positive})
            df_negative_words = pd.DataFrame({'List Negative': list_negative})
            df_dictionary = pd.concat([df_positive_words, df_negative_words], axis=1)
            st.markdown("_Proses melakukan Labeling (positif & negatif) pada ulasan._")
            with st.expander("Dictionary Positif & Negative", expanded=True):
                st.dataframe(df_dictionary.head(100) , use_container_width=True)

            hasil = df['tokenizing'].apply(sentiment_analysis_dictionary_id)        
            hasil = list(zip(*hasil))
            df['polarity_score'] = hasil[0]
            df['polarity'] = hasil[1]

            with st.expander("Hasil Polarity", expanded=True):
                st.write("Jumlah data: ", df['polarity'].value_counts().sum())
                st.write("")
                st.write("Detail :")
                st.write(df['polarity'].value_counts())

            # Delete & save Neutral
            neutral_df = df[df.polarity == 'neutral'][['tokenizing', 'polarity', 'polarity_score']]
            # neutral_df.to_csv('data/polarity_neutral.csv', index=False)

            ## Sidebar Download
#             with st.sidebar:
#                 with st.expander("Download", expanded=True):
#                     output = BytesIO()
#                     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#                     neutral_df.to_excel(writer, index=False, sheet_name='Sheet1')
#                     workbook = writer.book
#                     worksheet = writer.sheets['Sheet1']
#                     format1 = workbook.add_format({'num_format': '0.00'}) 
#                     worksheet.set_column('A:A', None, format1)  
#                     writer.close()
#                     processed_data = output.getvalue()

#                     st.download_button(label = "🖨️ Download Excel (Neutral)", 
#                                               data = processed_data,
#                                               file_name = "Polarity Neutral.xlsx", 
#                                               mime = 'text/xlsx')
                    
            df = df[df.polarity != 'neutral']
            
            hasil_kata_positive = hasil[3]
            hasil_kata_negative = hasil[4]

            total_polar = len(df['tokenizing'])
            polar_result = df['polarity'].value_counts()

            with st.expander("Process Labeling Positive & Negative", expanded=True):
                # st.write("Neutral telah disimpan dalam local.")
                # st.write("")
                # st.write("Neutral dihapus dari polarity.")
                # st.write("")
                st.write("Hasil Polarity: ")
                st.write(f"{total_polar} (Positive & Negative)")
                st.write("Detail :")
                st.write(polar_result)
            
            
            with st.expander("Hasil Labeling Positive & Negative", expanded=True):
                df_label_polarity = df
                st.dataframe(df.head(11) , use_container_width=True)
        
        positive_words = df[df.polarity == 'positive']['tokenizing'].apply(pd.Series).stack().tolist()
        positive_word_counts = Counter(positive_words)

        negative_words = df[df.polarity == 'negative']['tokenizing'].apply(pd.Series).stack().tolist()
        negative_word_counts = Counter(negative_words)
        
        with tabTFidf:
            df = df.copy()
            df['tokenizing'] = df['tokenizing'].astype(str)
            
            tf_idf = TfidfVectorizer()

            review = df['tokenizing'].values.tolist()

            tf_idf_vector = tf_idf.fit(review)

            X = tf_idf_vector.transform(review)
            y = df['polarity']
            
            st.markdown("_Proses memberikan nilai bobot pada dokumen. Proses TF-IDF (Term Frequency-Inverse Document Frequency) tujuannya untuk mengetahui seberapa penting suatu kata dalam dokumen tersebut._")
            with st.expander("Pembobotan TF-IDF", expanded=True):
                # st.write("Max features:", content_tfidf)
                # st.write("")
                st.text(X[0:2])

        st.divider()
        
        st.subheader("Analisis Deskriptif :")
        tabPie, tabWordcloud, tabTopWords = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Pie Chart** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Wordcloud** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Top Words** &nbsp;&nbsp;&nbsp;"
        ])
        with tabPie:
            st.markdown("_Proses melakukan visualisasi jumlah sentimen positive & negative menggunakan Pie Chart._")
            # Membuat subset data hanya dengan sentimen positif dan negatif
            df_sub = df.loc[df.polarity.isin(['positive', 'negative'])]
            sizes = [count for count in df_sub.polarity.value_counts()]
            explode = (0.1, 0)
            total_sizes = sum(sizes)
            fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')

            labels = ['Negative', 'Positive']
            colors = ['#ff9999', '#66b3ff']
            wedgeprops = {'width': 0.7, 'edgecolor': 'white', 'linewidth': 2}

            pie = ax.pie(x=sizes, labels=['', ''], colors=colors, explode=explode,
                autopct=lambda pct: "{:.1f}%\n({:d})".format(pct, int(pct / 100 * total_sizes)),
                textprops={'fontsize': 7, 'color': 'black'}, shadow=True,
                wedgeprops=wedgeprops)

            ax.legend(pie[0], labels, loc='center left', fontsize=7)
            ax.set_title(f"Sentiment Analysis on KAI Access Reviews \n(Total: {total_sizes} reviews)", 
                         fontsize=8, color='white', pad=4)

            with st.expander("Output Pie Chart", expanded=True):
                st.pyplot(fig)

        with tabWordcloud:
#             positive_words = df[df.polarity == 'positive']['tokenizing'].apply(pd.Series).stack().tolist()
#             positive_word_counts = Counter(positive_words)

#             negative_words = df[df.polarity == 'negative']['tokenizing'].apply(pd.Series).stack().tolist()
#             negative_word_counts = Counter(negative_words)

            mask_pos = np.array(Image.open("img/train_pos.jpg"))
            mask_neg = np.array(Image.open("img/train_neg.jpg"))

            positive_wordcloud = WordCloud(width=800, height=600, mask=mask_pos, max_words=2000,
                                           background_color='black').generate_from_frequencies(positive_word_counts)

            negative_wordcloud = WordCloud(width=800, height=600, mask=mask_neg, max_words=2000,
                                           background_color='black').generate_from_frequencies(negative_word_counts)

            figPos, axPos = plt.subplots(figsize=(12, 8))
            axPos.imshow(positive_wordcloud.recolor(color_func=ImageColorGenerator(mask_pos)), interpolation='bilinear')
            axPos.axis('off')
            
            st.markdown("_Proses menampilkan seluruh kata dalam sentimen pada Wordcloud. Jika kata semakin sering muncul, maka ditampilkan dengan ukuran yang lebih besar._")
            with st.expander("Wordcloud - Kata Positive", expanded=True):
                st.subheader("Label Positive")
                st.pyplot(figPos)

            figNeg, axNeg = plt.subplots(figsize=(12, 8))
            axNeg.imshow(negative_wordcloud.recolor(color_func=ImageColorGenerator(mask_neg)), interpolation='bilinear')
            axNeg.axis('off')
            with st.expander("Wordcloud - Kata Negative", expanded=True):
                st.subheader("Label Negative")
                st.pyplot(figNeg)
                
        with tabTopWords:
            st.markdown("Top 20 Words dari label positive dan negative")
            def top_words(hasil_kata_positive, hasil_kata_negative):
                all_positive_words = [word for sublist in hasil_kata_positive for word in sublist]
                all_negative_words = [word for sublist in hasil_kata_negative for word in sublist]
                positive_freq = pd.Series(all_positive_words).value_counts().reset_index().rename(columns={'index': 'Positive Word', 0: 'Frequency'})
                negative_freq = pd.Series(all_negative_words).value_counts().reset_index().rename(columns={'index': 'Negative Word', 0: 'Frequency'})
                top_20_positive = positive_freq.head(20)
                top_20_negative = negative_freq.head(20)
                return top_20_positive, top_20_negative

            top_kata_positive, top_kata_negative = top_words(hasil_kata_positive, hasil_kata_negative)

            with st.expander("Top Words Positive"):
                st.dataframe(top_kata_positive, use_container_width=True)
            with st.expander("Top Words Negative"):
                st.dataframe(top_kata_negative, use_container_width=True)


            st.divider()        
        # 6. Modeling
        st.subheader("Modeling :", anchor="modeling")
        tabPisahData, tabRFC = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Pemisahaan Data (Train & Test)** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Random Forest Classifier** &nbsp;&nbsp;&nbsp;",
        ])

        with tabPisahData:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)
            all_data = len(y)
            data_train = len(y_train)
            data_test = len(y_test)
            
            st.markdown("_Proses pemisahan data latih (train) & data uji (test). Data latih (train) ditetapkan 90%, dan data uji (test) sebanyak 10%._")
            with st.expander("Hasil Pemisahan Data Train & Test", expanded=True):
                st.write("Total Data : ", all_data)
                st.write("")
                st.write("Total Data Train : ", data_train)
                st.write("Total Data Test : ", data_test)

        with tabRFC:
            rfc = RandomForestClassifier(
                n_estimators=100, 
                criterion = "gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                n_jobs=-1, 
                oob_score=True)
            
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            predict = rfc.predict(X_test)

            akurasi = accuracy_score(y_pred, y_test) * 100
            akurasi_bulat = round(akurasi, 1)
            
            st.markdown("_Pada proses ini, data yang telah dibagi akan dimodeling dengan Random Forest Classifier untuk mendapatkan akurasi._")
            with st.expander("Hasil Modeling", expanded=True):
                st.write("Random Forest Classifier (Accuracy): ", akurasi_bulat, "%")

        st.divider()
    
        # 7. Evaluasi Performa Model
        st.subheader("Evaluasi Performa :", anchor="evaluasi-performa")
        tabClass, tabCM = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Classification Report** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Confusion Matrix** &nbsp;&nbsp;&nbsp;",
        ])

        with tabClass:            
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            classification_df = pd.DataFrame(classification_rep).transpose()
            classification_df = classification_df.drop(['accuracy'], axis=0)
            classification_df = classification_df.round(2)
            
            st.markdown("_Proses menampilkan hasil kinerja model klasifikasi. Membantu dalam menganalisis dan memahami seberapa baik model dapat memprediksi label dengan benar. Jika semakin tinggi persentase Precision, Recall, dan F1-Score maka model sudah seimbang dan baik._")
            with st.expander("Hasil Classification Report", expanded=True):
                st.dataframe(classification_df)


        with tabCM:
            cm = confusion_matrix(y_test, y_pred)
            figCM, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=rfc.classes_).plot(ax=ax)
            
            st.markdown("_Proses menampilkan Confusion Matrix dan Menghitung Akurasi Model. Confusion Matrix menyatakan jumlah data uji (test) yang benar dan salah diklasifikasi. Menghasilkan output True Positive, True Negative, False Positive, dan False Negative. Jika jumlah True (Positive & Negative) lebih banyak dari False (Positive & Negative), maka hasil data uji (test) dikatakan sudah baik._")
            with st.expander("Confusion Matrix", expanded=True):
                st.pyplot(figCM)
                
            with st.expander("Perhitungan Accuracy Confusion Matrix", expanded=True):
                TP = cm[0, 0]
                TN = cm[1, 1]
                FP = cm[1, 0]
                FN = cm[0, 1]
                resultAccuracy = (TN + TP) / (TP + TN + FP + FN)
                resultAccuracy = round(resultAccuracy, 3)
                
                df_cm = {
                    "Value": [TP, TN, FP, FN],
                    "Label": [
                        "True Positive", "True Negative", "False Positive", "False Negative"
                    ],
                    "As": ["TP", "TN", "FP", "FN"]
                }
                
                st.dataframe(df_cm, use_container_width=True)
                
                rumusCol1, perhitunganCol2 = st.columns(2)
                with rumusCol1:    
                    st.markdown("Equation Accuracy")
                    st.latex(r'''
                        \small \frac{TP + TN}{TP + TN + FP + FN} = Accuracy
                    ''')
                with perhitunganCol2:
                    st.markdown("Calculate Accuracy")
                    st.latex(r'''
                        \small \frac{%d + %d}{%d + %d + %d + %d} = %s
                    ''' % (TP, TN, TP, TN, FP, FN, resultAccuracy)) 
        
    # run process
    df = df()
    process(df)
    
    
    print()
    print()
    #downloader
    df_convert = pd.read_csv("data/df_label_polarity.csv")
    selected_columns = ['content', 'cleaning', 'normalize', 'removal', 'tokenizing', 'polarity_score', 'polarity']
    df_convert = df_convert.loc[:, selected_columns]
    
    @st.cache_data()
    def convert_df(df_convert):
        return df_convert.to_csv(index=False)
    
    @st.cache_data(show_spinner=False)
    def convert_excel(df_convert):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df_convert.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
        writer.close()
        processed_data = output.getvalue()
        return processed_data
    
    with st.sidebar:
        csv = convert_df(df_convert)

        st.download_button(
            "🖨️ Download Data berlabel (CSV)",
            csv,
            "df_label_polarity.csv",
            "text/csv",
            key='browser-data'
        )
        
        excels = convert_excel(df_convert)
        st.download_button(label = "🖨️ Download Data berabel (Excel)", 
              data = excels,
              file_name = "df_label_polarity.xlsx", 
              mime = 'text/xlsx',
              key='browser-data2'
        )
        
        st.write("")
        st.write("")
        st.write("")

hide_streamlit = """ <style> footer {visibility: hidden;} </style> """                            
st.markdown(hide_streamlit, unsafe_allow_html=True)            
