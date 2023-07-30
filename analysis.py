import streamlit as st

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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go

from io import StringIO
    
def analysis_page():
    @st.cache_data(show_spinner=False)
    def df():
        df = pd.read_csv("data/ulasan_tiket_kai_access_labeling_reviewer.csv")
        return df
        
    @st.cache_data(show_spinner=False)
    def process(df):        
        # 1. Data Checking
        st.subheader("Data Checking :", anchor='data-checking')
        st.markdown("_Melakukan pengecekan data, seperti nilai null dan memisahkan fitur yang akan digunakan._")
        tabMD, tabMKF= st.tabs([
            "&nbsp;&nbsp;&nbsp; **Menampilkan DataFrame** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Pilih Fitur & Cek Null** &nbsp;&nbsp;&nbsp;",
        ])

        # code :
        df['reviewer_1_sadam'] = df['reviewer_1_sadam'].replace({1:"Positive", 2: "Negative"})
        df['reviewer_2_malik'] = df['reviewer_2_malik'].replace({1:"Positive", 2: "Negative"})
        df['reviewer_3_dafit'] = df['reviewer_3_dafit'].replace({1:"Positive", 2: "Negative"})

        df.rename(columns = {'reviewer_1_sadam':'reviewer1',
                             'reviewer_2_malik':'reviewer2',
                             'reviewer_3_dafit':'reviewer3'}, inplace=True
                 )

        # menghapus kolom selain content, at, reviewer, total positive, total negative, dan label
        df.drop(df.columns[[0, 1, 2, 4, 5, 6, 8, 9, 10]], axis=1, inplace=True)

        with tabMD:
            with st.expander("DataFrame (Dengan Reviewer)", expanded=True):
                st.write("Total data: ", len(df))
                st.dataframe(df, use_container_width=True)

        # menghapus kolom selain content, at, dan label
        df.drop(df.columns[[2,3,4,5,6]], axis=1, inplace=True)

        # cek fitur yang kosong
        value_null = df['content'].isnull().sum()
        # cek fitur yang memiliki kata 'tiket'
        value_counts = df.content.str.contains('tiket', case=False).sum()

        with tabMKF:
            with st.expander("Output Data Checking", expanded=True):
                st.write("Cek jumlah data (null) :", value_null)
                st.write("Cek jumlah kolom 'content' yang memiliki kata 'tiket' :", value_counts)
                st.dataframe(df.head(), use_container_width=True)

        st.write("\n\n")
        st.markdown("Analisis Keluhan Pengguna :")
        st.markdown("_Dilakukan analisis keluhan pengguna terkait pemesanan, pembayaran, serta harga tiket._")
        tabPemesanan, tabPembayaran, tabHarga= st.tabs([
            "&nbsp;&nbsp;&nbsp; **Pemesanan Tiket** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Pembayaran Tiket** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Harga Tiket** &nbsp;&nbsp;&nbsp;",
        ])

        with tabPemesanan:
            keyword1 = 'pesan tiket'
            keyword2 = 'pemesanan tiket'
            keyword3 = 'order tiket'
            keyword4 = 'book tiket'
            keyword5 = 'booking tiket'
            key1 = df['content'].str.contains(keyword1, case=False)
            key2 = df['content'].str.contains(keyword2, case=False)
            key3 = df['content'].str.contains(keyword3, case=False)
            key4 = df['content'].str.contains(keyword4, case=False)
            key5 = df['content'].str.contains(keyword5, case=False)    
            data_keluhan_pemesanan_tiket = key1 | key2 | key3 | key4 | key5
            df_keluhan_pemesanan_tiket = df[data_keluhan_pemesanan_tiket]
            df_keluhan_pemesanan_tiket = df_keluhan_pemesanan_tiket.drop_duplicates()

            with st.expander("DataFrame", expanded=True):
                st.write("Total data keluhan pemesanan tiket :", len(df_keluhan_pemesanan_tiket))
                st.dataframe(df_keluhan_pemesanan_tiket.reset_index(drop=True), use_container_width=True)

            with st.expander("Histogram: Keluhan Pemesanan Tiket", expanded=True):
                # st.write("")
                keywords = [keyword1, keyword2, keyword3, keyword4, keyword5]
                df_keluhan_pemesanan_tiket['at'] = pd.to_datetime(df_keluhan_pemesanan_tiket['at'])
                df_keluhan_pemesanan_tiket['year'] = df_keluhan_pemesanan_tiket['at'].dt.year
                keluhan_per_tahun = df_keluhan_pemesanan_tiket['content'].str.contains('|'.join(keywords), case=False).groupby(df_keluhan_pemesanan_tiket['year']).sum()

                plt.figure(figsize=(10, 6))
                bars = plt.bar(keluhan_per_tahun.index, keluhan_per_tahun.values, color='#58508d')
                plt.xlabel('Tahun')
                plt.ylabel('Jumlah Keluhan')
                plt.title('Histogram: Keluhan Pemesanan Tiket')
                plt.xticks(rotation=45, ha='right')
                for bar, count in zip(bars, keluhan_per_tahun.values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height, count, ha='center', va='bottom', fontsize=10)

                total_per_tahun = keluhan_per_tahun.sum()
                plt.legend([f'{total_per_tahun} data (keluhan pemesanan tiket)'])
                plt.tight_layout()
                st.pyplot(plt)

            with st.expander("Histogram: Kategori Umum dalam Keluhan Pemesanan Tiket", expanded=True):
                # st.write("")
                df_keluhan_pemesanan_tiket['count_gagal'] = df_keluhan_pemesanan_tiket['content'].str.findall(r'gagal', flags=re.I).str.len()
                df_keluhan_pemesanan_tiket['count_sulit'] = df_keluhan_pemesanan_tiket['content'].str.findall(r'sulit', flags=re.I).str.len()
                df_keluhan_pemesanan_tiket['count_berhasil'] = df_keluhan_pemesanan_tiket['content'].str.findall(r'berhasil', flags=re.I).str.len()
                df_keluhan_pemesanan_tiket['count_mudah'] = df_keluhan_pemesanan_tiket['content'].str.findall(r'mudah', flags=re.I).str.len()
                total_gagal = df_keluhan_pemesanan_tiket['count_gagal'].sum()
                total_sulit = df_keluhan_pemesanan_tiket['count_sulit'].sum()
                total_berhasil = df_keluhan_pemesanan_tiket['count_berhasil'].sum()
                total_mudah = df_keluhan_pemesanan_tiket['count_mudah'].sum()
                total_data = total_gagal + total_berhasil + total_sulit + total_mudah
                colors = ['#9086b9', '#aca2d0','#c9c0e7','#746aa3']
                plt.figure(figsize=(8, 6))
                bars = plt.bar(['Gagal', 'Sulit', 'Berhasil', 'Mudah'], [total_gagal, total_sulit, total_berhasil , total_mudah], color=colors)
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate('{}'.format(height),
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10)
                plt.xlabel('Kategori')
                plt.ylabel('Jumlah')
                plt.title('Kategori Umum dalam Keluhan Pemesanan Tiket', size=10)
                plt.legend([f'Total: {total_data} '])
                st.pyplot(plt)

        with tabPembayaran:
            keyword1 = 'pembayaran tiket'
            keyword2 = 'bayar tiket'
            keyword3 = 'payment tiket'
            key1 = df['content'].str.contains(keyword1, case=False)
            key2 = df['content'].str.contains(keyword2, case=False)
            key3 = df['content'].str.contains(keyword3, case=False)
            data_keluhan_pembayaran_tiket = key1 | key2 | key3
            df_keluhan_pembayaran_tiket = df[data_keluhan_pembayaran_tiket]
            df_keluhan_pembayaran_tiket = df_keluhan_pembayaran_tiket.drop_duplicates()

            with st.expander("DataFrame", expanded=True):
                # st.write("")
                st.write("Total data keluhan pembayaran tiket :", len(df_keluhan_pembayaran_tiket))
                st.dataframe(df_keluhan_pembayaran_tiket.reset_index(drop=True),use_container_width=True)

            with st.expander("Histogram: Keluhan Pembayaran Tiket", expanded=True):
                # st.write("")
                keywords = [keyword1, keyword2, keyword3]
                df_keluhan_pembayaran_tiket['at'] = pd.to_datetime(df_keluhan_pembayaran_tiket['at'])
                df_keluhan_pembayaran_tiket['year'] = df_keluhan_pembayaran_tiket['at'].dt.year
                keluhan_per_tahun = df_keluhan_pembayaran_tiket['content'].str.contains('|'.join(keywords), case=False).groupby(df_keluhan_pembayaran_tiket['year']).sum()
                plt.figure(figsize=(10, 6))
                bars = plt.bar(keluhan_per_tahun.index, keluhan_per_tahun.values, color='#ff6361')
                plt.xlabel('Tahun')
                plt.ylabel('Jumlah Keluhan')
                plt.title('Histogram: Keluhan Pembayaran Tiket')
                plt.xticks(rotation=45, ha='right')
                for bar, count in zip(bars, keluhan_per_tahun.values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height, count, ha='center', va='bottom', fontsize=10)
                total_per_tahun = keluhan_per_tahun.sum()
                plt.legend([f'{total_per_tahun} data (keluhan pembayaran tiket)'])
                plt.tight_layout()
                st.pyplot(plt)

            with st.expander("Histogram: Kategori Umum dalam Keluhan Pembayaran Tiket", expanded=True):
                # st.write("")
                df_keluhan_pembayaran_tiket['count_gagal'] = df_keluhan_pembayaran_tiket['content'].str.findall(r'gagal', flags=re.I).str.len()
                df_keluhan_pembayaran_tiket['count_berhasil'] = df_keluhan_pembayaran_tiket['content'].str.findall(r'berhasil', flags=re.I).str.len()
                df_keluhan_pembayaran_tiket['count_sulit'] = df_keluhan_pembayaran_tiket['content'].str.findall(r'sulit', flags=re.I).str.len()
                df_keluhan_pembayaran_tiket['count_mudah'] = df_keluhan_pembayaran_tiket['content'].str.findall(r'mudah', flags=re.I).str.len()
                df_keluhan_pembayaran_tiket['count_linkaja'] = df_keluhan_pembayaran_tiket['content'].str.findall(r'linkaja', flags=re.I).str.len()
                df_keluhan_pembayaran_tiket['count_qris'] = df_keluhan_pembayaran_tiket['content'].str.findall(r'qris', flags=re.I).str.len()

                total_gagal = df_keluhan_pembayaran_tiket['count_gagal'].sum()
                total_berhasil = df_keluhan_pembayaran_tiket['count_berhasil'].sum()
                total_sulit = df_keluhan_pembayaran_tiket['count_sulit'].sum()
                total_mudah = df_keluhan_pembayaran_tiket['count_mudah'].sum()
                total_linkaja = df_keluhan_pembayaran_tiket['count_linkaja'].sum()
                total_qris = df_keluhan_pembayaran_tiket['count_qris'].sum()
                total_data = total_gagal + total_berhasil + total_sulit + total_mudah + total_linkaja + total_qris
                colors = ['#ff7d76', '#ffaba2','#ffd6d1','#ff6361','#ff6361','#ff948c']
                plt.figure(figsize=(8, 6))
                bars = plt.bar(['Gagal','Berhasil','Sulit', 'Mudah', 'LinkAja', 'QRIS'],
                               [total_gagal, total_berhasil, total_sulit, total_mudah, total_linkaja, total_qris], color=colors)
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate('{}'.format(height),
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10)
                plt.xlabel('Kategori')
                plt.ylabel('Jumlah')
                plt.title('Kategori Umum dalam Keluhan Pembayaran Tiket', size=10)
                plt.legend([f'Total: {total_data} '])
                st.pyplot(plt)

        with tabHarga:
            keyword1 = 'harga tiket'
            keyword2 = 'promo tiket'
            key1 = df['content'].str.contains(keyword1, case=False)
            key2 = df['content'].str.contains(keyword2, case=False)
            data_keluhan_harga_tiket = key1 | key2
            df_keluhan_harga_tiket = df[data_keluhan_harga_tiket]
            df_keluhan_harga_tiket = df_keluhan_harga_tiket.drop_duplicates()

            with st.expander("DataFrame", expanded=True):
                # st.write("")
                st.write("Total data keluhan harga tiket :", len(df_keluhan_harga_tiket))
                st.dataframe(df_keluhan_harga_tiket.reset_index(drop=True), use_container_width=True)

            with st.expander("Histogram: Keluhan harga tiket", expanded=True):
                # st.write("")
                keywords = [keyword1, keyword2]
                df_keluhan_harga_tiket['at'] = pd.to_datetime(df_keluhan_harga_tiket['at'])
                df_keluhan_harga_tiket['year'] = df_keluhan_harga_tiket['at'].dt.year
                keluhan_per_tahun = df_keluhan_harga_tiket['content'].str.contains('|'.join(keywords), case=False).groupby(df_keluhan_harga_tiket['year']).sum()
                plt.figure(figsize=(10, 6))
                bars = plt.bar(keluhan_per_tahun.index, keluhan_per_tahun.values, color='#439981')
                plt.xlabel('Tahun')
                plt.ylabel('Jumlah Keluhan')
                plt.title('Histogram: Keluhan Harga Tiket')
                plt.xticks(rotation=45, ha='right')
                for bar, count in zip(bars, keluhan_per_tahun.values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height, count, ha='center', va='bottom', fontsize=10)
                total_per_tahun = keluhan_per_tahun.sum()
                plt.legend([f'{total_per_tahun} data (keluhan harga tiket)'])
                plt.tight_layout()
                st.pyplot(plt)

            with st.expander("Histogram: Kategori Umum dalam Keluhan Harga Tiket", expanded=True):
                # st.write("")
                df_keluhan_harga_tiket['count_mahal'] = df_keluhan_harga_tiket['content'].str.findall(r'mahal', flags=re.I).str.len()
                df_keluhan_harga_tiket['count_murah'] = df_keluhan_harga_tiket['content'].str.findall(r'murah', flags=re.I).str.len()
                df_keluhan_harga_tiket['count_promo'] = df_keluhan_harga_tiket['content'].str.findall(r'promo', flags=re.I).str.len()
                total_mahal = df_keluhan_harga_tiket['count_mahal'].sum()
                total_murah = df_keluhan_harga_tiket['count_murah'].sum()
                total_promo = df_keluhan_harga_tiket['count_promo'].sum()
                total_data = total_mahal + total_murah + total_promo
                colors = ['#439981', '#5ca992','#74baa4']
                plt.figure(figsize=(8, 6))
                bars = plt.bar(['Mahal','Murah','Promo'], [total_mahal, total_murah, total_promo], color=colors)
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate('{}'.format(height),
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10)

                plt.xlabel('Kategori')
                plt.ylabel('Jumlah')
                plt.title('Kategori Umum dalam Keluhan Harga Tiket', size=10)
                plt.legend([f'Total: {total_data} '])
                st.pyplot(plt)

        st.divider()                

        # 2. Data Cleaning
        st.subheader("Data Cleaning :", anchor='data-cleaning')
        tabCasefolding, tabCleansing, tabHasilCleaning = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Casefolding** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Cleansing** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Hasil Cleaning** &nbsp;&nbsp;&nbsp;",
        ])

        def casefolding(text):
            text = text.lower()
            text = unidecode(text)
            return text

        def cleansing(text):
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        with tabCasefolding:
            df['cleaning'] = df['content'].apply(casefolding)
            st.markdown("_Proses melakukan konversi teks. Mengubah huruf besar menjadi huruf kecil, dan mengubah huruf aksen ke bentuk tanpa aksen yang setara (mis: huruf é menjadi e, huruf E menjadi e)._")
            with st.expander("Output Casefolding", expanded=True):
                st.dataframe(df['cleaning'].head(11), use_container_width=True)

        with tabCleansing:
            df['cleaning'] = df['cleaning'].apply(cleansing)
            st.markdown("_Proses membersihkan atau membuang noise (angka, tanda baca, emoji, multi spasi, dan baris enter)_")
            with st.expander("Output Cleansing", expanded=True):
                st.dataframe(df['cleaning'].head(11), use_container_width=True)

        with tabHasilCleaning:
            with st.expander("Output Data Cleaning", expanded=True):
                st.dataframe(df.head(11), use_container_width=True)            

        st.divider()

        # 3. Data Normalize
        st.subheader("Data Normalize :", anchor="data-normalize")
        tabStem, tabSlang, tabHasilNormalize = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Stemming** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Slang Word Normalization** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Hasil Normalize** &nbsp;&nbsp;&nbsp;",
        ])

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

        with tabStem:
            df['normalize'] = df['cleaning'].apply(stemming)
            st.markdown("_Proses menemukan kata dasar dengan menghilangkan semua imbuhan yang menyatu pada kata. Misalnya kata 'diperbaiki' akan diubah menjadi 'baik'._")
            with st.expander("Output Stemming", expanded=True):
                st.dataframe(df['normalize'].head(11), use_container_width=True)

        with tabSlang:
            st.markdown("_Proses mengubah kata non-baku (slang) menjadi kata baku._")
            with st.expander("Kamus Kata Slang Word", expanded=True):
                st.dataframe(kbba_dictionary.head(), use_container_width=True)

            df['normalize'] = df['normalize'].apply(convert_slangword)    
            with st.expander("Output Slangword Normalization", expanded=True):    
                st.dataframe(df['normalize'].head(11), use_container_width=True)

        with tabHasilNormalize:
            with st.expander("Output Data Normalize", expanded=True):
                st.dataframe(df.head(11), use_container_width=True)        

        st.divider() 

        # 4. Words Removal
        st.subheader("Words Removal :", anchor="words-removal")
        tabStopword, tabUnwanted, tabShortword, tabHasilWR = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Stopword** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Unwanted Word** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Short Word** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Hasil Words Removal** &nbsp;&nbsp;&nbsp;",
        ])

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

        with tabStopword:
            df['removal'] = df['normalize'].apply(remove_stopword)
            st.markdown("_Proses menghapus seluruh kata yang dianggap tidak memiliki makna. Seperti kata hubung 'yang', 'dan', 'dari'._")
            with st.expander("Output Stopwording", expanded=True):
                st.dataframe(df['removal'].head(11), use_container_width=True)

        with tabUnwanted:
            st.markdown("_Proses membuat dictionary kata-kata yang kurang dianggap bermakna, lalu menghapus kata yang sama dari ulasan. Kata yang dianggap tidak bermakna yaitu seperti nama bulan dalam kalender._")
            df['removal'] = df['removal'].apply(remove_unwanted_words)
            with st.expander("Output Unwanted Word Removal", expanded=True):
                st.dataframe(df['removal'].head(11), use_container_width=True)

        with tabShortword:
            df['removal'] = df['removal'].apply(remove_short_words)
            st.markdown("_Proses menghapus kata apapun yang kurang dari 3 karakter. Seperti kata 'di'._")
            with st.expander("Output Shortword Removal", expanded=True):
                st.dataframe(df['removal'].head(11), use_container_width=True)

        with tabHasilWR:
            with st.expander("Output Words Removal", expanded=True):
                st.dataframe(df.head(11), use_container_width=True)        

        st.divider()

        # 5. Tokenizing
        st.subheader("Tokenizing :", anchor="tokenizing")
        tabSplitwords, tabPie, tabWordcloud = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Split Words** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Pie Chart** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Wordcloud** &nbsp;&nbsp;&nbsp;",
            # "&nbsp;&nbsp;&nbsp; **TF-IDF** &nbsp;&nbsp;&nbsp;",
        ])

        tokenizer = Tokenizer()
        def tokenizing(text):
            return tokenizer.tokenize(text)

        with tabSplitwords:
            df['tokenizing'] = df['removal'].apply(tokenizing)
            st.markdown("_Proses pemisahan kata pada tiap ulasan._")
            with st.expander("Output Split Words", expanded=True):
                st.dataframe(df['tokenizing'].head(11), use_container_width=True)

        with tabPie:
            # pie chart
            df_sub = df.loc[df['label'].isin(['Positive', 'Negative'])]
            sizes = [count for count in df_sub['label'].value_counts()]
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

            st.markdown("_Proses melakukan visualisasi jumlah sentimen positive & negative menggunakan Pie Chart._")
            with st.expander("Output Pie Chart", expanded=True):
                st.pyplot(fig)

        with tabWordcloud:
            # wordcloud
            positive_words = df[df['label'] == 'Positive']['tokenizing'].apply(pd.Series).stack().tolist()
            positive_word_counts = Counter(positive_words)
            negative_words = df[df['label'] == 'Negative']['tokenizing'].apply(pd.Series).stack().tolist()
            negative_word_counts = Counter(negative_words)

            mask_pos = np.array(Image.open("img/train_pos.jpg"))
            mask_neg = np.array(Image.open("img/train_neg.jpg"))
            positive_wordcloud = WordCloud(width=800, height=600, mask=mask_pos, max_words=2000,
                                           background_color='black').generate_from_frequencies(positive_word_counts)
            negative_wordcloud = WordCloud(width=800, height=600, mask=mask_neg, max_words=2000,
                                           background_color='black').generate_from_frequencies(negative_word_counts)
            figPos, axPos = plt.subplots(figsize=(12, 8))
            axPos.imshow(positive_wordcloud.recolor(color_func=ImageColorGenerator(mask_pos)), interpolation='bilinear')
            axPos.axis('off')

            figNeg, axNeg = plt.subplots(figsize=(12, 8))
            axNeg.imshow(negative_wordcloud.recolor(color_func=ImageColorGenerator(mask_neg)), interpolation='bilinear')
            axNeg.axis('off')
            st.markdown("_Proses menampilkan seluruh kata dalam sentimen pada Wordcloud. Jika kata semakin sering muncul, maka ditampilkan dengan ukuran yang lebih besar._")
            with st.expander("Wordcloud - Kata Positive", expanded=True):
                st.subheader("Label Positive")
                st.pyplot(figPos)

            with st.expander("Wordcloud - Kata Negative", expanded=True):
                st.subheader("Label Negative")
                st.pyplot(figNeg)

        # 6. Filter Data Train & Test dan TF-IDF
        st.subheader("Filter Data Pengujian dan TF-IDF :", anchor="filter")
        tabFilter, tabTFidf = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Filter Data Pengujian** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Pembobotan Kalimat (TF-IDF)** &nbsp;&nbsp;&nbsp;",
        ])

        # df_2400_neg_pos = pd.read_excel("2400_pos_neg.xlsx")
        df_2400_neg_pos = pd.read_excel("2400_pos_neg.csv")
        df = df_2400_neg_pos

        cek_jumlah_positive = len(df[['content','label','tokenizing']][df['label'] == 'Positive'])
        cek_jumlah_negative = len(df[['content','label','tokenizing']][df['label'] == 'Negative'])

        with tabFilter:
            st.markdown("_Dari 6000 dataset yang ada, diambil 2400 data (1200 positive dan 1200 negative) untuk tahap pengujian._")
            with st.expander("Hasil Filter Data Pengujian", expanded=True):
                st.write("Total Data Pengujian :", len(df))
                st.write("Jumlah Positive :", cek_jumlah_positive)
                st.write("Jumlah Negative :", cek_jumlah_negative)
                st.dataframe(df.head(11))

        with tabTFidf:
            df = df.copy()
            df['tokenizing'] = df['tokenizing'].astype(str)
            tf_idf = TfidfVectorizer()
            review = df['tokenizing'].values.tolist()
            tf_idf_vector = tf_idf.fit(review)
            X = tf_idf_vector.transform(review)
            y = df['label']

            st.markdown("_Proses memberikan nilai bobot pada dokumen. Proses TF-IDF (Term Frequency-Inverse Document Frequency) tujuannya untuk mengetahui seberapa penting suatu kata dalam dokumen tersebut._")
            with st.expander("Hasil Pembobotan TF-IDF:", expanded=True):
                st.text(X[0:2])

        st.divider()

        # 7. Modeling
        st.subheader("Modeling :", anchor="modeling")
        tabPisahData, tabRFC = st.tabs([
            "&nbsp;&nbsp;&nbsp; **Pemisahaan Data (Train & Test)** &nbsp;&nbsp;&nbsp;",
            "&nbsp;&nbsp;&nbsp; **Random Forest Classifier** &nbsp;&nbsp;&nbsp;",
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)
        all_data = len(y)
        data_train = len(y_train)
        data_test = len(y_test)
        vector = X_train.shape, X_test.shape

        # rfc
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
        predict = rfc.predict(X_test)
        akurasi = accuracy_score(y_pred, y_test) * 100
        akurasi_bulat = round(akurasi, 1)

        with tabPisahData:
            st.markdown("_Proses pemisahan data latih (train) & data uji (test)._")
            with st.expander("Hasil Split Data (Train & Test)", expanded=True):
                st.write("Total Data : ", all_data)
                st.write("Total Data Train : ", data_train)
                st.write("Total Data Test : ", data_test)
                st.write("===")
                st.write("Label Train :")
                st.write(y_train.value_counts())
                st.write("\nLabel Test: ")
                st.write(y_test.value_counts())

        with tabRFC:
            st.markdown("_Pada proses ini, data yang telah dibagi akan dimodeling dengan Random Forest Classifier untuk mendapatkan akurasi._")
            with st.expander("Hasil Modeling", expanded=True):
                st.write("Random Forest Classifier (Accuracy): ", akurasi_bulat, "%")

        st.divider()

        # 8. Evaluasi Performa (Confusion Matrix)
        st.subheader("Evaluasi Performa (Confusion Matrix) :", anchor="evaluasi-performa")
        # classification_rep = classification_report(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred,output_dict=True)
        classification_rep = pd.DataFrame(classification_rep).transpose()
        # classification_df = pd.DataFrame(classification_rep).transpose()
        # classification_df = classification_df.drop(['accuracy'], axis=0)
        # classification_df = classification_df.round(2)

        cm = confusion_matrix(y_test, y_pred)
        figCM, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=rfc.classes_).plot(ax=ax)

        st.markdown("_Proses menampilkan Confusion Matrix. Confusion Matrix menyatakan jumlah data uji (test) yang benar dan salah diklasifikasi model._")

        # with st.expander("Hasil Classification Report", expanded=True):

        with st.expander("Confusion Matrix", expanded=True):
            st.pyplot(figCM)

        with st.expander("Tabel Confusion Matrix", expanded=True):
            TP = cm[0, 0]
            TN = cm[1, 1]
            FP = cm[1, 0]
            FN = cm[0, 1]
            resultAccuracy = (TN + TP) / (TP + TN + FP + FN)
            hitungAtas = TP + TN
            hitungBawah = TP + TN + FP + FN
            resultAccuracy = round(resultAccuracy, 3)

            hitung_true = TP+TN
            hitung_false = FP+FN
            df_cm = {
                "Value": [TP, TN, FP, FN],
                "Label": [
                    "True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"
                ],
                "Klasifikasi Benar (TP+TN)": [hitung_true,"-","-","-"],
                "Klasifikasi Salah (FP+FN)": [hitung_false,"-","-","-"]
            }

            st.dataframe(df_cm, use_container_width=True)

            st.dataframe(classification_rep, use_container_width=True)

            rumusCol1, perhitunganCol2 = st.columns(2)
            with rumusCol1:    
                st.markdown("Rumus Akurasi")
                st.latex(r'''
                    \small \frac{TP + TN}{TP + TN + FP + FN} = Accuracy
                ''')
            with perhitunganCol2:
                st.markdown("Hitung Akurasi")
                st.latex(r'''
                    \small \frac{%d + %d}{%d + %d + %d + %d} = %s
                ''' % (TP, TN, TP, TN, FP, FN, resultAccuracy))
                st.write("")
                st.latex(r'''
                    \small \frac{%d}{%d} = %s
                ''' % (hitungAtas, hitungBawah, resultAccuracy))        
    # run process
    df = df()
    process(df)
    
    
    print()
    print()
    #downloader
#     df_convert = pd.read_csv("data/df_label_polarity.csv")
#     selected_columns = ['content', 'cleaning', 'normalize', 'removal', 'tokenizing', 'polarity_score', 'polarity']
#     df_convert = df_convert.loc[:, selected_columns]
    
#     @st.cache_data()
#     def convert_df(df_convert):
#         return df_convert.to_csv(index=False)
    
#     @st.cache_data(show_spinner=False)
#     def convert_excel(df_convert):
#         output = BytesIO()
#         writer = pd.ExcelWriter(output, engine='xlsxwriter')
#         df_convert.to_excel(writer, index=False, sheet_name='Sheet1')
#         workbook = writer.book
#         worksheet = writer.sheets['Sheet1']
#         format1 = workbook.add_format({'num_format': '0.00'}) 
#         worksheet.set_column('A:A', None, format1)  
#         writer.close()
#         processed_data = output.getvalue()
#         return processed_data
    
#     with st.sidebar:
#         csv = convert_df(df_convert)

#         st.download_button(
#             "🖨️ Download Data berlabel (CSV)",
#             csv,
#             "df_label_polarity.csv",
#             "text/csv",
#             key='browser-data'
#         )
        
#         excels = convert_excel(df_convert)
#         st.download_button(label = "🖨️ Download Data berabel (Excel)", 
#               data = excels,
#               file_name = "df_label_polarity.xlsx", 
#               mime = 'text/xlsx',
#               key='browser-data2'
#         )
        
#         st.write("")
#         st.write("")
#         st.write("")

hide_streamlit = """ <style> footer {visibility: hidden;} </style> """                            
st.markdown(hide_streamlit, unsafe_allow_html=True)            
