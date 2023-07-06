import streamlit as st
session = st.session_state
import graphviz

import pandas as pd

@st.cache_data(show_spinner=True)
def split_frame(input_df, rows):
    #df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    df = [input_df.loc[i : i + rows, :] for i in range(0, len(input_df), rows)]
    return df

def tabulate():
    df_home = pd.read_csv("data/ulasan_tiket_kai_access.csv")
    # session['df_home'] = df_home
    
    # dataset = session['df_home']
    dataset = df_home
    top_menu = st.columns(3)
    with top_menu[0]:
        sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1)
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options=dataset.columns)
        with top_menu[2]:
            sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
        dataset = dataset.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )
    pagination = st.container()

    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[25, 50, 100, 1000, 2000])
    with bottom_menu[1]:
        total_pages = (
            int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")
    
    #--tambahan menghapus index dan mengurutkan dari 1--
    dataset.reset_index(drop=True, inplace=True)
    dataset.index += 1
    #--tambahan--
    pages = split_frame(dataset, batch_size)
    pagination.dataframe(data=pages[current_page - 1], use_container_width=True)
    

def admin_home_page():
    st.title("App Home", help="Menampilkan ringkasan Sentiment Analysis Aplikasi KAI Access")
    
    name = session["name"]
    st.info(f"Welcome, **_{name}_**!", icon="ℹ️")
    st.success("Sentiment Analysis Menggunakan Metode Text Mining dan Random Forest Untuk Klasifikasi Keluhan Pengguna (Studi Kasus: Aplikasi KAI Access)", icon="ℹ️")
    
    st.divider()
    
    st.header("Dataset Ulasan KAI Access :")
    st.write("")
    
    with st.expander("Show / Hide", expanded=True):
        st.write("")
        tabulate()
        
    st.write("")
    st.header("Graphviz Chart :")
    with st.expander("Proses Sentiment Analysis", expanded=True):
        # Create a graphlib graph object
        if st.checkbox("Graph 1", value=False):
            graph = graphviz.Digraph()
            graph.edge('Start', 'Data Checking')
            graph.edge('Data Checking', 'Data Cleaning')
            graph.edge('Data Cleaning', 'Casefolding')
            graph.edge('Casefolding', 'Cleansing')
            graph.edge('Cleansing', 'Data Cleaning')
            st.graphviz_chart(graph)
        
        if st.checkbox("Graph 2", value=False):
            graph2 = graphviz.Digraph()
            graph2.edge('Data Checking', 'Data Normalize')
            graph2.edge('Data Normalize','Lemmatization')
            graph2.edge('Lemmatization', 'Stemming')
            graph2.edge('Stemming','Slang Word Normalization')
            graph2.edge('Slang Word Normalization','Data Normalize')
            st.graphviz_chart(graph2)
        
        if st.checkbox("Graph 3", value=False):
            graph3 = graphviz.Digraph()
            graph3.edge('Data Checking', 'Words Removal')
            graph3.edge('Words Removal','Stopword Removal')
            graph3.edge('Stopword Removal','Unwanted Words Removal')
            graph3.edge('Unwanted Words Removal','Short Word Removal')
            graph3.edge('Short Word Removal','Words Removal')
            st.graphviz_chart(graph3)
            
        if st.checkbox("Graph 4", value=False):
            graph4 = graphviz.Digraph()
            graph4.edge('Data Checking', 'Tokenizing')
            graph4.edge('Tokenizing', 'Split Words')
            graph4.edge('Split Words','Labeling')
            graph4.edge('Labeling','Visualisasi Pie Chart')
            graph4.edge('Labeling','Visualisasi Wordcloud')
            graph4.edge('Labeling','Pembobotan TF-IDF')
            graph4.edge('Pembobotan TF-IDF', 'Tokenizing')
            st.graphviz_chart(graph4)    
            
        if st.checkbox("Graph 5", value=False):
            graph5 = graphviz.Digraph()
            graph5.edge('Data Checking', 'Modeling')
            graph5.edge('Modeling', 'Pemisahan Data (Train & Test)')
            graph5.edge('Pemisahan Data (Train & Test)','Random Forest Classifier')
            graph5.edge('Random Forest Classifier','Modeling')
            st.graphviz_chart(graph5) 
            
        if st.checkbox("Graph 6", value=False):
            graph6 = graphviz.Digraph()
            graph6.edge('Data Checking', 'Evaluasi Performa')
            graph6.edge('Evaluasi Performa','Diagram ROC-AUC')
            graph6.edge('Diagram ROC-AUC','Classification Report')
            graph6.edge('Classification Report','Confusion Matrix')
            graph6.edge('Confusion Matrix','Evaluasi Performa')
            graph6.edge('Data Checking','End')
            
            st.graphviz_chart(graph6) 
        
hide_streamlit = """ <style> footer {visibility: hidden;} </style> """        
st.markdown(hide_streamlit, unsafe_allow_html=True)
