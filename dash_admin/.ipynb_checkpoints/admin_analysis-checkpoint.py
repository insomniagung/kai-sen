import streamlit as st
session = st.session_state
from analysis import analysis_page

def admin_analysis_page():
    #sidebar
    @st.cache_data()
    def section():
        with st.sidebar.expander("Data Preparation", expanded=True):
            st.write("")
            st.write('''
                     <a href="#data-checking" style="text-decoration: none;
                     font-family: 'Montserrat'; font-size: 13px; font-weight: bold;
                     color: black; background-color: #ffa91e; 
                     border-radius: 50px; padding: 9px 22px;">
                     #1 Data Checking</a>
                     ''', unsafe_allow_html = True)
            st.write('''
                     <a href="#data-cleaning" style="text-decoration: none;
                     font-family: 'Montserrat'; font-size: 13px; font-weight: bold;
                     color: white; background-color: #3b5998; 
                     border-radius: 50px; padding: 9px 20px;">
                     #2 Data Cleaning</a>
                     ''', unsafe_allow_html = True)
            st.write('''
                     <a href="#data-normalize" style="text-decoration: none;
                     font-family: 'Montserrat'; font-size: 13px; font-weight: bold;
                     color: white; background-color: #317256; 
                     border-radius: 50px; padding: 9px 20px;">
                     #3 Data Normalize</a>
                     ''', unsafe_allow_html = True)
            st.write('''
                     <a href="#words-removal" style="text-decoration: none;
                     font-family: 'Montserrat'; font-size: 13px; font-weight: bold;
                     color: white; background-color: #7b7d7b; 
                     border-radius: 50px; padding: 9px 20px;">
                     #4 Words Removal</a>
                     ''', unsafe_allow_html = True)
            st.write('''
                     <a href="#tokenizing" style="text-decoration: none;
                     font-family: 'Montserrat'; font-size: 13px; font-weight: bold;
                     color: white; background-color: #e0474c; 
                     border-radius: 50px; padding: 9px 20px;">
                     #5 Tokenizing</a>
                     ''', unsafe_allow_html = True)
            st.write('''
                     <a href="#modeling" style="text-decoration: none;
                     font-family: 'Montserrat'; font-size: 13px; font-weight: bold;
                     color: black; background-color: #ffe303; 
                     border-radius: 50px; padding: 9px 20px;">
                     #6 Modeling</a>
                     ''', unsafe_allow_html = True)
            st.write('''
                     <a href="#evaluasi-performa" style="text-decoration: none;
                     font-family: 'Montserrat'; font-size: 13px; font-weight: bold;
                     color: white; background-color: #8250e5; 
                     border-radius: 50px; padding: 9px 20px;">
                     #7 Evaluasi Performa</a>
                     ''', unsafe_allow_html = True)
            st.write("")

    
    #main
    st.title("Sentiment Analysis KAI Access", help="Menampilkan sentiment analysis secara detail pada Ulasan KAI Access di Google Play Store.")
    st.divider()
    
    section()
    analysis_page()
    
hide_streamlit = """ <style> footer {visibility: hidden;} </style> """                            
st.markdown(hide_streamlit, unsafe_allow_html=True)