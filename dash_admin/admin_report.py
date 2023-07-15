import streamlit as st
import time
session = st.session_state
import pandas as pd
import datetime

import xlsxwriter
from io import BytesIO

@st.cache_data(show_spinner=False)
def convert_csv(df):
    return df.to_csv().encode('utf-8')

@st.cache_data(show_spinner=False)
def convert_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def admin_report_page():
    st.title("Report", help="Halaman laporan dari dataset.")
    st.divider()
    
    df_report = pd.read_csv("data/ulasan_tiket_kai_access.csv")
    
    session['df_report'] = df_report
    df_report = session['df_report']
    
    st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 13px;
             color: white; background-color: #046cd4; 
             border-radius: 5px; padding: 7px 13px;">
             <b>Cari data berdasarkan tanggal</b></span>
         ''', unsafe_allow_html = True)
    
    df_report['at'] = pd.to_datetime(df_report['at'])
    minimum_date = df_report['at'].dt.date.min()
    maximum_date = df_report['at'].dt.date.max()
        
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.date_input(
            "Input Tanggal Awal:",
            min_value = minimum_date,
            max_value = maximum_date,
            value=datetime.date(2023, 1, 1)
        )
    with col2:
        d2 = st.date_input(
            "Input Tanggal Akhir:",
            min_value = minimum_date,
            max_value = maximum_date,
            value=datetime.date(2023, 2, 1)
        )
        
    start_date = d1.strftime('%Y-%m-%d')
    IDstart_date = d1.strftime('%d %B %Y')
    session['IDstart_date'] = IDstart_date
    
    end_date = d2.strftime('%Y-%m-%d')
    IDend_date = d2.strftime('%d %B %Y')
    session['IDend_date'] = IDend_date
    
    end_date_plus = (d2 + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    with st.expander("Show Data", expanded=True):
        
        with st.spinner('Memproses...'):
            st.empty()
            time.sleep(1)


        # Convert the date to datetime64
        df_report['at'] = pd.to_datetime(df_report['at'], format='%Y-%m-%d')

        # Filter data between two dates
        filtered_df = df_report.loc[(df_report['at'] >= start_date) & (df_report['at'] <= end_date_plus)]

        sorted_df = filtered_df.sort_values('at', ascending=True)
        selected_columns = sorted_df[['content', 'at']]

        # mengubah format
        # selected_columns['at'] = selected_columns['at'].dt.strftime('%d %B, %Y')
        selected_columns = selected_columns.copy()
        selected_columns['at'] = selected_columns['at'].dt.strftime('%d-%m-%Y')

        selected_columns.reset_index(drop=True, inplace=True)
        selected_columns.index += 1

        # Display
        st.write("")
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 13px;
             color: white; background-color: #046cd4; 
             border-radius: 20px; padding: 7px 13px;">
             Data dari <b>{IDstart_date}</b> sampai <b>{IDend_date}</b></span>
         ''', unsafe_allow_html = True)

        session['selected_columns'] = selected_columns
        selected_columns = session['selected_columns']
        st.dataframe(selected_columns, use_container_width=True, hide_index=False)

    with st.sidebar:
        with st.expander("Download", expanded=True):
            csv = convert_csv(selected_columns)
            
            excels = convert_excel(selected_columns)
            
            st.download_button(label = "🖨️ Download CSV", 
                                  data = csv,
                                  file_name = f"Report Data CSV ({session['IDstart_date']}) to ({session['IDend_date']}).csv", 
                                  mime = 'text/csv')
            
            st.download_button(label = "🖨️ Download Excel", 
                                  data = excels,
                                  file_name = f"Report Data Excel ({session['IDstart_date']}) to ({session['IDend_date']}).xlsx", 
                                  mime = 'text/xlsx')
                

hide_streamlit = """ <style> footer {visibility: hidden;} </style> """    
st.markdown(hide_streamlit, unsafe_allow_html=True)             