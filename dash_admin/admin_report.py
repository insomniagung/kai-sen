import streamlit as st
import time
session = st.session_state
import pandas as pd
import datetime

import matplotlib.pyplot as plt
import xlsxwriter
from io import BytesIO

@st.cache_data(show_spinner=False)
def convert_csv(df):
    return df.to_csv(index=False)

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
    @st.cache_data(show_spinner=False)
    def df_report():
        # df_report = pd.read_csv("data/ulasan_tiket_kai_access.csv")
        # df_report = pd.read_csv("data/df_label_polarity_tanggal.csv")
        df_report = pd.read_csv("data/ulasan_tiket_kai_access_labeling_reviewer.csv")
        return df_report
    
    @st.cache_data(show_spinner=False)
    def df():
        df = pd.read_csv("data/ulasan_tiket_kai_access_labeling_reviewer.csv")
        return df
        
    st.title("Report", help="Halaman laporan dari dataset.")
    st.divider()
    
    df_report = df_report()
    # st.write(len(df_report))
    st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 13px;
             color: white; background-color: #046cd4; 
             border-radius: 20px; padding: 7px 13px;">
             <b>Data berdasarkan tanggal</b></span>
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
            value=minimum_date,
            format="DD/MM/YYYY",
        )
    with col2:
        d2 = st.date_input(
            "Input Tanggal Akhir:",
            min_value = minimum_date,
            max_value = maximum_date,
            value=maximum_date,
            format="DD/MM/YYYY",
        )
    
    start_date = d1.strftime('%Y-%m-%d')
    IDstart_date = d1.strftime('%d %B %Y')
    
    end_date = d2.strftime('%Y-%m-%d')
    IDend_date = d2.strftime('%d %B %Y')
    
    end_date_plus = (d2 + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    with st.expander("Show Data", expanded=True):

        # Convert the date to datetime64
        df_report['at'] = pd.to_datetime(df_report['at'], format='%Y-%m-%d')

        # Filter data between two dates
        filtered_df = df_report.loc[(df_report['at'] >= start_date) & (df_report['at'] <= end_date_plus)]

        sorted_df = filtered_df.sort_values('at', ascending=True)
        selected_columns = sorted_df[['content','label','at']]

        # mengubah format
        selected_columns = selected_columns.copy()        
        selected_columns['at'] = selected_columns['at'].dt.strftime('%d-%m-%Y')
        
        # menghapus index
        # selected_columns.reset_index(drop=True)
        # selected_columns.index += 1
        # selected_columns.sort_values(by='at', ascending=True)

        # Display
        st.write("")
        st.markdown(f'''
             <span style="text-decoration: none;
             font-family: 'Open Sans'; font-size: 13px;
             color: white; background-color: #046cd4; 
             border-radius: 20px; padding: 7px 13px;">
             Data dari <b>{IDstart_date}</b> sampai <b>{IDend_date}</b></span>
         ''', unsafe_allow_html = True)
        
        st.dataframe(selected_columns, use_container_width=True, hide_index=True)
        
        kolom = ['content','label','at']
        selected_columns = selected_columns.loc[:, kolom]
        
    with st.sidebar:
        with st.expander("Download (berdasarkan tanggal)", expanded=False):
            csv = convert_csv(selected_columns)
            excels = convert_excel(selected_columns)
            
            st.download_button(label = "🖨️ Download CSV",
                               data = csv, 
                               file_name = f"Report Data CSV ({IDstart_date}) to ({IDend_date}).csv",
                               mime = 'text/csv')
            
            st.download_button(label = "🖨️ Download Excel",
                               data = excels,
                               file_name = f"Report Data Excel ({IDstart_date}) to ({IDend_date}).xlsx",
                               mime = 'text/xlsx')
            
    # ---------
    st.divider()
    # tabPelayanan, tabQuery = st.tabs(["&nbsp;&nbsp;&nbsp; **Data (Pelayanan)** &nbsp;&nbsp;&nbsp;", 
    #                                   "&nbsp;&nbsp;&nbsp; **Query Data** &nbsp;&nbsp;&nbsp;"])
    # with tabPelayanan:
    # st.write("")
    st.markdown(f'''
         <span style="text-decoration: none;
         font-family: 'Open Sans'; font-size: 13px;
         color: white; background-color: #046cd4; 
         border-radius: 20px; padding: 7px 13px;">
         <b>Data berdasarkan Pelayanan</b></span>
     ''', unsafe_allow_html = True)

    # st.write("")
    menu_pelayanan = ["Pemesanan Tiket","Pembayaran Tiket", "Harga Tiket"]
    select_option = st.selectbox(label="Pilih Pelayanan", options=menu_pelayanan, index=0, label_visibility="collapsed")

    df = df()
    df['at'] = pd.to_datetime(df['at'], format='%Y-%m-%d')
    df['at'] = df['at'].dt.strftime('%d-%m-%Y')        
    if select_option == "Pemesanan Tiket":
        keyword1 = 'pesan tiket'
        key1 = df['content'].str.contains(keyword1, case=False)
        data_keluhan_pemesanan_tiket = key1
        df_keluhan_pemesanan_tiket = df[data_keluhan_pemesanan_tiket]
        df_keluhan_pemesanan_tiket = df_keluhan_pemesanan_tiket.drop_duplicates()

        df_keluhan_pemesanan_tiket = df_keluhan_pemesanan_tiket[['content','label','at']]
        with st.expander("DataFrame", expanded=True):
            # st.write("Total data keluhan pemesanan tiket :", len(df_keluhan_pemesanan_tiket))
            st.dataframe(df_keluhan_pemesanan_tiket.reset_index(drop=True), use_container_width=True)
            pelayanan_df = df_keluhan_pemesanan_tiket.reset_index(drop=True)

    elif select_option == "Pembayaran Tiket":
        keyword1 = 'bayar tiket'
        key1 = df['content'].str.contains(keyword1, case=False)
        
        data_keluhan_pembayaran_tiket = key1
        df_keluhan_pembayaran_tiket = df[data_keluhan_pembayaran_tiket]
        df_keluhan_pembayaran_tiket = df_keluhan_pembayaran_tiket.drop_duplicates()

        df_keluhan_pembayaran_tiket = df_keluhan_pembayaran_tiket[['content','label','at']]
        with st.expander("DataFrame", expanded=True):
            # st.write("Total data keluhan pembayaran tiket :", len(df_keluhan_pembayaran_tiket))
            st.dataframe(df_keluhan_pembayaran_tiket.reset_index(drop=True),use_container_width=True)
            pelayanan_df = df_keluhan_pembayaran_tiket.reset_index(drop=True)

    elif select_option == "Harga Tiket":
        keyword1 = 'harga tiket'
        key1 = df['content'].str.contains(keyword1, case=False)
        # data_keluhan_harga_tiket = key1 | key2
        data_keluhan_harga_tiket = key1
        df_keluhan_harga_tiket = df[data_keluhan_harga_tiket]
        df_keluhan_harga_tiket = df_keluhan_harga_tiket.drop_duplicates()

        df_keluhan_harga_tiket = df_keluhan_harga_tiket[['content','label','at']]
        with st.expander("DataFrame", expanded=True):
            # st.write("")
            # st.write("Total data keluhan harga tiket :", len(df_keluhan_harga_tiket))
            st.dataframe(df_keluhan_harga_tiket.reset_index(drop=True), use_container_width=True)
            pelayanan_df = df_keluhan_harga_tiket.reset_index(drop=True)
            
    # Menghitung jumlah data berdasarkan kata kunci
    st.write("")
    pesan_tiket_count = df[df['content'].str.contains('pesan tiket', case=False)].shape[0]
    bayar_tiket_count = df[df['content'].str.contains('bayar tiket', case=False)].shape[0]
    harga_tiket_count = df[df['content'].str.contains('harga tiket', case=False)].shape[0]

    # Membuat Pie Chart
    labels = ['Pemesanan Tiket', 'Pembayaran Tiket', 'Harga Tiket']
    sizes = [pesan_tiket_count, bayar_tiket_count, harga_tiket_count]
    colors = ['#aca2d0', '#ff948c', '#5ca992']
    explode = (0.1, 0.1, 0)

    total_sizes = sum(sizes)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
    wedgeprops = {'width': 0.7, 'edgecolor': 'white', 'linewidth': 1}
    pie = ax.pie(x=sizes, 
                 # labels=labels, 
                 colors=colors, explode=explode,
                 autopct=lambda pct: "\n{:.1f}%\n".format(pct),
                 textprops={'fontsize': 10, 'color': 'black'}, shadow=True,
                 wedgeprops=wedgeprops)
    ax.legend(pie[0], labels, loc='center left', fontsize=8)
    ax.set_title('(Keluhan Pengguna KAI Access)', fontsize=10, color='white', pad=1)
    ax.set_facecolor('none')
    with st.expander("Pie Chart", expanded=True):
        st.pyplot(fig)
    
    with st.sidebar:
        with st.expander("Download (berdasarkan pelayanan)", expanded=False):
            csv = convert_csv(pelayanan_df)
            excels = convert_excel(pelayanan_df)
            
            st.download_button(label = "🖨️ Download CSV",
                               data = csv, 
                               file_name = f"Report Data CSV Pelayanan.csv",
                               mime = 'text/csv')
            
            st.download_button(label = "🖨️ Download Excel",
                               data = excels,
                               file_name = f"Report Data Excel Pelayanan.xlsx",
                               mime = 'text/xlsx')
    # with tabQuery:
    # ----------------------
    st.divider()
    # st.write("")
    st.markdown(f'''
         <span style="text-decoration: none;
         font-family: 'Open Sans'; font-size: 13px;
         color: white; background-color: #046cd4; 
         border-radius: 20px; padding: 7px 13px;">
         <b>Data berdasarkan Query</b></span>
     ''', unsafe_allow_html = True)

    keyword_input = st.text_input(label="Input Query", 
                                  # label_visibility="collapsed"
                                 )
    # keyword = ''
    if st.button("Cari"):
        if keyword_input:
            keywords = df['content'].str.contains(keyword_input, case=False)
            df_query = df[keywords]
            df_query = df_query.drop_duplicates()
            df_query = df_query[['content','label','at']]
            with st.expander("DataFrame", expanded=True):
                st.write(f"Hasil Query : {keyword_input}")
                st.write("Total : ", len(df_query))
                st.dataframe(df_query.reset_index(drop=True), use_container_width=True)
                query_df = df_query.reset_index(drop=True)
                
            with st.sidebar:
                with st.expander("Download (berdasarkan query)", expanded=False):
                    csv = convert_csv(query_df)
                    excels = convert_excel(query_df)

                    st.download_button(label = "🖨️ Download CSV",
                                       data = csv, 
                                       file_name = f"Report Data CSV Query.csv",
                                       mime = 'text/csv')

                    st.download_button(label = "🖨️ Download Excel",
                                       data = excels,
                                       file_name = f"Report Data Excel Query.xlsx",
                                       mime = 'text/xlsx')            
        else:
            # with st.expander("DataFrame", expanded=True):
            #     st.dataframe(df.reset_index(drop=True), use_container_width=True)
            st.warning("Input Query dahulu.")     
    
        
hide_streamlit = """ <style> footer {visibility: hidden;} </style> """    
st.markdown(hide_streamlit, unsafe_allow_html=True)             