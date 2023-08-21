import streamlit as st
import datetime
from dateutil.relativedelta import relativedelta

nama = st.text_input("Masukkan Nama:")
d = st.date_input("Tanggal Lahir", datetime.date(2000, 8, 14),
                 # format="DD/MM/YYYY",
                 )

# st.write('Your birthday is:', d)

today = datetime.datetime.now().date()

# st.write('Today is:', today)

if st.button("Hasil"):
    if nama and d:
        with st.expander("", expanded=True):
            delta = relativedelta(today, d)
            st.write(f"Selamat Pagi {nama}. Umur anda:")
            st.write(f"{delta.years} Tahun, {delta.months} Bulan")
            # st.write(f"{delta.years} Tahun, {delta.months} Bulan, {delta.days} Hari.")
