import streamlit as st
import pandas as pd
from numerize import numerize

with st.expander("Input",expanded=True):
    harga_beli = st.number_input(label="Masukkan Harga (Rp)", step=0, value=5000)
    diskon = st.number_input(label="Masukkan Diskon (%)", step=0, value=10) 
    jumlah_diskon = harga_beli * (diskon/100)
    jumlah_bayar = harga_beli - jumlah_diskon
    
    #numerize
    # harga_beli = numerize.numerize(harga_beli)
    # jumlah_diskon = numerize.numerize(jumlah_diskon)
    # jumlah_bayar = numerize.numerize(jumlah_bayar)

# with st.expander("Output", expanded=True):
#     st.write("Harga: ", harga_beli)
#     st.write("Diskon: ", diskon, "%")
#     st.write("Jumlah Bayar: ", jumlah_bayar)
if st.button("Hitung"):
    st.write("")
    if harga_beli and diskon:
        # st.write("Harga: ", harga_beli)
        # st.write("Diskon: ", diskon, "%")
        # st.write("Jumlah Bayar: ", jumlah_bayar)
        df=pd.DataFrame({
            "Harga":[f"Rp{harga_beli}"],
            "Diskon":[f"{diskon}% (Rp{jumlah_diskon})"],
            "Jumlah Bayar":[f"Rp{jumlah_bayar}"],
        })
        st.dataframe(df, use_container_width=True,hide_index=True)
    else:
        st.write("Input terlebih dahulu harga dan diskonnya.")