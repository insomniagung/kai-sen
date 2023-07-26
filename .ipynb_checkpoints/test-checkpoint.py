import streamlit as st
import pandas as pd

input1 = st.number_input("Masukkan Panjang", step=0)
input2 = st.number_input("Masukkan Lebar", step=0)
hasil = input1 * input2
if st.button("Hitung"):
    if input1 and input2:
        # st.write(f"Input 1: {input1}")
        # st.write(f"Input 2: {input2}")
        # st.write(f"Hasil P*L: {hasil}")
        df = pd.DataFrame({
            'Panjang': [input1],
            'Lebar': [input2],
            'Hasil (P x L)': [hasil]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    elif input1 == 0 or input2 == 0:
        st.write("Jangan input 0")
    else:
        st.write("Isi kolom kosong.")