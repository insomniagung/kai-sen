import streamlit as st
import pandas as pd

#program persegi panjang
with st.expander("", expanded=True):
    input1 = st.number_input("Masukkan Panjang", step=0, value=3)
    input2 = st.number_input("Masukkan Lebar", step=0, value=1)
    hasil = input1 * input2
if st.button("Hitung"):
    if input1 and input2:
        # st.write(f"Input 1: {input1}")
        # st.write(f"Input 2: {input2}")
        # st.write(f"Hasil P*L: {hasil}")
        df = pd.DataFrame({
            'Panjang': [input1],
            'Lebar': [input2],
            'Hasil Persegi Panjang (P X L)': [hasil]
        })

        st.write("")
        st.dataframe(df, use_container_width=True, hide_index=True)

    elif input1 == 0 or input2 == 0:
        st.write("Mohon jangan input angka 0.")
    else:
        st.write("Isi kolom kosong.")