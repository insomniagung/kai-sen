import streamlit as st
import pandas as pd

df = pd.read_excel("sidang.xlsx")

st.dataframe(df, hide_index=True)

a = df['Nilai'] >= 30
# st.write(a)
# if a.sum() == 0:
#     st.write("Hati-hati")
if a.sum() == 0:
    st.write("Tidak ada lebih dari 30")
else:
    st.write("Normal/Ada")

st.write()

# hanya data yang nilainya >= 30
# hanya nilai < 29