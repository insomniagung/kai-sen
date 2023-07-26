import streamlit as st
import pandas as pd

df = pd.read_excel("program.xlsx")

st.header("Program Menghitung Data")
st.write("")
st.dataframe(df, use_container_width=False, hide_index=True)

total_rata2 = df["Rata2"].sum() / 4
total_rata2 = round(total_rata2,1)
st.write(f"Rata-rata semua: {total_rata2}")

