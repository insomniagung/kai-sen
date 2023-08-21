import streamlit as st
import pandas as pd
import statistics

# df = pd.read_excel("program.xlsx")
# total = [235,200,215,225]
no = [1,2,3,4]
nama = ['A','B','C','D']
n1 = [70,65,80,75]
n2 = [82,65,70,80]
n3 = [83,70,65,70]
total0 = n1[0] + n2[0] + n3[0]
total1 = n1[1] + n2[1] + n3[1]
total2 = n1[2] + n2[2] + n3[2]
total3 = n1[3] + n2[3] + n3[3]
total = [total0,total1,total2,total3]
# rerata = [78.3,66.6,71.6,75]
rerata = [total0/3, total1/3, total2/3, total3/3]
df = pd.DataFrame({
    'No':no,
    'Nama':nama,
    'N1':n1,
    'N2':n2,
    'N3':n3,
    'Total':total,
    'Rata2':rerata    
})

st.header("Program Menghitung Data")
st.write("")
st.dataframe(df, use_container_width=False, hide_index=True)

# total_rata2 = df["Rata2"].sum() / 4
total_rata2 = df["Rata2"].mean()
# total_rata2 = round(total_rata2,1)
st.write(f"Rata-rata semua: {total_rata2}")

