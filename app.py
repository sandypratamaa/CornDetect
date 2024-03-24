import streamlit as st

# Membaca isi dari file index.html
with open('templates/index.html', 'r') as file:
    html_code = file.read()

# Menampilkan HTML di aplikasi Streamlit menggunakan komponen st.components.html()
st.components.v1.html(html_code)
