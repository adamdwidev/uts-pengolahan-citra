import streamlit as st

st.title('Profile')
st.subheader('Nama : Adam Dwi Maulana')
st.subheader('NIM : 312210242')
st.subheader('Kelas : TI.22.B1')
st.subheader('Prodi : Teknik Infomatika')

st.image("./img/adam3.jpg", caption="Adam", use_column_width="Always", width=200)


st.link_button("Instagram", "https://www.instagram.com/adam_webdev/")
st.link_button("Github", "https://github.com/adam-webdev")