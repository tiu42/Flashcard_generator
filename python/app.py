import os
import streamlit as st
import langchain_helper as lch
import textwrap
import tempfile

st.title("PDF Assistant")

with st.sidebar:
    with st.form(key='myform'):
        uploaded_file = st.file_uploader("File upload", type="pdf")
        submit_button = st.form_submit_button(label="Submit")

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    db = lch.create_vector_db(pdf_path)
    response = lch.get_response(db)
    st.subheader("Answer")
    st.text(textwrap.fill(response, width= 80))