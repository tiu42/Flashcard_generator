import os
import streamlit as st
import langchain_helper as lch
import tempfile

st.title("Flashcard Generator")
st.subheader("Your flashcards will be shown here: ")

if 'query' not in st.session_state:
    st.session_state['query'] = ''
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file_name'] = ''
if 'deck' not in st.session_state:
    st.session_state['deck'] = []
if 'index' not in st.session_state:
    st.session_state['index'] = 0
if 'side' not in st.session_state:
    st.session_state['side'] = 'front'

def handle_flip_button():
    if st.session_state['side'] == 'front': st.session_state['side'] = 'back'
    else: st.session_state['side'] = 'front'

def handle_prev_button():
    if st.session_state['index'] != 0:
        st.session_state['side'] = 'front'
        st.session_state['index'] -= 1

def handle_next_button():
    if st.session_state['index'] != 19:
        st.session_state['side'] = 'front'
        st.session_state['index'] += 1

with st.sidebar:
    with st.form(key='myform'):
        openai_api_key = st.text_input(
            label="OpenAI API Key:",
            max_chars=100,
            type="password"
            )
        st.write("###### Don't have one? Get an OpenAI API key *[here.](https://platform.openai.com/account/api-keys)*")
        query = st.text_input("Make flashcards about: ")
        uploaded_file = st.file_uploader("Upload pdf file: ", type="pdf")
        submit_button = st.form_submit_button(label="Submit")

if query and uploaded_file:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if query != st.session_state['query'] and uploaded_file.name != st.session_state['uploaded_file_name']:
        with st.spinner(text="This might take a while..."):
            st.session_state['query'] = query
            st.session_state['uploaded_file_name'] = uploaded_file.name
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            db = lch.create_vector_db(pdf_path)
            response = lch.get_response(query, db)
            st.session_state['deck'] = response['deck']
    deck =  st.session_state['deck']
    cards_num = 20
    index = st.session_state['index']
    side = st.session_state['side']
    with st.container(height=200, border=True):
        st.header(deck[index][side])
    st.session_state['fliped'] = False
    st.write(index+1,"/",cards_num)
    col1, col2, col3 = st.columns(3, gap='large')
    with col1:
        st.button(
            label="Previous",
            on_click= handle_prev_button,
            disabled = (index==0),
            use_container_width = True
        )
    with col2:
        st.button(
            label="Flip",
            on_click= handle_flip_button,
            use_container_width = True
        )
    with col3:
        st.button(
            label="Next",
            on_click= handle_next_button,
            disabled = (index==19),
            use_container_width = True
        )