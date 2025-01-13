import streamlit as st
import pandas as pd
import time
from graph import graph


def response_generator(response):
    for letter in response:
        yield letter
        time.sleep(0.005)


# Streamlit app
st.title("Pharma Knowledge Assistant")

# Maintain chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

messages = st

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        messages.chat_message("user").write(f"{chat['content']}")
    else:
        messages.chat_message("assistant").write(f"{chat['content']}")

# User input
user_input = st.chat_input("Your message")

# if st.button("Send"):
if user_input:
    # Append user input to chat history
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input})

    messages.chat_message("user").write(f"{user_input}")

    # Get LLM response

    inputs = {"messages":[(chat_message['role'],chat_message['content']) for chat_message in st.session_state.chat_history]}
    # response = get_completion_chat(st.session_state.chat_history)
    for output in graph.stream(inputs):
        for key, value in output.items():
            if key=="generate":
                response=value['messages'][0]
    # response = "sample_r"
    # Append assistant response to chat history
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response})

    with messages.chat_message("assistant"):
        st.write_stream(response_generator(response))
    # messages.chat_message("assistant").write(f"{response}")


