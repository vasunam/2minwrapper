#!/usr/bin/env python3
"""Streamlit UI for the Native New Yorker Chatbot.
Run with:
    streamlit run nyc_chat_streamlit.py
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------------------------------------------------------
# Initialisation helpers
# -----------------------------------------------------------------------------

def init_openai() -> OpenAI:
    """Return an OpenAI client, loading the API key from env/.env."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set. Add it to a .env file.")
        st.stop()
    return OpenAI(api_key=api_key)


def get_persona_prompt() -> str:
    """System prompt defining the New Yorker persona."""
    return (
        "You are a true native New Yorker: fast-talking, slightly sarcastic, uses phrases like "
        "\"fuhgeddaboudit\" and \"I’m walkin’ here!\" Keep answers short, witty, and direct."
    )


# -----------------------------------------------------------------------------
# Chat logic
# -----------------------------------------------------------------------------

def stream_llm_response(client: OpenAI, messages: list[dict[str, str]]):
    """Yield content tokens from the streaming Chat Completions API."""
    response = client.chat.completions.create(
        model="gpt-4o",  # use 3.5-turbo if 4o not available
        messages=messages,
        temperature=0.9,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="NYC Chatbot", page_icon="🗽")
    st.title("🗽 New Yorker Chatbot")
    st.caption("Talk like you're on 7th Ave. Powered by OpenAI.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": get_persona_prompt()},
        ]

    client = init_openai()

    # Display conversation history (skip system message)
    for msg in st.session_state.messages[1:]:
        avatar = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(avatar):
            st.markdown(msg["content"])

    # Chat input box
    if prompt := st.chat_input("Say something..."):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant streamed answer
        with st.chat_message("assistant"):
            stream_container = st.empty()
            answer_accum = ""
            for token in stream_llm_response(client, st.session_state.messages):
                answer_accum += token
                stream_container.markdown(answer_accum)
            st.session_state.messages.append({"role": "assistant", "content": answer_accum})

        # Trim history if too long
        if len(st.session_state.messages) > 40:
            st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-38:]

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Options")
        if st.button("Reset conversation"):
            st.session_state.messages = [
                {"role": "system", "content": get_persona_prompt()},
            ]
            st.experimental_rerun()
        st.markdown("—")
        st.markdown("Made with ❤️ in NYC & Streamlit")


if __name__ == "__main__":
    main()
