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
# NYC Persona Quiz data and helpers
# -----------------------------------------------------------------------------

QUESTIONS = [
    {
        "text": "Your ideal weekend pace?",
        "options": [
            ("Non-stop action, events, and seeing it all.", "Manhattan"),
            ("Exploring cool new spots, art, and unique eats.", "Brooklyn"),
            ("Relaxing with family/friends, maybe a park or diverse food.", "Queens"),
            ("Connecting with the community, local vibes, keeping it real.", "Bronx"),
            ("Chilling at home, quiet, prefer my own space.", "Staten Island"),
        ],
    },
    {
        "text": "When you think 'NYC food,' what's your first craving?",
        "options": [
            ("A fancy meal or the latest trendy restaurant.", "Manhattan"),
            ("Artisanal anything or a cult-favorite food truck.", "Brooklyn"),
            ("Authentic global cuisine from a hidden gem.", "Queens"),
            ("A classic slice or a hearty hero from the corner spot.", "Bronx"),
            ("A good home-cooked meal or reliable local takeout.", "Staten Island"),
        ],
    },
    {
        "text": "How do you feel about crowds?",
        "options": [
            ("Energizing! The more, the merrier.", "Manhattan"),
            ("Okay in specific doses, like a cool concert or market.", "Brooklyn"),
            ("I prefer spaces where I can breathe, but I can handle them.", "Queens"),
            ("Depends on the vibe – a block party is different from Times Square.", "Bronx"),
            ("Honestly, I'd rather avoid them.", "Staten Island"),
        ],
    },
    {
        "text": "Your style is best described as:",
        "options": [
            ("Polished, ambitious, always on-trend or classic.", "Manhattan"),
            ("Unique, vintage, expressive, maybe a little edgy.", "Brooklyn"),
            ("Comfortable, practical, but with personal flair.", "Queens"),
            ("Streetwise, confident, true to myself.", "Bronx"),
            ("Laid-back, casual, no-fuss.", "Staten Island"),
        ],
    },
]

BOROUGH_DESCRIPTIONS = {
    "Manhattan": "You're always on the move and know where the action is. Ambitious, polished, and unfazed by the hustle.",
    "Brooklyn": "You've got an artistic soul and a love for authenticity—from pour-over coffee to underground shows.",
    "Queens": "Diverse, laid-back, and adventurous with food. You appreciate community and comfort in equal measure.",
    "Bronx": "Street-smart, loyal, and full of heart. You keep it real and value local vibes over tourist traps.",
    "Staten Island": "Chill, independent, and a fan of breathing room. You know the value of a good view and some quiet.",
}


def reset_quiz_state() -> None:
    """Reset all session_state variables related to the quiz."""
    st.session_state.quiz_started = False
    st.session_state.quiz_index = 0
    st.session_state.quiz_answers = []
    st.session_state.quiz_result = None


def run_quiz() -> None:
    """Render the quiz UI and handle its state transitions."""
    if "quiz_started" not in st.session_state:
        reset_quiz_state()

    if not st.session_state.quiz_started:
        st.subheader("What Kind of New Yorker Are YOU?")
        st.write("Answer a few quick questions to discover your NYC borough persona!")
        if st.button("Start Quiz", key="start_quiz"):
            st.session_state.quiz_started = True
        return

    # If result already exists, show it
    if st.session_state.quiz_result:
        borough = st.session_state.quiz_result
        st.success(f"You're a {borough} New Yorker! 🗽")
        st.write(BOROUGH_DESCRIPTIONS[borough])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Chat with NYC Bot", key="back_to_chat"):
                st.session_state.mode = "Chat Mode"
                reset_quiz_state()
                st.experimental_rerun()
        with col2:
            if st.button("Retake Quiz", key="retake_quiz"):
                reset_quiz_state()
                st.experimental_rerun()
        return

    # Show current question
    q_idx = st.session_state.quiz_index
    question = QUESTIONS[q_idx]
    st.markdown(f"**Question {q_idx + 1} of {len(QUESTIONS)}**")
    st.write(question["text"])

    option_texts = [opt[0] for opt in question["options"]]
    choice = st.radio("", option_texts, key=f"quiz_q{q_idx}")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Previous", key="prev_q") and q_idx > 0:
            st.session_state.quiz_index -= 1
            st.experimental_rerun()
    with cols[1]:
        if st.button("Next", key="next_q"):
            if choice:
                # Save/update answer
                if len(st.session_state.quiz_answers) > q_idx:
                    st.session_state.quiz_answers[q_idx] = choice
                else:
                    st.session_state.quiz_answers.append(choice)

                if q_idx + 1 < len(QUESTIONS):
                    st.session_state.quiz_index += 1
                    st.experimental_rerun()
                else:
                    # Compute result
                    scores = {b: 0 for b in BOROUGH_DESCRIPTIONS}
                    for qi, answer_text in enumerate(st.session_state.quiz_answers):
                        borough = next(b for text, b in QUESTIONS[qi]["options"] if text == answer_text)
                        scores[borough] += 1
                    st.session_state.quiz_result = max(scores, key=scores.get)
                    st.experimental_rerun()

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

# -----------------------------------------------------------------------------
# Quiz-enabled main (overrides earlier definition)
# -----------------------------------------------------------------------------

def _run_chat() -> None:
    """Original chat interface extracted for reuse."""
    client = init_openai()

    # Display conversation history (skip system message)
    for msg in st.session_state.messages[1:]:
        avatar = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Say something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream_container = st.empty()
            answer_accum = ""
            for token in stream_llm_response(client, st.session_state.messages):
                answer_accum += token
                stream_container.markdown(answer_accum)
            st.session_state.messages.append({"role": "assistant", "content": answer_accum})

        # Trim history
        if len(st.session_state.messages) > 40:
            st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-38:]


def main() -> None:
    """Entry point with Chat vs Quiz toggle."""
    st.set_page_config(page_title="NYC Chatbot", page_icon="🗽")
    st.title("🗽 New Yorker Chatbot")
    st.caption("Talk like you're on 7th Ave. Powered by OpenAI.")

    # Session defaults
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": get_persona_prompt()}]
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat Mode"

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Options")
        st.session_state.mode = st.radio(
            "Mode",
            ("Chat Mode", "NYC Persona Quiz"),
            index=0 if st.session_state.mode == "Chat Mode" else 1,
        )
        if (
            st.session_state.mode == "Chat Mode"
            and st.button("Reset conversation")
        ):
            st.session_state.messages = [{"role": "system", "content": get_persona_prompt()}]
            st.experimental_rerun()
        st.markdown("—")
        st.markdown("Made with ❤️ in NYC & Streamlit")

    # Route
    if st.session_state.mode == "Chat Mode":
        _run_chat()
    else:
        run_quiz()


if __name__ == "__main__":
    main()
