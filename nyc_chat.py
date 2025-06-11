#!/usr/bin/env python3
"""Terminal-based Native New Yorker Chatbot.

Usage:
  python nyc_chat.py

Commands inside chat:
  /exit, /quit, /q   – leave the chat
  /reset             – reset conversation history

Relies on an OpenAI API key loaded from environment or a `.env` file.
"""
from __future__ import annotations

import os
import sys

import openai
from dotenv import load_dotenv
from openai import OpenAI


def initialize_openai() -> OpenAI:
    """Load env vars and return an OpenAI client instance (SDK ≥1.0)."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY is not set. Put it in a .env file or export it.")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def main() -> None:
    client = initialize_openai()

    system_prompt = (
        "You are a true native New Yorker: fast-talking, slightly sarcastic, uses phrases like "
        "\"fuhgeddaboudit\" and \"I’m walkin’ here!\" Keep answers short, witty, and direct."
    )

    # Conversation state: always start with system persona instruction
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("NYC Chatbot (type /exit to quit, /reset to start over)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSee ya! 🎤⬇️")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit", "/q"}:
            print("Catch ya later! 🗽")
            break
        if user_input.lower().startswith("/reset"):
            messages = [{"role": "system", "content": system_prompt}]
            print("Conversation reset. Shoot.")
            continue

        # Add user message and call the model
        messages.append({"role": "user", "content": user_input})
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.9,
            )
        except openai.OpenAIError as exc:
            print(f"[OpenAI error] {exc}")
            # Remove last user message to keep state consistent
            messages.pop()
            continue

        assistant_reply = response.choices[0].message.content.strip()
        print(f"Bot: {assistant_reply}")
        messages.append({"role": "assistant", "content": assistant_reply})

        # Prevent runaway context size: keep system + last 18 messages
        if len(messages) > 20:
            messages = [messages[0]] + messages[-18:]


if __name__ == "__main__":
    main()
