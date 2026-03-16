"""Streamlit chat interface components."""
from __future__ import annotations

from typing import Any

import streamlit as st


def init_session_state() -> None:
    """Initialise chat-related session state keys if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False
    if "result_df" not in st.session_state:
        st.session_state.result_df = None


def render_chat_history() -> None:
    """Display all messages stored in session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def add_message(role: str, content: str) -> None:
    """Append a message and display it immediately."""
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)


def get_user_input() -> str | None:
    """Render the chat input and return the user's text (or None)."""
    return st.chat_input(
        "Describe the data you need (e.g. 'Export all Medline glove spend for last month')",
        disabled=st.session_state.get("awaiting_confirmation", False),
    )
