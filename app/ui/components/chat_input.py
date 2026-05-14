"""Modern chatbot-style unified prompt bar."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def _install_enter_to_send_shortcut(send_label: str) -> None:
    components.html(
        f"""
        <script>
        const root = window.parent.document;

        function wirePromptShortcut() {{
            const textareas = root.querySelectorAll('textarea');
            const buttons = root.querySelectorAll('button');

            const textarea = textareas[textareas.length - 1];

            const sendButton = Array.from(buttons).find(
                btn => btn.innerText.trim() === '{send_label}'
            );

            if (!textarea || !sendButton) return;

            if (textarea.dataset.shortcutInstalled === "true") return;

            textarea.dataset.shortcutInstalled = "true";

            textarea.addEventListener("keydown", function(e) {{
                if (e.key === "Enter" && !e.shiftKey) {{
                    e.preventDefault();
                    sendButton.click();
                }}
            }});
        }}

        setTimeout(wirePromptShortcut, 400);
        </script>
        """,
        height=0,
    )


def render_chat_input(
    *,
    text_key: str,
    mic_active: bool,
    disabled: bool = False,
):

    st.markdown('<div class="mega-chatbar"></div>', unsafe_allow_html=True)

    cols = st.columns([1, 0.08, 0.08])

    with cols[0]:
        st.text_area(
            "Message",
            key=text_key,
            placeholder="Ask for songs, describe your mood, or speak...",
            label_visibility="collapsed",
            disabled=disabled,
            height=68,
        )

    with cols[1]:
        mic_clicked = st.button(
            "🎤" if not mic_active else "⏹",
            key="mic_button",
            disabled=disabled,
            use_container_width=True,
        )

    with cols[2]:
        send_clicked = st.button(
            "➤",
            key="send_button",
            disabled=disabled or mic_active,
            use_container_width=True,
        )

    st.markdown('<div class="upload-row">', unsafe_allow_html=True)

    uploaded_audio = st.file_uploader(
        "Upload voice note",
        type=["wav", "mp3", "m4a", "ogg"],
        label_visibility="visible",
        disabled=disabled or mic_active,
    )

    st.markdown('</div>', unsafe_allow_html=True)

    _install_enter_to_send_shortcut("➤")

    return mic_clicked, send_clicked, uploaded_audio