# Importing Libraries
import streamlit as st








def main():
    if "geminiapi" not in st.session_state:
        st.session_state.geminiapi = st.secrets["GEMINI_API_KEY"]

    # Title and description
    st.title("Welcome to Career Decisions with GeminiAPI GPT")
    st.markdown("Helping you make informed career decisions")


    # GitHub stars link
    st.markdown("[![GitHub stars](https://img.shields.io/github/stars/your_username/your_repo.svg?style=social)](https://github.com/FlameCerberus/Career-Decision-Maker-using-GeminiAPI)")

    # Emoji
    st.markdown("<p style='font-size: 24px;'>🌟💼✨</p>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()