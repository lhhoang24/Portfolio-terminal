import streamlit as st

import charts
import theme

st.set_page_config(page_title="Portfolio Analytics Terminal", page_icon=":material/monitoring:", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)

pages = {
    "Bản hoàn chỉnh": [
        st.Page("terminal.py", title="Portfolio Analytics Terminal", icon=":material/monitoring:", default=True),
        st.Page("assistant.py", title="Portfolio AI Assistant", icon=":material/smart_toy:"),
    ],
}

pg = st.navigation(pages)
pg.run()
