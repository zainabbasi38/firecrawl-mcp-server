import streamlit as st
from hello import main
import asyncio
st.title("Web Scraping Agent")
st.write("Scrape content with the help of Firecrawl MCP integration with an AI Agent..")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
user_input = st.chat_input("How can i assist you?")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        
            try:
                result = asyncio.run(main(user_input))  # run async MCP agent
                response = result if result else "⚠️ No response from agent."
            except Exception as e:
                response = f"❌ Error: {e}"

            st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})