# app.py
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from openai import AsyncOpenAI
from agents import (
    Agent, OpenAIChatCompletionsModel, Runner,
    SQLiteSession, set_tracing_disabled,
    set_default_openai_api, set_default_openai_client
)
from agents.mcp import MCPServerStdio, MCPServerStdioParams

# ------------------- Load environment -------------------
load_dotenv(find_dotenv())

def safe_get_env(key: str) -> str | None:
    value = os.getenv(key)
    if not value:
        st.warning(f"⚠️ Environment variable `{key}` is missing in .env")
    return value

# Keys
openai_api_key = safe_get_env("OPENAI_API_KEY")
gemini_api_key = safe_get_env("GEMINI_API_KEY")
firecrawl_api_key = safe_get_env("FIRECRAWL_API_KEY")

# Only set if not None
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Disable tracing to reduce noise
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

# Gemini client as OpenAI-compatible wrapper
if gemini_api_key:
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    set_default_openai_client(client)
else:
    client = None  # fallback if missing

# SQLite session for persistence
# session = SQLiteSession("13", "dbsession.db")

# MCP Firecrawl params
params = MCPServerStdioParams(
    command="npx",
    args=["-y", "firecrawl-mcp"],
    env={"FIRECRAWL_API_KEY": firecrawl_api_key or ""}
)

# ------------------- Core Agent -------------------
async def run_agent(user_input: str):
    async with MCPServerStdio(params=params, client_session_timeout_seconds=20) as mcp_server:
        try:
            assistant = Agent(
                name="WebScraperAgent",
                instructions=(
                    "You are a web scraping agent. "
                    "You can use Firecrawl MCP tools to scrape, search, or extract website content. "
                    "Use multiple tools if needed until you get the right content. "
                    "Always return final content in clean markdown format."
                ),
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model="gemini-2.0-flash-exp",
                    openai_client=client
                ) if client else "gpt-4o-mini"  # fallback if Gemini key missing
            )
            result = await Runner.run(assistant, user_input)
            return result.final_output
        except Exception as e:
            return f"❌ Error in agent: {e}"

# Helper for running async inside Streamlit


# ------------------- Streamlit UI -------------------
st.title("🌐 Web Scraping Agent")
st.write("Scrape websites using Firecrawl MCP + Gemini AI (fallback to GPT if missing).")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("What do you want to scrape?")
if user_input:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent response
    with st.chat_message("assistant"):
        response = asyncio.run(run_agent(user_input))
        st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
