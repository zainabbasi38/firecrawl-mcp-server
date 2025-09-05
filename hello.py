import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, enable_verbose_stdout_logging, SQLiteSession, set_tracing_disabled
from agents.mcp import MCPServerStdio, MCPServerStdioParams, MCPServerStreamableHttp, MCPServerStreamableHttpParams
# enable_verbose_stdout_logging()

_: bool = load_dotenv(find_dotenv())

# URL of our standalone MCP server (from shared_mcp_server)
set_tracing_disabled(True)
gemini_api_key = os.getenv("GEMINI_API_KEY")
session = SQLiteSession("13", "dbsession.db")

#Reference: https://ai.google.dev/gemini-api/docs/openai
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# params = MCPServerStdioParams(
#     command= "/path/to/github-mcp-server",
#     args=["stdio"], 
#     env={ "GITHUB_PERSONAL_ACCESS_TOKEN": "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wh"}

# )

params = MCPServerStdioParams(
    command="npx",
    args= ["-y", "firecrawl-mcp"],
    env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")}
)
async def main(input:str):
    # 1. Configure parameters for the MCPServerStreamableHttp client
    # These parameters tell the SDK how to reach the MCP server.
    async with MCPServerStdio(params=params, client_session_timeout_seconds=20) as github_server:
        # for tool in await github_server.list_tools():
            # print(f"Tool: {tool.name} - {tool.description} \n\n ")

        # lst = await github_server.list_tools()
        # print(len(lst))
        try:
            assistant = Agent(
                name="MyMCPConnectedAssistant",
                instructions="You are web scraper agent. You can use the tools to scrape the content from the web . You can use multiple tools until you get the right content. Always return the final content in markdown format.You can end the turn when you satisfied that the query of user is now fulfilled.",
                mcp_servers=[github_server],
                model=OpenAIChatCompletionsModel(model="gemini-2.0-flash-exp", openai_client=client),
            )

            result = await Runner.run(assistant, input, session=session)
            return result.final_output
            # print(f"Agent result: {result.final_output}")
            
        except Exception as e:
            print(f"An error occurred during agent setup or tool listing: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An unhandled error occurred in the agent script: {e}")