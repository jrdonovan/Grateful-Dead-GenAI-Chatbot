from dotenv import load_dotenv
from dateutil import parser
import asyncio
import itertools
import internetarchive
from youtube_search import YoutubeSearch
import json
import os
import streamlit as st

# Langchain libs
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.globals import set_verbose
from langchain_core.tools import tool

@tool
async def get_gd_setlist(date):
    """Gets a Grateful Dead setlist on the provided date"""
    date = parse_input_as_date(date)
    try:
        loader = JSONLoader(
            file_path=f"./data/setlists/grateful_dead/{date.year % 100}/{date.strftime('%m.%d.%y')}.json",
            jq_schema=".setlist",
            text_content=False
        )
        return loader.load()
    except FileNotFoundError:
        return f"No Grateful Dead setlist found for {date.strftime('%B %d, %Y')}"


def parse_input_as_date(input_string):
    try:
        return parser.parse(input_string)
    except ValueError:
        return "Couldn't parse out which date was being referenced"


async def get_reviews_from_archive(item_id):
    try:
        item = await asyncio.to_thread(internetarchive.get_item, item_id)
    except Exception:
        return f"Failed to retrieve data from internetarchive for {item_id}"

    reviews = []
    for review in item.reviews:
        title = review.get('reviewtitle', 'No Title')
        comment = review.get('reviewbody', 'No Content')
        stars = review.get('stars', 'No Stars')
        reviews.append({"Title": title, "Stars": stars, "Review": comment})
    return reviews


@tool
async def fetch_grateful_dead_show(date):
    """Fetches the details of a Grateful Dead show from a given date."""
    date = parse_input_as_date(date)
    items = []
    with open("./data/internetarchive_gd_items.csv", 'r') as file:
        for line in file:
            if date.strftime("%Y-%m-%d") in line or date.strftime("%y-%m-%d") in line:
                items.append(line.strip())
        if len(items) == 0:
            return f"No data found for {date}."
    tasks = [get_reviews_from_archive(i) for i in items]
    results = await asyncio.gather(*tasks)
    reviews = list(itertools.chain(*results))
    return reviews

@tool
async def search_youtube(search_terms):
    """Searches YouTube using the provided query and returns a list of URLs that link to the results of the search"""
    results = YoutubeSearch(search_terms).to_json()
    data = json.loads(results)
    return [
        {
            "title": video["title"],
            "url": "https://www.youtube.com" + video["url_suffix"]
        }
        for video in data["videos"]
    ]


async def main():
    tools = [fetch_grateful_dead_show, get_gd_setlist, search_youtube]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
                You are a Deadhead who has been to hundreds of Grateful Dead shows over the years.
                Use the reviews retrieved from the provided data sources to summarize the Grateful Dead show on the provided date by the user.
                If a user does not provide a specific date, do your best to intuitively pick the show based on the user's input.
                Return the information in the following format: General Summary, Highlights, Average rating from the reviews, Entire Setlist, & Links to up to 5 Youtube videos if you can find any.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    history = StreamlitChatMessageHistory(key="chat_messages")
    if history.messages == []:
        history.add_ai_message("Ask me about a show in the history of the Grateful Dead")
    for msg in history.messages:
        st.chat_message(msg.type, avatar=("https://lh5.googleusercontent.com/proxy/3XnE3SV_w-N5ZfRDAjkfjLbjZID_HtzLbNPHhD-is6TxcrEBaHx5DHH43lbm34E" if msg.type == "ai" else None)).write(msg.content)

    with st.sidebar:
        st.title('Deadbot')
        st.write('This chatbot summarizes information on Grateful Dead shows using online reviews & setlists')

        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a model', ['gpt-4o','gpt-4o-mini','gpt-4','gpt-35-turbo'], key='selected_model')
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)

    if input := st.chat_input():
        st.chat_message("user").write(input)

        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=temperature,
            model_name=selected_model,
            top_p=top_p
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)

        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: history,
            input_messages_key="input",
            history_messages_key="history",
        )
        with st.status("Looking for information on the requested show...") as status:
            config = {"configurable": {"session_id": "any"}}
            response = await agent_with_chat_history.ainvoke({"input": input}, config)
            status.update(
                label="Here's what I found", state="complete", expanded=False
            )
        st.chat_message("ai", avatar="https://lh5.googleusercontent.com/proxy/3XnE3SV_w-N5ZfRDAjkfjLbjZID_HtzLbNPHhD-is6TxcrEBaHx5DHH43lbm34E").write(response["output"])


if __name__ == "__main__":
    set_verbose(True)
    load_dotenv()

    st.set_page_config(page_title="Deadbot")

    asyncio.run(main())
