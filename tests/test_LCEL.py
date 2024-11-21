from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [("user", "Tell me a {adjective} joke")],
)

runnable = prompt | ChatOpenAI(logprobs=True)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    # input_messages_key="input",
    # history_messages_key="history",
)

response_1 = with_message_history.invoke(
    {"adjective": "funny"}, config={"configurable": {"session_id": "abc123"}}
)
print(store)

exit()

legacy_chain = LLMChain(
    llm=ChatOpenAI(logprobs=True),
    prompt=prompt,
    output_parser=StrOutputParser(),
)
response_2 = legacy_chain.invoke({"adjective": "funny"})
print(legacy_chain)
print(response_2)
