from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from chatgpt_ui.configs import template

# define prompt template
prompt_template = PromptTemplate(
    input_variables=["conversation_history", "question"], template=template
)

# define memory
memory = ConversationBufferMemory(
    memory_key="conversation_history",
    ai_prefix="assistant",
    human_prefix="user",
)

if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI

    from chatgpt_ui.configs.params import Settings

    load_dotenv()

    models = Settings.load().models

    llm_chain = LLMChain(
        llm=ChatOpenAI(model=models[0]),
        prompt=prompt_template,
        memory=memory,
    )

    msg = llm_chain.predict(question="Hello")
    print(msg)
