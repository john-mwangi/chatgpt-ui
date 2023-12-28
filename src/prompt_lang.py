from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from params import CHATGPT_ROLE, models

load_dotenv()

# define prompt_template
template = (
    CHATGPT_ROLE
    + """Answer the question step by step. 
    {conversation_history}
    user: {question}
    assistant:
    """
)

prompt_template = PromptTemplate(
    input_variables=["conversation_history", "question"], template=template
)

# define memory
memory = ConversationBufferMemory(memory_key="conversation_history")

llm_chain = LLMChain(
    llm=ChatOpenAI(model=models[0]),
    prompt=prompt_template,
    memory=memory,
)

if __name__ == "__main__":
    msg = llm_chain.predict(question="Hello")
    print(msg)
