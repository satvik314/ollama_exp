import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(model="mistral", temperature=1, streaming= True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    memory = ConversationBufferMemory(return_messages=True)
    chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
            | prompt
            | model
    )

    cl.user_session.set('chain', chain)
    cl.user_session.set('memory', memory)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get('chain')
    memory = cl.user_session.get('memory')
    inputs = {"input": message.content}
    response = chain.invoke(inputs)
    memory.save_context(inputs, {"output": response.content})

    await cl.Message(content=response.content).send()
