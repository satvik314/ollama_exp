import chainlit as cl
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOllama

# Initialise the memory and model
memory = ConversationBufferMemory(k=3)
llm = ChatOllama(model="mistral", temperature=1)

# Create the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

@cl.on_message
async def get_response(message: str):
    # Generate the response using LangChain
    response_messages = conversation.predict(input=message)
    
    print(response_messages)
    # Extract the content from each HumanMessage and join them into a single string
    response = ' '.join([message.content for message in response_messages])

    # Send the response back to the user
    await cl.Message(content=response).send()
