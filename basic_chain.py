from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="openhermes2-mistral", 
            #  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            temperature= 0.9
             )

prompt = PromptTemplate(
    input_variables= ['num', 'topic'],
    template = "Give me {num} controversial tweets on {topic}."
)

tweet_chain = LLMChain(llm = llm, 
                       prompt = prompt,
                       verbose = False)

print(tweet_chain.run(num = "3", topic = "Sam Altman"))