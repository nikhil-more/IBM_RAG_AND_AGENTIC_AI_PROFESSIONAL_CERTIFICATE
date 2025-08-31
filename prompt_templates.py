#https://python.langchain.com/api_reference/core/prompts.html

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate, FewShotPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptWithTemplates, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate, StringPromptTemplate, AIMessagePromptTemplate, BaseChatPromptTemplate, BasePromptTemplate, DictPromptTemplate, PipelinePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm_model = None

def instantiate_model(modelName : str):
    try:
        llm_model = OllamaLLM(model=modelName, temperature=0.7)
        return llm_model
    except:
        print("Error creating instance of LLM")
        return None

def chat_prompt_template_example():
    #1. 
    template = """
        You are a helpful personal assistant. 
        Help user creating a routine based on their query

        User Query : {question}"""

    question = "Plan for study of AI on sunday"
    
    prompt = ChatPromptTemplate.from_template(template)
    outputParser = StrOutputParser()
    
    llm_chain = (
        prompt
        | llm_model 
        | outputParser
    )

    response = llm_chain.invoke({"question":question})

    print("\nPrompt : \n", prompt)
    print("\n\nResponse :\n", response)
    print("\n------------------------------------------------\n")

    pass

def chat_message_prompt_template_example():
    pass

def few_shot_prompt_template_example():
    pass

def few_shot_chat_message_prompt_template_example():
    pass

def few_shot_with_template_example():
    pass

def message_placeholder_example():
    pass

def human_message_prompt_template_example():
    pass

def system_message_prompt_template_example():
    pass

def string_prompt_template_example():
    pass

def ai_message_prompt_template_example():
    pass

def base_chat_prompt_template_example():
    pass

def base_prompt_template_example():
    pass

def dict_prompt_template_example():
    pass

def pipeline_prompt_template_example():
    pass


if __name__=="__main__":
    llm_model = instantiate_model(os.getenv("GEMMA_MODEL"))
    if llm_model:
        chat_prompt_template_example()
