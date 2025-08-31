from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = OllamaLLM(
    model=os.getenv("GEMMA_MODEL"),
    temperature=0.7)

print("\n-------------------------------------------------------------\n")

# Basic Prompt

basic_prompt = """
The wind is
"""

response = llm(basic_prompt)

print("User Prompt : \n", basic_prompt)

print("\nResponse : \n", response)

print("\n-------------------------------------------------------------\n")

# Zero-shot Prompt

zero_shot_prompt = """
Classify the following statement as true or false:
'The Eiffel Tower is located in Berlin.
"""

response = llm(zero_shot_prompt)

print("User Prompt : \n", zero_shot_prompt)

print("\nResponse : \n", response)

print("\n-------------------------------------------------------------\n")

# One-shot Prompt

one_shot_prompt = """
You are a helpful arithmetic assistant
Here is an example for arithmetic expression :
User Query : What is 2*3
Response : 6

Now answer following query
User Query : What 0.9-0.11
"""

response = llm(one_shot_prompt)

print("User Prompt : \n", one_shot_prompt)

print("\nResponse : \n", response)

print("\n-------------------------------------------------------------\n")

# Few-shot Prompt

few_shot_prompt = """
Here are few examples of classifying emotions in statements:
Statement: 'I just won my first marathon!'
Emotion: Joy
Statement: 'I can't believe I lost my keys again.'
Emotion: Frustration
Statement: 'My best friend is moving to another country.'
Emotion: Sadness
Now, classify the emotion in the following statement:
Statement: 'That movie was so scary I had to cover my eyes.'
"""

response = llm(few_shot_prompt)

print("User Prompt : \n", few_shot_prompt)

print("\nResponse : \n", response)

print("\n-------------------------------------------------------------\n")

# Chain of Though Prompt

chain_of_though_prompt = """
Consider the problem: 'A store had 22 apples. They sold 15 apples today a
How many apples are there now?' 
Break down each step of your calculation
"""

response = llm(chain_of_though_prompt)

print("User Prompt : \n", chain_of_though_prompt)

print("\nResponse : \n", response)

print("\n-------------------------------------------------------------\n")

# Self Consistency Prompt

self_consistency_prompt = """
When I was 6, my sister was half of my age. Now I am 70, what age is my sister
Provide three independent calculations and explanations, then determine the age
"""

response = llm(self_consistency_prompt)

print("User Prompt : \n", self_consistency_prompt)

print("\nResponse : \n", response)

print("\n-------------------------------------------------------------\n")