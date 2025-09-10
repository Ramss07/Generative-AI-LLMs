from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

# Load API key from .env
load_dotenv()

def generate_pet_name(animal_type, pet_color):
    # Create an LLM wrapper
    llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo-instruct")

    #Define the prompt template and placeholders for input
    prompt_template_name = PromptTemplate(
        input_variables = ['animal_type','pet_color'],
        template = "I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest me five cool names for my pet."
    )

    #Build the chain combining the desired LLM and the prompt template
    name_chain = prompt_template_name | llm

    #Invoke the chain to pass input values into the prompt
    response = name_chain.invoke({'animal_type' : animal_type, 'pet_color' : pet_color})
    return response

def langchain_agent():
    llm = OpenAI(temperature = 0.5)

    tools = load_tools(["wikipedia", "llm-math"], llm = llm)

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    result = agent.invoke(
        "1. Check wikipedia to find the average age of a dog. " \
        "2. Multiply that age by three"
    )

    print(result)

if __name__ == "__main__":
    print(generate_pet_name("pig", "pink"))
    #langchain_agent()
