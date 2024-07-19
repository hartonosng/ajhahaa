import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.runnables import Runnable

# Initialize OpenAI
openai.api_key = 'your-openai-api-key'

# Define the schema
details_schema = {
    "type": "object",
    "properties": {
        "Name": {"type": "string"},
        "Age": {"type": "integer"},
        "Occupation": {"type": "string"}
    },
    "required": ["Name", "Age", "Occupation"]
}

# Define the ChatPromptTemplate with a specific schema
template = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please provide the following information: Name, Age, and Occupation for {input_text}."},
        {"role": "assistant", "content": """{
            "Name": "",
            "Age": "",
            "Occupation": ""
        }"""}
    ]
)

# Create an LLMChain with OpenAI
llm = OpenAI(model="text-davinci-003", openai_api_key=openai.api_key)
chain = LLMChain(llm=llm, prompt=template)

# Define a function to use runnable.invoke for more flexibility
class OpenAIInvoker(Runnable):
    def __init__(self, chain):
        self.chain = chain

    def invoke(self, input_text):
        return self.chain.run(input_text=input_text)

# Create an instance of the invoker with the chain
invoker = OpenAIInvoker(chain=chain)

# Define a prompt
prompt = "Tell me about John Doe."

# Get the structured output using runnable.invoke
response = invoker.invoke(prompt)

# Function to parse JSON output
def parse_json_response(response):
    import json
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response"}

structured_output = parse_json_response(response)

print(structured_output)
