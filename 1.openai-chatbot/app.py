import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A ChatBot with OpenAI"

#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant please response to the user queries"),
        ("user","Question:{question}")
    ]
)
def generate_response(question,api_key,llm,temperature,max_tokens):
    openai.api_key=api_key
    llm=ChatOpenAI(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer


#Title of the app
st.title("Enhanced Q&A Chatbot with OpenAI")

#sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API Key:",type="password")

#drop down to select various Open AI models
llm= st.sidebar.selectbox("Select an Open AI Model ",["gpt-4o","gpt-4-turbo","gpt-4"])

#adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("please provide the query")




