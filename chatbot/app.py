from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional: check if API keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond."),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI
st.title('Langchain Demo with OpenAI')
input_text = st.text_input("Search the topic you want")

# LangChain + OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | llm | output_parser

# Run only if user provides input
if input_text:
    response = chain.invoke({'question': input_text})
    st.write(response)
