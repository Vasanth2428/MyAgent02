import os
from dotenv import load_dotenv

# Load env
load_dotenv()

print("GROQ_API_KEY from os.getenv:", os.getenv("GROQ_API_KEY"))
print("GROQ_CORE_KEY from os.getenv:", os.getenv("GROQ_CORE_KEY"))
print("AGENT_API_KEY from os.getenv:", os.getenv("AGENT_API_KEY"))

from src.core.llm import LLMService
llm = LLMService()
print("LLMService is_mock:", llm.is_mock)

try:
    import langchain_groq
    print("ChatGroq class name:", langchain_groq.ChatGroq.__name__)
except Exception as e:
    print("Could not import ChatGroq:", e)
