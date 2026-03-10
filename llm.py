import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are a repository reproduction agent.

Goal:
Understand how to run the repository based on README and project files.

Respond ONLY with JSON:

{
 "action": "run | read | list | finish",
 "args": {...},
 "explanation": "reason"
}

Rules:
- Inspect README first
- Prefer creating a python virtual environment
- Install dependencies
- Run experiments/tests
- Fix errors if commands fail
"""


def ask_llm(messages):

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content