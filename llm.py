import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)


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

do not add any other text, information, etc. Provide in response pure JSON with structure given above.

Rules:
- Inspect README first
- Prefer creating a python virtual environment
- Install dependencies
- Run experiments/tests
- Fix errors if commands fail
"""


def ask_llm(messages):

    response = client.chat.completions.create(
        model="anthropic/claude-4-opus",
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content