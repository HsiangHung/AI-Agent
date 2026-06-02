import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from helper import TelemetryLog

from dotenv import load_dotenv
load_dotenv()


# Initialize the LLM (ensure it supports structured outputs, like GPT-4o or a local LLM)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
structured_generator = llm.with_structured_output(TelemetryLog)

# The Simulation Prompt
prompt = ChatPromptTemplate.from_template("""
You are a synthetic data generator for a fraud-detection machine learning pipeline.
Generate a realistic JSON telemetry log for the following scenario: {scenario}

Keep these rules in mind:
- If generating a BOT attack, the environment should have LOW ENTROPY: 100% battery, data center ISPs (like AWS or OVH), impossibly fast typing, and mathematically straight mouse movements (entropy near 0.0).
- If generating a HUMAN, the environment should have HIGH ENTROPY: variable battery, residential ISPs (like Comcast), normal typing speeds, and erratic mouse movements (entropy > 0.6).
- If generating a SYNTHETIC IDENTITY (Fake ID), the device might look human, but they take a strangely long time to fill out forms as they copy/paste fake details.
""")

chain = prompt | structured_generator

print("--- Generating Bot Attack Log ---")
bot_log = chain.invoke({
    "scenario": "A headless Chrome bot script attempting to test stolen credit cards on a checkout page. It is running on an AWS server farm."
})
print(bot_log.model_dump_json(indent=2))

print("\n--- Generating Fake ID / Synthetic Identity Log ---")
fake_id_log = chain.invoke({
    "scenario": "A human fraudster manually creating a new bank account using a stolen SSN and a fake name. They are on a normal MacBook but copy-pasting data."
})
print(fake_id_log.model_dump_json(indent=2))

