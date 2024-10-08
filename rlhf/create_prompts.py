import os
import re
import google.generativeai as genai
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import asyncio
import aiohttp
import time
import os
import re
import json
import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import google.generativeai as genai
import csv

load_dotenv()

topics = {
    "summarization": ["asking the AI to do text summarization", "asking the AI to do article summarization", "asking the AI to do lecture summarization", "asking the AI to do book summarization", "asking the AI to do legal summarization"],
    "paraphrasing": ["asking the AI to transform formal to informal", "asking the AI to transform a complex to a simple one", "asking the AI to do rephrasing for clarity", "asking the AI to do summarization into bullet points from a text or an article"],
    "sentiment and emotion related": ["asking the AI to perform sentiment analysis"],
    "long context": ["making the AI understand a long prompt", "making the AI summarize a very long text"]

}

c = 0
for k, v in topics.items():
    c += len(v)
print("Total subtopics: ", c)
difficulty_list = ["ultra easy and fast", "very easy", "easy", "a bit difficult", "hard"]


logging.basicConfig(filename='./conversa/error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding="utf-8")

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-pro")




async def generate_prompts(types, topic, difficulty):

    prompt = f"""Create a list of 50 prompts in catalan that a user could ask an AI. All the prompts should ask for or showcase {types} about {topic}, and be {difficulty} to answer. They should include all the necessary information, and they can include making the model choose between options, generate unique content, asking for help or other rellevant instruction styles for the topic.
Follow this template (using - , do not use - anywhere else that isn't the start of the prompt):
- Prompt
- Prompt 
- Prompt 
- Prompt
...
The prompt should include the complete, full text to perform summarization or paraphrazing on. Never use placeholders (like saying "inclou un text" or [...]) and provide the text, not a link to a website. It can be long, if you want. Do not include the answers. It's for an LLM with no vision capabilities (only text).
Remember: follow the specified theme, and in catalan and without errors. Generate just the prompts (do not explain them), but include all the complete text/information that the model would need to answer in them (never use placeholders or [insert text here]). Very important: provide all the text, never under any circumstance using placeholders.
"""
    
    try:
        response = await model.generate_content_async(prompt, safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
})
        return response.text
    except Exception as e:
        logging.error(f"Failed to generate content topic: {topic}, type: {types}, difficulty: {difficulty}. Error: {e}")
        return None

total_errors = 0
async def generate_all_prompts(csv_writer, csv_file):
    global total_errors
    tasks = []

    concurrency_limit = 10
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def generate_limited_prompts(types, topic, difficulty):
        async with semaphore:
            output = await generate_prompts(types, topic, difficulty)
            return output, types, topic, difficulty
        
    
    
    for types, llista in topics.items():
        for topic in llista:
            for difficulty in difficulty_list:
                tasks.append(generate_limited_prompts(types, topic, difficulty))
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        conversa, types, topic, difficulty = await task
        if conversa != None:
            csv_writer.writerow({'Prompts': conversa, 'Types': types, 'Topic': topic, 'Difficulty': difficulty})
            csv_file.flush()
        else:
            total_errors += 1

async def main():
 
    with open('./rlhf/synthetic_prompts2.csv', 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Prompts', 'Types', 'Topic', 'Difficulty']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        await generate_all_prompts(csv_writer, csv_file)

    print(total_errors)

if __name__ == "__main__":
    asyncio.run(main())
