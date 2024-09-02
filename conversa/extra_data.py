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

    "AI capabilities": ["summarize a long text that the user has sent (include the text by the user).", "answer questions about a text that the user has sent (include the text by the user)",
                        "translate a text that the user gives to it to another language", "identify the key themes in a text sent by the user (include the text by the user)",
                        "generate python code from scratch given a problem", "determine the sentiment from a text the user has sent (include it)", "Deduce conclusions from given information using logical rules.",
                        "Provide clear and understandable explanations for its reasoning and decision-making processes.", "generate a text given some constraints the user gives it",
                        "show a clear example of following specific instructions", "answer a multiple choice question", "retrieve a passage from a document (include it)",
                        "adapt to a persona the user describes to it", "paraphrase some text the user gives it (include it)", "style transfer some text (rewrite it in the suggested style)",
                        "generate a text following some constraints: using specified words, max number of characters..."]

}
c = 0
for k, v in topics.items():
    c += len(v)
print("Total subtopics: ", c)
instruction_styles = ["The AI answers should be long.", ""]
print("Total instruction styles: ", len(instruction_styles))

greetings_list = ["Include greetings", "Do not include greetings"]



logging.basicConfig(filename='./conversa/error_log2.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding="utf-8")

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")




async def generate_conversa(topic, subtopic, instruction_style, greetings):

    prompt = f"""Create a multi-turn conversation, in catalan, between a User and an AI Assistant. The user wants the AI to output something related to {topic}, specifically about {subtopic}.
The intended instructions style for the user are {instruction_style}. Include 2-5 exchanges, (you can include an initial system prompt if needed using System:. {greetings}). The output format should be:
User: [User utterance]
AI: [AI response]
User: [User utterance]
AI: [AI response]
...

Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.
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
        logging.error(f"Failed to generate content topic: {topic}, subtopic: {subtopic}, instruction style: {instruction_style}, greetings: {greetings}. Error: {e}")
        return None

total_errors = 0
async def generate_conversations(csv_writer, csv_file):
    global total_errors
    tasks = []

    concurrency_limit = 10
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def generate_limited_conversa(topic, subtopic, style, greet):
        async with semaphore:
            output = await generate_conversa(topic, subtopic, style, greet)
            return output, topic, subtopic, style, greet
        
    
    
    for topic, llista in topics.items():
        for subtopic in llista:
            for style in instruction_styles:
                for greet in greetings_list:
                    for i in range(4):
                        tasks.append(generate_limited_conversa(topic, subtopic, style, greet))
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        conversa, topic, subtopic, style, greet = await task
        if conversa != None:
            csv_writer.writerow({'Converse': conversa, 'Topic': topic, 'Subtopic': subtopic, 'Style': style, 'Greetings': greet})
            csv_file.flush()
        else:
            total_errors += 1

async def main():
 
    with open('./conversa/synthetic_conversations3.csv', 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Converse', 'Topic', 'Subtopic', 'Style', 'Greetings']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        await generate_conversations(csv_writer, csv_file)

    print(total_errors)

if __name__ == "__main__":
    asyncio.run(main())
