# The goal is to have emails + poems + reports + essays + articles

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

types = {
    "emails": {
        "causal email": ["personal updates", "friendly invitations", "social catch-ups", "sharing news", "sharing achievements"],
        "professional email": ["networking introductions", "job applications", "client follow-ups", "meeting summaries", "business proposals", "talking to a professor", "talking to a superior"],
        "marketing email": ["promotional offers", "product launches", "newsletter updates", "event announcements", "customer feedback requests"],
        "educational email": ["course updates", "assignment feedback", "exam reminders", "study group invitations", "academic announcements"]
    },
    "linkedin posts": {
        "professional achievements": ["career milestones", "certifications earned", "new job announcements", "project completions", "speaking engagements", "competition winners", "competition participation", "post with emojis"],
        "industry insights": ["market trends", "thought leadership", "best practices", "case studies", "industry events"],
        "company updates": ["new hires", "product releases", "corporate social responsibility initiatives", "partnership announcements", "company milestones"],
        "personal branding": ["work-life balance tips", "professional development", "leadership advice", "career growth stories", "motivational content"]
    },
    "tweets": {
        "personal thoughts": ["daily reflections", "funny observations", "random musings", "shout-outs", "book or movie recommendations", "memes", "jokes"],
        "news reactions": ["breaking news commentary", "political opinions", "pop culture reactions", "sports events", "global events"],
        "social activism": ["raising awareness", "advocacy for causes", "community support", "charitable initiatives", "environmental concerns"],
        "tech updates": ["new app features", "product reviews", "coding tips", "gadget news", "startup announcements"]
    },
    "whatsapp messages": {
        "family chat": ["family updates", "event planning", "birthday reminders", "holiday plans", "photo sharing"],
        "friend groups": ["weekend plans", "funny memes", "inside jokes", "movie night ideas", "vacation planning", "helping a friend", "emotional support"],
        "work groups": ["meeting reminders", "project updates", "task assignments", "deadline discussions", "office gossip"],
        "community groups": ["event coordination", "local news", "charity drives", "volunteer opportunities", "shared resources"]
    }
}

c = 0

logging.basicConfig(filename='./escriba/error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding="utf-8")

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-pro")




async def generate_text(text,type, subtype = None, sentiment = None, rima = None):

    prompts = {
        "emails": f"""You are an expert in writting emails and generating conversations. Generate a conversation, in catalan, where the user asks for a {type} about {subtype}. The conversation should only include the user's prompt (with details) and the appropiate email from the assistant (only one turn, try to be direct). The assistant should write an email, and the user should the data data and information he wants in the instruction. Avoid too much markdown formatting, prioritize plain text.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.""",
    "linkedin posts": f"""You are an expert in crafting professional LinkedIn posts and generating engagement. Generate a conversation, in catalan, where the user asks for a {type} post about {subtype}. The conversation should only include the user's prompt (with details) and the appropriate LinkedIn post from the assistant (only one turn, try to be impactful). The assistant should write the post, and the user should provide the necessary data and information in the instruction. You can use emojis if the user asks for it.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.""",
    "tweets": f"""You are an expert in composing tweets and generating social media engagement. Generate a conversation, in catalan, where the user asks for a {type} tweet about {subtype}. The conversation should only include the user's prompt (with details) and the appropriate tweet from the assistant (only one turn, try to be concise). The assistant should write the tweet, and the user should provide the necessary data and information in the instruction. Avoid too much markdown formatting, prioritize plain text.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.""",
    "whatsapp messages": f"""You are an expert in writing WhatsApp messages and creating engaging conversations. Generate a conversation, in catalan, where the user asks for a {type} message about {subtype}. The conversation should only include the user's prompt (with details) and the appropriate WhatsApp message from the assistant (only one turn, try to be natural and friendly). The assistant should write the message, and the user should provide the necessary data and information in the instruction if needed. Avoid too much markdown formatting, prioritize plain text.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation."""
    }
    prompt = prompts[text]
    
    try:
        response = await model.generate_content_async(prompt, safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
})
        return response.text
    except Exception as e:
        logging.error(f"Failed to generate content {text}. Type: {type}, Subtype: {subtype}, Sentiment: {sentiment}, Rime: {rima} Error: {e}")
        return None

total_errors = 0
async def generate_writings(csv_writer, csv_file):
    global total_errors
    tasks = []

    concurrency_limit = 10
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def generate_limited_text(text, type, subtype = None, sentiment = None, rima = None):
        async with semaphore:
            output = await generate_text(text, type, subtype, sentiment, rima)
            return output, text, type, subtype, sentiment, rima
        
    
    
    for text, topics in types.items():
        for type, subtypes in topics.items():
            for subtype in subtypes:
                for i in range(8):
                    tasks.append(generate_limited_text(text, type, subtype))
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        conversa, text, type, subtype, sentiment, rima = await task
        if conversa != None:
            csv_writer.writerow({'Converse': conversa, 'Writing': text, 'Type': type, 'About': subtype, 'Sentiment': sentiment, 'Rima': rima})
            csv_file.flush()
        else:
            total_errors += 1

async def main():
 
    with open('./escriba/synthetic_writtings2.csv', 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Converse', 'Writing', 'Type', 'About', 'Sentiment', 'Rima']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        await generate_writings(csv_writer, csv_file)

    print(total_errors)

if __name__ == "__main__":
    asyncio.run(main())
