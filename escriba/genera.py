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
        "causal email" : ["personal updates", "friendly invitations", "social catch-ups", "sharing news", "sharing achievements"],
        "job application email": ["career change", "entry-level positions", "experienced professional roles", "salary negotiation"],
        "inquiry email": ["product information", "service availability", "priding and quotes", "general information requests"],
        "resignation email": ["formal resignation", "immediate resignation", "career transition"],
        "sales email": ["product launch", "special offers", "promotions"],
        "proposal email": ["business proposal", "parnthership opportunities", "grant proposals"],
        "customer service email": ["issue resolution", "feedback requests", "order confirmation", "follow-up assistence"],
        "newsletter email": ["industry news", "event highlights"], 
        "formal email": ["offical notices", "infitation to formal event", "professional requests", "legal and compliance notes", "email to a professor"], 
    },
    "poems": { # different format to account for all types of rimes
        "tipus" : ["sonet", "haiku", "limeric","oda", "elegia", "rondeau", "vers lliure (free verse)", "ballada", "cançó", "poesia lírica", "epigrama", "sàtira", "epopeia", "lletra de cançó (song lyrics)"],
        "sentiments": ["alegria", "tristesa", "por", "nostàlgia", "melancolia", "amor", "odi", "solitud", "tendresa", "esperança", "natura", "identitat", "somni"],
        "rimes": ["tetrasíl·labs (4 versos)", "pentasíl·labs (5 versos)", "hexasíl·labs (6 versos)", "heptasíl·labs (7 versos)", "octosíl·labs (8 versos)", "enneasíl·labs (9 versos)", "decasíl·labs (10 versos)", "hendecasíl·labs (11 versos)", "alexandrins (12 versos)", "any"]
    },
    "essay": {
        "expository essay": ["high school exposition", "social media", "biology", "economy", "benefits and drawbacks", "climate change", "internet", "technology"],
        "descriptive essay": ["nature", "city", "house", "monument", "a place", "an event", "a person", "a book", "a foreign country"],
        "narrative essay": ["a memory", "an experience", "personal adventure", "a journey", "major challenge"], 
        "persuasive essay": ["renewable energy", "the future", "why something is better", "education", "rights"],
        "analytical essay": ["impact of something in society", "literary themes", "film", "politics/economics"],
        "comparative essay": ["urban vs rural", "tradition vs future", "public vs private", "sea vs mountain", "popular debates"], 
        "critical essay": ["new technologies", "historical figures", "literature", "social issues"], 
        "argumentative essay": ["logical resaoning", "refuting counterarguments"],
        "process essay": ["starting a business", "planning a vacation", "learning", "cooking", "coding", "maths"], 
        "personal essay": ["defining moment", "sentiments", "relationships"], 
        "historical essay": ["major wars", "ancient civilizations", "cultural revolutions"], 
        "cause and effect essay": ["ecnonomy", "health", "technology", "society", "events and actions"], 
        "synthesis essay": ["research article", "combining information from various sources", "new perspective on topics"]
    },
    "articles": {
        "news" : ["economic", "sports", "international", "local", "national", "breaking news", "natural disasters", "politics"], 
        "feature": ["human interest stories", "long form investigation", "innovation and technology", "cultural trends"], 
        "editorial": ["policy analysis", "social issues", "education reform", "environment"], 
        "opinion": ["personal reflections", "controversial topics", "lifestyle choices"], 
        "review": ["product", "amazon", "book", "movie", "restaurant", "event"], 
        "how-to": ["DIY project", "tech tutorial", "cooking and recipe", "personal fiance", "fitness"], 
        "profile": ["celebrities", "community leaders", "artists", "historical figures", "professionals"], 
        "interview": ["one on one", "industry expert", "celebrity", "panel interviews", "only questions for an interview"]
    },
    "reports": {
        "business report": ["financial performance", "strategic planning", "human resources"], 
        "technical report": ["system design and architecture", "performance metrics", "problem analysis and solutions", "implementation and testing"],
        "academic report": ["literature review", "research methodology", "results and analysis", "discussion", "research findings", "hypothesis testing"], 
        "market analysis report": ["current trends", "future trends", "consumer demographics"]}
        ,
    "others": {
        "diary entry": ["daily activities", "emotional reflections", "relationships", "dreams", "creative ideas"],
        "speech": ["inspirational", "personal", "educational", "corporate"], 
        "planning": ["free time", "project management", "event", "goal setting", "budgeting"], 
        "ad copy": ["product benefits", "consumer", "brand identity", "call-to-action", "engagement"], 
        "tutorial": ["basic concepts", "step-by-step guide", "creative skills", "software", "troubleshooting", "cooking and recipes", "health and fitness"]
    }
}
c = 0

logging.basicConfig(filename='./escriba/error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding="utf-8")

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")




async def generate_text(text,type, subtype = None, sentiment = None, rima = None):

    prompts = {
        "emails": f"""You are an expert in writting emails and generating conversations. Generate a conversation, in catalan, where the user asks for a {type} about {subtype}. The conversation should only include the user's prompt (with details) and the appropiate email from the assistant (only one turn, try to be direct). The assistant should write an email, and the user can include his data and information in the instruction. Avoid too much markdown formatting, prioritize plain text.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.""",
        "poems": f"""You are an expert in writting poems in catalan and generating conversations. Generate a conversation, in catalan, where the user aks for a {type} (poem) about {sentiment}. The conversation should only include the user's prompt and the response that uses {rima} rime (not necessary that the user specifies it) (only one turn, try to be direct). Avoid too much markdown formatting, prioritize plain text.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme and the poem structure, and in catalan and without errors. Generate just the conversation.""",
        "essay": f"""You are an expert in writting essays in catalan and generating conversations. Generate a conversation, in catalan, where the users asks for a {type} on {subtype}.  The conversation should only include the user's question and the detailed essay as the response (only one turn, try to be direct). Avoid too much markdown formatting, prioritize plain text.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.""",
        "articles": f"""You are an expert in writting articles in catalan and generating conversations. Generate a conversation, in catalan, where the users asks for a {subtype} {type} article.  The conversation should include the user's instruction and the complete output article as the response (only one turn, try to be direct). Avoid too much markdown formatting, prioritize plain text.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.""",
        "reports": f"""You are an expert in writting reports in catalan and generating conversations. Generate a conversation, in catalan, where the user asks for a {type} about {subtype}. The conversation should include what the user asks for (the question) and the complete report (only one turn, try to be direct). Avoid too much markdown formatting, prioritize plain text.
    The output format should be:
    User: [User utterance]
    AI: [AI response]
    Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.""",
        "others": f"""You are an expert in writting in catalan and generating conversations. Generate a conversation, in catalan, where the user asks for a {subtype} {type}.The conversation should include the user's question and the complete response (the written text the user asked) (only one turn, try to be direct). Avoid too much markdown formatting, prioritize plain text.
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
        if text != "poems":
            times = 4 if text != "emails" else 8
            for type, subtypes in topics.items():
                for subtype in subtypes:
                    for i in range(times):
                        tasks.append(generate_limited_text(text, type, subtype))
        else:
            for type in topics["tipus"]:
                for sentiment in topics["sentiments"]:
                    for rima in topics["rimes"]:
                        for i in range(2):
                            tasks.append(generate_limited_text(text, type, sentiment=sentiment, rima = rima))
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        conversa, text, type, subtype, sentiment, rima = await task
        if conversa != None:
            csv_writer.writerow({'Converse': conversa, 'Writing': text, 'Type': type, 'About': subtype, 'Sentiment': sentiment, 'Rima': rima})
            csv_file.flush()
        else:
            total_errors += 1

async def main():
 
    with open('./escriba/synthetic_writtings.csv', 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Converse', 'Writing', 'Type', 'About', 'Sentiment', 'Rima']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        await generate_writings(csv_writer, csv_file)

    print(total_errors)

if __name__ == "__main__":
    asyncio.run(main())
