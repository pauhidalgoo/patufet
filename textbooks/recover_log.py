import os
import re
import json
import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")

logging.basicConfig(filename='./textbooks/recover_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MAX_CONCURRENT_REQUESTS = 90
MAX_REQUESTS_PER_MINUTE = 1000
MAX_TOKENS_PER_MINUTE = 4000000

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

total_tokens_estimate = 0
token_usage = 0
requests_count = 0
minute_start_time = datetime.now()

def load_topics(filename='./textbooks/topics.json'):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config

def load_logfile(filename='./textbooks/error_log.txt'):
    with open(filename, 'r', encoding="utf-8") as file:
        log = file.readlines()
    return log

audiences = {
    "General": "provide a comprehensive explanation suitable for a general audience. Include examples and exercises where relevant.",
    "Kid": "explain the topic in simple language and include engaging stories or analogies. Make the content fun and easy to understand.",
    "High-School": "offer a detailed explanation with examples and exercises appropriate for high-school students. Ensure clarity and rigor in the explanations.",
    "College": "provide an in-depth and rigorous explanation with detailed examples and exercises suitable for college-level students. Include advanced concepts where relevant.",
    "Researcher": "engage a highly knowledgeable audience with deep expertise. Include analysis of recent research, advanced theories, and detailed examples."
}

async def generate_chapters(field, topic, subtopic, audience):
    units_prompt = f"""Create the units and subunits for an imaginary textbook for the topic "{field} - {topic}: {subtopic}" intended for a {audience} audience. Focus on this topic. The textbook is in catalan, but you can use your knowledge in english. The output should have the following format:
EXAMPLE:
    
1. Introducció a la biologia molecular
1.1 Introducció
2. Estructura i funció de les biomolècules
2.1 Proteïnes: estructura primària i funcions biològiques
2.2 Estructura tridimensional de les Proteïnes
2.3 Relació estructura-funció i evolució de Proteïnes
2.4 Caracterització estructural de macromolècules

3. Properties of Proteins
3.1 Catàlisi
3.2 Enzims
3.3 Cinètica enzimàtica
3.4 Transducció de senyals
3.5 Receptors
4. Projectes i activitats
... You must only write this index, do not in any case provide any type of explanation. At most, the output should contain 25 sub-units, don't make the index too long. Remember the audience you are aiming for, and to just output the units in Exactly the same format provided (same way of indexing). 
Very important: the output style should match exactly the example."""
    
    async with semaphore:
        global requests_count, minute_start_time, token_usage
        if datetime.now() - minute_start_time >= timedelta(minutes=1):

            requests_count = 0
            token_usage = 0
            minute_start_time = datetime.now()

        if requests_count >= MAX_REQUESTS_PER_MINUTE:
            sleep_time = timedelta(minutes=1) - (datetime.now() - minute_start_time)
            await asyncio.sleep(sleep_time.total_seconds())
            requests_count = 0
            token_usage = 0
            minute_start_time = datetime.now()

        try:
            response = await model.generate_content_async(units_prompt, safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
})
            requests_count += 1
            units = extract_units(response.text)
            return units
        except Exception as e:
            logging.error(f"Failed to generate chapters for field: {field}, topic: {topic}, subtopic: {subtopic}, audience: {audience}. Error: {e}")
            return {}

async def generate_content(field, topic, subtopic, chapters, current_chapter, current_subunits, current, audience):
    audience_description = audiences[audience]
    prompt = f"""Write a detailed, very long and comprehensive textbook chapter on the topic of '{topic}-{subtopic}' under '{field}'. The previous chapter(s) that have already been covered are: {chapters}. 
    The current chapter is called {current_chapter}, and we have written the following part(s) of it: {current_subunits}. You are going to be writing the sub-unit titled {current}. 
    Create it while trying to provide an in-depth explanation, be rigorous, engaging and avoiding incorrect information. You can use the knowledge you have in English, but the text must be in Catalan. The content should be targeted to a {audience} audience, so {audience_description}.
    Include any examples, exercises (solved), proofs, detailed analyses, equations, dates, key events, names, places... relevant to the chapter. Avoid including a conclusion. Do not make orthography errors.
    Do not include a headline, title, introduction, nor indications, simply write it. Make it more narrative and like a real-life book. Prioritize in-depth explanation to exercises. Don't use **, #... . The language of the textbook must be in Catalan: do not include any word in Spanish and make sure what you write is correct. Do not include "[Continuarà]" or similar things
    Remember: in-depth explanations, detailed, very long, narrative and comprehensive.
"""
    
    async with semaphore:
        global requests_count, token_usage, minute_start_time, total_tokens_estimate
        if datetime.now() - minute_start_time >= timedelta(minutes=1):
            requests_count = 0
            token_usage = 0
            minute_start_time = datetime.now()

        if requests_count >= MAX_REQUESTS_PER_MINUTE:
            sleep_time = timedelta(minutes=1) - (datetime.now() - minute_start_time)
            await asyncio.sleep(sleep_time.total_seconds())
            requests_count = 0
            token_usage = 0
            minute_start_time = datetime.now()

        try:
            response = await model.generate_content_async(prompt, safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
})
            requests_count += 1
            token_usage += len(response.text.split())
            total_tokens_estimate += len(response.text.split())
            if token_usage > MAX_TOKENS_PER_MINUTE:
                logging.warning(f"Possible Token usage exceeded limit. Current usage: {token_usage} tokens.")
            result = {
                    'text': response.text,
                    'field': field,
                    'topic': topic,
                    'subtopic': subtopic,
                    'chapter': current_chapter,
                    'subunit': current,
                    'audience': audience
                }
            await save_results([result])
            return result
        except Exception as e:
            logging.error(f"Failed to generate content for field: {field}, topic: {topic}, subtopic: {subtopic}, chapter: {current_chapter}, subunit: {current}. Error: {e}")
            return None

def extract_units(text):
    units = {}
    current_unit = None
    current_subunits = []
    
    lines = text.splitlines()
    
    for line in lines:
        unit_match = re.match(r'^\s*(\d+)\.\s+([^\d].*?)\s*$', line)
        subunit_match = re.match(r'^\s*(\d+\.\d+)\.?\s+(.+?)\s*$', line)
        
        if unit_match:
            if current_unit:
                units[current_unit] = current_subunits
            
            current_unit = unit_match.group(2)
            current_subunits = []
        
        elif subunit_match and current_unit:
            current_subunits.append(subunit_match.group(2))
    
    if current_unit:
        units[current_unit] = current_subunits
    
    return units

async def save_results(data):
    df = pd.DataFrame(data)
    print("Another topic done!")
    df.to_csv("./textbooks/synthetic_textbooks.csv", mode='a', index=False)

async def process_topic(field, topic, subtopic, audience):
    chapters = await generate_chapters(field, topic, subtopic, audience)
    done_chapters = []
    results = []

    tasks = []
    for chapter, subunits in list(chapters.items()):
        done_subunits = []
        for subunit in subunits:
            print(f"Creating task for chapter: {chapter}, subunit: {subunit}")
            task = asyncio.create_task(
            generate_content(field, topic, subtopic, done_chapters, chapter, done_subunits, subunit, audience))
            tasks.append(task)
            done_subunits.append(subunit)
        done_chapters.append(chapter)
    
    try:
        done, pending = await asyncio.wait(tasks)
        contents = [task.result() for task in done]

        results = [e for e in contents if e != None]

        await save_results(results)
    except Exception as e:
        logging.error(f"Failed to generate content for field: {field}, topic: {topic}, subtopic: {subtopic}. Error: {e}")

async def retry_failed_configurations():
    log = load_logfile()
    current_audience = None

    for line in log:
        audience_match = re.search(r'Audience:\s+(\w+)', line)
        if audience_match:
            current_audience = audience_match.group(1)

        if current_audience not in ["General", "Kid"]:
            if current_audience == "High":
                current_audience = "High-School"
        
            subtopic_error_match = re.search(r'Failed to generate content for field:\s+(.+?),\s+topic:\s+(.+?),\s+subtopic:\s+(.+?). Error:', line)
            chapter_error_match = re.search(r'Failed to generate content for field:\s+(.+?),\s+topic:\s+(.+?),\s+subtopic:\s+(.+?),\s+chapter:\s+(.+?),\s+subunit:\s+(.+?). Error:', line)

            if chapter_error_match:
                field, topic, subtopic, chapter, subunit = chapter_error_match.groups()
                print(f"Retrying chapter-level error for {field} - {topic}: {subtopic}, chapter: {chapter}, subunit: {subunit}")
                await generate_content(field, topic, subtopic, [], chapter, [], subunit, current_audience)

            elif subtopic_error_match:
                field, topic, subtopic = subtopic_error_match.groups()
                print(f"Retrying subtopic-level error for {field} - {topic}: {subtopic}")
                await process_topic(field, topic, subtopic, current_audience)

async def main():
    await retry_failed_configurations()

if __name__ == "__main__":
    asyncio.run(main())
