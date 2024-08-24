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
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


MAX_CONCURRENT_REQUESTS = 30
MAX_REQUESTS_PER_MINUTE = 1000
MAX_TOKENS_PER_MINUTE = 4000000
requests_count = 0
token_usage = 0

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
total_tokens_estimate = 0
minute_start_time = datetime.now()
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")

logging.basicConfig(filename='./python_code/error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding="utf-8")


themes = ["arrays", "strings", "hashing", "linked lists", "stacks",
          "queues", "trees", "graphs", "heaps", "greedy", "binary search",
          "backtracking", "dynamic programming", "system design", "sorting",
          "logical operators", "control clow", "loops", "recursion", "object oriented programming",
          "classes", "graphical user interface", "games", "csv data", "list comprehension",
          "errors and exceptions", "API", "scraping", "web development", "data exploration",
          "data visualization", "parallelization", "concurrency", "special methods",
          "basic mathematics", "advanced mathematics", "working with audio", "working with images",
          "saving files", "sets", "tuples", "dictionaries", "type hinting", "generators",
          "iterables", "mutability", "regular expressions", "itertools", "logging",
          "NumPy", "Pandas", "Scikit-learn", "TensorFlow", "Matplotlib", "Flask", "pytorch",
          'FastAPI', 'SQLAlchemy', 'PySpark', 'AWS SDK', 'the transformers library', 'the datasets library',
          "seaborn", "selenium", "decorators", "inheritance", "pythonic", 
          "efficiency", "modules", "data analysis", "machine learning", "Abductive Reasoning ",
           "causal relationships", "day to day life situations", "basic science",
           "basic finances", "file handling", "complexity", "reflection and instrospection",
           "computer vision", "linear algebra", "statistics", "time series", "geospatial data",
           "clustering algorithms", "asynchronous", "cryptography", "symbolic logic",
           "combinatorics", "puzzles", "world problems", "game theory", "basic arithmetic",
           "counting and summing", "measurement units", "simple fractions", "if/else statements",
           "correcting errors", "fill the gap", "decision making"]


difficulty_levels = ['kids','beginners', 'advanced beginners', 'intermediate', 'advanced', 'experts']


solution_style_prompt = {
        "Text-book": "provide a text-book style correct solution (indicate the python code by using ```python and ```).",
        "Conversational": "provide a conversational style correct solution. (indicate the python code by using ```python and ```)",
        "Step-by-Step": "provide a step-by-step correct solution. (indicate the python code by using ```python and ```)",
        "Only-Code": "provide only the code correct solution, no other explanation. Everything should be code, no text.",
    }


async def generate_exercises(topic, audience):
    prompt = f"""Write 30 Python exercises on the topic of {topic} (focus on it).
They should be aimed at an audience of {audience}, and they should also be educative.
The answers to the exercises should be code snippets, not too long but adequate to the topic and level of the audience. You don't need to include them. Avoid repetitive phrases on the exercises, and try to be creative.
Separate each exercise by using:
- EXERCISE:
(statement)

- EXERCISE:
(statement)
...
(make sure that there is an intro between the "- EXERCISE:" and the statement)
Do not include anything after the last exercise. They must be written in catalan and try to cover all the aspects of the topic. They should resemble a prompt for a model that is going to solve them.
Remember: in catalan, only the statements (no responses) but the responses of the questions should be code, and in the format specified.
"""
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
            response = await model.generate_content_async(prompt)
            requests_count += 1
            exercises = extract_exercise_statements(response.text)
            return exercises
        except Exception as e:
            logging.error(f"Failed to generate exercises for theme: {topic} audience: {audience}. Error: {e}")
            return {}

async def generate_solutions(exercise, topic, audience, style):
    style_description = solution_style_prompt[style]

    prompt = f"""Solve the following Python exercise:
    {exercise}.
    Make sure all your code works, and to comment it in catalan. The intended audience is a {audience}. Don't make mistakes, the code should run perfectly.
    Don't use phrases like "sure, here's your answer"... Provide the response directly.
    Very important: your response should {style_description}.
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
            response = await model.generate_content_async(prompt)
            requests_count += 1
            token_usage += len(response.text.split())
            total_tokens_estimate += len(response.text.split())
            if token_usage > MAX_TOKENS_PER_MINUTE:
                logging.warning(f"Possible Token usage exceeded limit. Current usage: {token_usage} tokens.")
            result = {
                    'exercise': exercise,
                    'solution': response.text,
                    'audience': audience,
                    'topic': topic,
                    'style': style,
                    'prompt': prompt,
                }
            return result
        except Exception as e:
            logging.error(f"Failed to generate content for exercise: {exercise}, topic: {topic}, audience: {audience}, style: {style}. Error: {e}")
            return None


def extract_exercise_statements(text):
    lines = text.splitlines()

    exercises = []
    current_exercise = ""
    first = True

    exercise_start_pattern = re.compile(r'^\s*[-*]*\s*[-*]*(EXERCICI|EXERCISE)\s*[:-]*[-*]*\s*', re.IGNORECASE)
    for line in lines:
        if exercise_start_pattern.match(line):
            if current_exercise:
                if first:
                    current_exercise = ""
                else:
                    exercises.append(current_exercise.strip())
                    current_exercise = ""
                first = False
        else:
            current_exercise += " " + line.strip()
    
    if current_exercise:
        exercises.append(current_exercise.strip())

    return exercises

async def save_results(data):
    df = pd.DataFrame(data)
    print("Another exercise done!")
    file_path = "./python_code/synthetic_code.csv"
    if os.path.isfile(file_path):
        df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        df.to_csv(file_path, mode='w', index=False, header=True)

async def process_topic( topic, audience):
    exercises = await generate_exercises(topic, audience)

    results = []

    tasks = []
    for style in solution_style_prompt.keys():
        for exercise in exercises:
            task = asyncio.create_task(
            generate_solutions(exercise, topic, audience, style))
            tasks.append(task)
    
    try:
        done, pending = await asyncio.wait(tasks)
        contents = [task.result() for task in done]

        results = [e for e in contents if e != None]

        await save_results(results)
    except Exception as e:
        logging.error(f"Failed to generate content for topic: {topic}, audience: {audience}. Error: {e}")

async def main():
    tasks = []
    for topic in themes:
        for audience in difficulty_levels:
            tasks.append(process_topic(topic, audience))
        await asyncio.gather(*tasks)
        tasks = []
    print(total_tokens_estimate)

if __name__ == "__main__":
    asyncio.run(main())