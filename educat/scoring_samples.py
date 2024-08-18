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

load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")

dataset = load_dataset("oscar-corpus/OSCAR-2301", "ca", split="train", streaming=True, trust_remote_code=True)

async def fetch_educational_score(session, text, semaphore):
    extract = text[:1500]
    
    prompt = f"""
Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting using the additive 7-point scoring system described below. Take into account that this is just the first words of the page. Be generous, but not overly. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract contains information about a story or a fact, even if it the content lacks structure, depth or completeness.
- Give a third point if the extract is about potentially useful topics, even if they may not be directly related to education or it is in a disorganized or incoherent manner.
- Add a fourth point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, present information in a disorganized manner and incoherent writing style, or be relevant for people specialized in some kind of field.
- Award a fifth point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has limitations like treating concepts that are too complex for grade school students. 
- Grant a sixth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a seventh point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
{extract}

After examining the extract: 
- Briefly justify your total score, up to 50 words.
- Conclude with the score using the format: "Educational score:  <total points>"
"""
    async with semaphore:
        try:
            response = await model.generate_content_async(prompt)
            match = re.search(r"Educational score:\s*(\d+)", response.text)
            score = match.group(1) if match else "0"
        except Exception as e:
            print(f"Error: {e}")
            score = "999"
            response = None
    
    return score, response.text if response else "Error in response"

async def process_batch(examples, semaphore):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_educational_score(session, example["text"], semaphore) 
            for example in examples
        ]
        results = await asyncio.gather(*tasks)
        for example, (score, response_text) in zip(examples, results):
            example["Educational score"] = score
            example["Model response"] = response_text
    return examples


def save_results(data):
    df = pd.DataFrame(data)
    df.to_csv("oscar_educational_scores_responses.csv", mode='a', index=False)

async def main():
    results = []
    batch_size = 1000
    count = 0
    semaphore = asyncio.Semaphore(10) 
    batch = []

    for example in tqdm(dataset, total=100000):
        batch.append(example)
        count += 1
        
        if len(batch) >= batch_size:
            start_time = time.time() 
            results.extend(await process_batch(batch, semaphore))
            save_results(results)
            results = []
            batch = []

            elapsed_time = time.time() - start_time
            print(f"Processed {count} examples so far... ({elapsed_time}s)")

            if elapsed_time < 60:
                await asyncio.sleep(60 - elapsed_time)

        if count >= 100000:
            break

    if batch:
        results.extend(await process_batch(batch, semaphore))
    if results:
        save_results(results)

    print(f"Completed processing {count} examples.")

if __name__ == "__main__":
    asyncio.run(main())