import os
import re
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")

# Set up logging
logging.basicConfig(filename='patufet-summaries/error_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Limits for API requests
MAX_CONCURRENT_REQUESTS = 90
RATE_LIMIT_PER_MINUTE = 525  # Actual quota limit

# Initialize control variables
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
requests_count = 0
token_usage = 0
minute_start_time = datetime.now()

# Counter for summaries created
summaries_created = 0
processed_examples = 0  # Global variable to track processed examples count

# Load datasets and sample data
def load_and_sample_datasets():
    logging.info("Loading datasets.")
    
    # Load datasets
    ds1 = load_dataset("baiges/patufet-QA")['train']
    ds2 = load_dataset("baiges/patufet-stories-prompts")['train']
    
    # Sample datasets
    ds1_sampled = ds1.shuffle(seed=42).select(range(7500))  # Randomly select 7500 samples from 'answer'
    ds2_sampled = ds2.shuffle(seed=42).select(range(2500))  # Randomly select 2500 samples from 'stories'
    
    # Combine and shuffle the datasets
    combined_dataset = ds1_sampled['answer'] + ds2_sampled['story']
    combined_dataset = pd.Series(combined_dataset).sample(frac=1, random_state=42).tolist()  # Shuffle combined dataset
    
    
    logging.info("Datasets loaded and sampled.")
    return combined_dataset

# Function to generate prompts and summaries
async def generate_prompt_and_summary(text):
    global requests_count, token_usage, minute_start_time

    prompt = f"""La teva tasca és crear un prompt en català que demani resumir un text (pot ser der diferents maneres com ara resumeix el text, fes un resum, m'agradaria tenir resumit, explica amb menys paraules...), seguit d'un resum del text proporcionat (pot incloure una frase com ara un resum del text anterior podria ser, el resum sobre el tema x o simplement directament el resum sense introducció). 
    El format ha de ser exactament així: "Prompt: <el teu prompt aquí> Summary: <el resum aquí>". Cal incloure el text original al prompt pot ser entren cometes o directament.
    Aquí tens el text per resumir juntament amb prompt: {text}"""

    async with semaphore:
        logging.info("Acquired semaphore, preparing to call the API.")

        # Check remaining time for the current minute
        time_elapsed = datetime.now() - minute_start_time
        if time_elapsed < timedelta(minutes=1):
            # Check if request count is approaching the limit
            if requests_count >= RATE_LIMIT_PER_MINUTE - 5:  # A buffer to prevent hitting the exact limit
                sleep_time = (timedelta(minutes=1) - time_elapsed).total_seconds()
                logging.warning(f"Approaching rate limit, sleeping for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
                # Reset counters after sleep
                requests_count = 0
                token_usage = 0
                minute_start_time = datetime.now()
        else:
            # Reset counters for a new minute
            requests_count = 0
            token_usage = 0
            minute_start_time = datetime.now()
            logging.info("Resetting requests and token count for the new minute.")

        try:
            logging.info(f"Sending API request with prompt (truncated): {prompt[:60]}...")
            response = await model.generate_content_async(prompt)

            # Check if the response is valid
            if not response or not response.text:
                logging.error("Received an empty response from the API.")
                return None

            requests_count += 1
            token_usage += len(response.text.split())
            logging.info(f"API responded with {len(response.text.split())} tokens.")
            return response.text
        except Exception as e:
            logging.error(f"Failed to generate prompt and summary for the provided text. Error: {e}")
            
            # Handle rate limit errors more gracefully
            if "RATE_LIMIT_EXCEEDED" in str(e):
                sleep_time = (timedelta(minutes=1) - (datetime.now() - minute_start_time)).total_seconds()
                logging.warning(f"Rate limit exceeded, waiting for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
            return None

# Function to extract prompt and summary
def extract_prompt_and_summary(text):
    logging.info("Extracting prompt and summary from the text.")
    
    # Regex pattern to extract "Prompt: ..." and "Summary: ..."
    pattern = re.compile(r'Prompt:\s*(.*?)(?=Summary:|$)Summary:\s*(.*)', re.DOTALL)

    match = pattern.search(text)
    if match:
        prompt_text = match.group(1).strip().replace('**', '\'')  # Replace ** with single quotes
        summary_text = match.group(2).strip().replace('**', '\'') # Replace ** with single quotes
        return prompt_text, summary_text
    else:
        print('Failed to find the prompt and summary in the text.')
        logging.error("Failed to find the prompt and summary in the text.")
        return None, None

# Function to process the dataset and save incrementally
async def process_dataset(dataset):
    global summaries_created, processed_examples  # Track the total number of summaries created and processed examples
    total_examples = len(dataset)  # Total number of examples

    tasks = []
    
    for text in dataset:
        logging.info(f"Creating task for text processing.")
        tasks.append(process_and_save_example(text, total_examples))
    
    await asyncio.gather(*tasks)
    logging.info("All tasks completed.")

async def process_and_save_example(text, total_examples):
    global summaries_created, processed_examples
    response_text = await generate_prompt_and_summary(text)
    
    if response_text:
        prompt, summary = extract_prompt_and_summary(response_text)
        
        if prompt and summary:
            summaries_created += 1  # Increment the summary counter
            result = {
                'prompt': prompt,
                'summary': summary
            }
            print(result)
            # Save the result to CSV
            df = pd.DataFrame([result])
            df.to_csv('patufet-summaries/patufet-summaries-new.csv', mode='a', index=False, header=not os.path.exists('patufet-summaries/patufet-summaries-new.csv'))
    
    processed_examples += 1
    progress_percentage = (processed_examples / total_examples) * 100
    print(f"Processed and saved prompt and summary. {progress_percentage:.2f}% done. Total summaries created: {summaries_created}")
    logging.info(f"Processed and saved prompt and summary.")

async def main():
    logging.info("Starting main function.")
    dataset = load_and_sample_datasets()
    await process_dataset(dataset)
    logging.info(f"Finished processing dataset. Total summaries created: {summaries_created}")

if __name__ == "__main__":
    asyncio.run(main())
