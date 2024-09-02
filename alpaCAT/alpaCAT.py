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
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("API Key not found. Ensure GEMINI_API_KEY is set in your environment variables.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Set up logging
logging.basicConfig(filename='alpaCAT/error_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Limits for API requests
MAX_CONCURRENT_REQUESTS = 90
RATE_LIMIT_PER_MINUTE = 1000

# Initialize control variables
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
requests_count = 0
token_usage = 0
minute_start_time = datetime.now()

# Counter for translations created
translations_created = 0
processed_examples = 0  # Global variable to track processed examples count

async def translate_text(text, is_instruction):
    global requests_count, token_usage, minute_start_time

    if is_instruction:
        prompt = f"Translate the following text from English to Catalan in a natural and clear manner, without adding any extra commentary or formatting: {text}. As this is an instruction, please ensure that the translation is clear and concise."
    else:
        prompt = f"Translate the following text from English to Catalan in a natural and clear manner, without adding any extra commentary or formatting: {text}"

    async with semaphore:
        logging.info("Acquired semaphore, preparing to call the API.")

        # Check remaining time for the current minute
        time_elapsed = datetime.now() - minute_start_time
        if time_elapsed < timedelta(minutes=1):
            if requests_count >= RATE_LIMIT_PER_MINUTE - 5:
                sleep_time = (timedelta(minutes=1) - time_elapsed).total_seconds()
                logging.warning(f"Approaching rate limit, sleeping for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
                requests_count = 0
                token_usage = 0
                minute_start_time = datetime.now()
        else:
            requests_count = 0
            token_usage = 0
            minute_start_time = datetime.now()
            logging.info("Resetting requests and token count for the new minute.")

        try:
            logging.info(f"Sending API request with prompt (truncated): {prompt[:60]}...")
            response = await model.generate_content_async(prompt)

            if not response or not response.text:
                logging.error("Received an empty response from the API.")
                return None

            requests_count += 1
            token_usage += len(response.text.split())
            logging.info(f"API responded with {len(response.text.split())} tokens.")
            return response.text.strip()
        except Exception as e:
            logging.error(f"Failed to translate text. Error: {e}")
            
            if "RATE_LIMIT_EXCEEDED" in str(e):
                sleep_time = (timedelta(minutes=1) - (datetime.now() - minute_start_time)).total_seconds()
                logging.warning(f"Rate limit exceeded, waiting for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
                requests_count = 0
                token_usage = 0
                minute_start_time = datetime.now()
            else:
                await asyncio.sleep(5)
            return None

async def process_dataset():
    global translations_created, processed_examples
    logging.info("Loading dataset.")
    print("Loading dataset...")
    ds = load_dataset("yahma/alpaca-cleaned")['train']
    print(f"Total examples in the dataset: {len(ds)}")
    total_examples = len(ds)

    tasks = []
    
    for i, example in enumerate(ds):
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']

        print(f"Creating task for example with instruction: {instruction[:30]}...")
        logging.info(f"Creating task for example with instruction: {instruction[:30]}...")
        tasks.append(process_and_save_example(instruction, input_text, output, total_examples, i))
    
    await asyncio.gather(*tasks)
    logging.info("All tasks completed.")

async def process_and_save_example(instruction, input_text, output, total_examples, example_index):
    global translations_created, processed_examples
    print('Processing example...')
    
    translated_instruction = await translate_text(instruction, is_instruction=True if instruction else False)
    print(f'Translated instruction for example {example_index}.')
    translated_input = await translate_text(input_text, is_instruction=False)
    print(f'Translated input for example {example_index}.')
    translated_output = await translate_text(output, is_instruction=False)
    print(f'Translated output for example {example_index}.')
    
    print('Translated example.')
    if translated_instruction or translated_input or translated_output:
        translations_created += 1
        print('Saving translations...')
        result = {
            'instruction': translated_instruction,
            'input': translated_input,
            'output': translated_output
        }

        # Ensure directory exists
        if not os.path.exists('alpaCAT'):
            os.makedirs('alpaCAT')

        df = pd.DataFrame([result])
        df.to_csv('alpaCAT/alpaCAT.csv', mode='a', index=False, header=not os.path.exists('alpaCAT/alpaCAT.csv'))
    
    processed_examples += 1
    progress_percentage = (processed_examples / total_examples) * 100
    print(f"Processed and saved translations for example. {progress_percentage:.2f}% done. Total translations created: {translations_created}")
    logging.info(f"Processed and saved translations for example")

async def main():
    logging.info("Starting main function.")
    await process_dataset()
    logging.info(f"Finished processing dataset. Total translations created: {translations_created}")

if __name__ == "__main__":
    asyncio.run(main())
