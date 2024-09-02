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
logging.basicConfig(filename='patufet-IT/error_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Limits for API requests
MAX_CONCURRENT_REQUESTS = 90
RATE_LIMIT_PER_MINUTE = 1000

# Initialize control variables
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
requests_count = 0
minute_start_time = datetime.now()

# Counter for tracking progress
processed_examples = 0
total_examples = 0

async def generate_instruction_based_prompt(story):
    """Generate an instruction-based prompt in Catalan using the Gemini API based on the story."""
    global requests_count, minute_start_time

    # Construct a prompt for API to create an instruction based on the story
    prompt = (f"Based on the following Catalan story, create a concise, instruction-based prompt in Catalan that would guide a model to write a new story inspired by the original. "
          f"The instruction should encourage creativity and provide a general direction for the new story's theme or mood, without mentioning specific plot details or elements from the original. "
          f"Focus on creating an open-ended instruction that allows for imaginative freedom while maintaining the spirit of the original story. "
          f"The prompt must be very short and direct."
          f"Story: '{story}'"
          f"Prompt: ")


    async with semaphore:
        # Manage rate limiting
        time_elapsed = datetime.now() - minute_start_time
        if time_elapsed < timedelta(minutes=1):
            if requests_count >= RATE_LIMIT_PER_MINUTE - 5:
                sleep_time = (timedelta(minutes=1) - time_elapsed).total_seconds()
                logging.warning(f"Approaching rate limit. Sleeping for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
                requests_count = 0
                minute_start_time = datetime.now()
        else:
            requests_count = 0
            minute_start_time = datetime.now()
            logging.info("Resetting request count for the new minute.")

        try:
            logging.info(f"Sending API request with prompt (truncated): {prompt[:60]}...")
            response = await model.generate_content_async(prompt)

            if not response or not response.text:
                logging.error("Received an empty response from the API.")
                return None

            requests_count += 1
            logging.info(f"API responded with text of length {len(response.text.split())} tokens.")
            return response.text.strip()
        except Exception as e:
            logging.error(f"Error in generating prompt. Exception: {e}")
            if "RATE_LIMIT_EXCEEDED" in str(e):
                sleep_time = (timedelta(minutes=1) - (datetime.now() - minute_start_time)).total_seconds()
                logging.warning(f"Rate limit exceeded, waiting for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(5)
            return None

async def process_dataset():
    """Load dataset, filter, and process each example to generate instruction-based prompts."""
    global processed_examples, total_examples
    logging.info("Loading dataset.")
    
    # Load the dataset and sample 10,000 stories
    ds = load_dataset("pauhidalgoo/patufet-stories")['train'].shuffle(seed=42).select(range(10000))
    
    total_examples = len(ds)
    tasks = []

    for idx, story in enumerate(ds):
        story_text = story['Story']

        # Create task to generate an instruction-based prompt for each story
        tasks.append(process_and_save_example(story_text, idx))
    
    await asyncio.gather(*tasks)
    logging.info("All tasks completed.")

async def process_and_save_example(story, idx):
    """Generate and save instruction-based prompt for a single story."""
    global processed_examples, total_examples

    # Generate instruction-based prompt using the API
    generated_prompt = await generate_instruction_based_prompt(story)

    if generated_prompt:
        # Save to CSV
        result = {
            'story': story,
            'prompt': generated_prompt
        }

        # Ensure directory exists
        if not os.path.exists('patufet-IT'):
            os.makedirs('patufet-IT')

        df = pd.DataFrame([result])
        df.to_csv('patufet-IT/patufet_stories_prompts.csv', mode='a', index=False, header=not os.path.exists('patufet-IT/patufet_stories_prompts.csv'))

    processed_examples += 1
    progress_percentage = (processed_examples / total_examples) * 100
    print(f"Processed {processed_examples}/{total_examples} stories. Progress: {progress_percentage:.2f}%")
    logging.info(f"Processed {processed_examples}/{total_examples} stories. Progress: {progress_percentage:.2f}%")

async def main():
    """Main function to start the dataset processing."""
    logging.info("Starting dataset processing.")
    await process_dataset()
    logging.info(f"Finished processing dataset.")

if __name__ == "__main__":
    asyncio.run(main())