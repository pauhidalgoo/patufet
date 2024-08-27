import os
import re
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
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
minute_start_time = datetime.now()

# Counter for tracking progress
processed_examples = 0
total_examples = 0

async def generate_input(instruction, output):
    """Generate an input in Catalan using the Gemini API based on the instruction and output."""
    global requests_count, minute_start_time

    # Construct prompt for API to generate the appropriate input or 'NONE'
    prompt = (f"Based on the following instruction and output, generate a suitable input text in Catalan if necessary. "
              f"Instruction: '{instruction}' Output: '{output}'. "
              "If the instruction and output require an input, provide the most appropriate input text in Catalan. "
              "If no input is needed, respond with 'NONE'. Make sure the response is clear and precise.")

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
            logging.error(f"Error in generating input text. Exception: {e}")
            if "RATE_LIMIT_EXCEEDED" in str(e):
                sleep_time = (timedelta(minutes=1) - (datetime.now() - minute_start_time)).total_seconds()
                logging.warning(f"Rate limit exceeded, waiting for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(5)
            return None

async def process_dataset():
    """Load dataset, filter, and process each example to generate missing inputs."""
    global processed_examples, total_examples
    logging.info("Loading dataset.")
    df = pd.read_csv('alpaCAT/alpaCAT.csv')

    total_examples = len(df)
    tasks = []

    for idx, row in df.iterrows():
        instruction = row['instruction']
        output = row['output']
        input_text = row.get('input', '')

        if pd.isna(input_text):
            # Create task to generate input where it is missing
            tasks.append(process_and_save_example(df, idx, instruction, output))
        else:
            # Save rows that already have inputs
            df.iloc[[idx]].to_csv('alpaCAT/alpaCAT_final.csv', mode='a', index=False, header=not os.path.exists('alpaCAT/alpaCAT_final.csv'))

    await asyncio.gather(*tasks)
    logging.info("All tasks completed.")

async def process_and_save_example(df, idx, instruction, output):
    """Generate and save input for a single example if needed."""
    global processed_examples, total_examples

    # Generate input using the API
    generated_input = await generate_input(instruction, output)

    if generated_input and generated_input.strip().upper() != 'NONE':
        # Update the input column only if a new input is generated
        df.at[idx, 'input'] = generated_input

    # Save incrementally to CSV
    df.iloc[[idx]].to_csv('alpaCAT/alpaCAT_final.csv', mode='a', index=False, header=not os.path.exists('alpaCAT/alpaCAT_final.csv'))

    processed_examples += 1
    progress_percentage = (processed_examples / total_examples) * 100
    print(f"Processed {processed_examples}/{total_examples} examples. Progress: {progress_percentage:.2f}%")
    logging.info(f"Processed {processed_examples}/{total_examples} examples. Progress: {progress_percentage:.2f}%")

async def main():
    """Main function to start the dataset processing."""
    logging.info("Starting dataset processing.")
    await process_dataset()
    logging.info(f"Finished processing dataset.")

if __name__ == "__main__":
    asyncio.run(main())
