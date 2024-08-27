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
logging.basicConfig(filename='patufet-QA/error_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Limits for API requests
MAX_CONCURRENT_REQUESTS = 90
RATE_LIMIT_PER_MINUTE = 1500  # Actual quota limit

# Initialize control variables
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
requests_count = 0
token_usage = 0
minute_start_time = datetime.now()

# Counter for questions created
questions_created = 0
processed_examples = 0  # Global variable to track processed examples count

# Function to generate question-answer pairs
async def generate_qa_pairs(text):
    global requests_count, token_usage, minute_start_time

    prompt = f"""Your task is to create different questions in order to create a very high-quality questions-answers dataset in Catalan. 
    You need to ask very valuable questions (as many as there are with very high quality and if there is nothing to ask just say NONE).
    If you do have interesting questions, answer in Catalan in the format: 'Pregunta x: Resposta x:'. 
    **Important**: The questions should be general and answerable without direct reference to the specific context provided. 
    This is the context: {text}"""
    
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
            logging.error(f"Failed to generate Q&A for the provided text. Error: {e}")
            
            # Handle rate limit errors more gracefully
            if "RATE_LIMIT_EXCEEDED" in str(e):
                sleep_time = (timedelta(minutes=1) - (datetime.now() - minute_start_time)).total_seconds()
                logging.warning(f"Rate limit exceeded, waiting for {sleep_time:.2f} seconds.")
                await asyncio.sleep(sleep_time)
            return None

# Improved function to extract questions and answers
def extract_qa_pairs(qa_text):
    logging.info("Extracting QA pairs from the text.")
    qa_pairs = []
    
    # Improved pattern to handle case insensitivity and missing punctuation
    qa_pattern = re.compile(r'(pregunta\s*\d*:\s*)(.+?)(?:resposta\s*\d*:\s*)(.+?)(?=pregunta\s*\d*:|$)', re.IGNORECASE | re.DOTALL)
    matches = qa_pattern.findall(qa_text)

    for match in matches:
        question_text = match[1].replace('*', '').strip()  # Remove asterisks
        answer_text = match[2].replace('*', '').strip()    # Remove asterisks
        qa_pairs.append((question_text, answer_text))
    
    logging.info(f"Extracted {len(qa_pairs)} QA pairs.")
    return qa_pairs

# Function to process the dataset and save incrementally
async def process_dataset():
    global questions_created, processed_examples  # Track the total number of questions created and processed examples
    logging.info("Loading dataset.")
    ds = load_dataset("pauhidalgoo/patufet-textbooks")['train']
    highschool_content = ds.filter(lambda example: example['audience'] == 'High-School')
    total_examples = len(highschool_content)  # Total number of examples
    
    tasks = []
    
    for example in highschool_content:
        field = example['field']
        topic = example['topic']
        subtopic = example['subtopic']
        chapter = example['chapter']
        subunit = example['subunit']
        text = example['text']

        logging.info(f"Creating task for {field} - {topic} - {subtopic} - {chapter} - {subunit}.")
        tasks.append(process_and_save_example(field, topic, subtopic, chapter, subunit, text, total_examples))
    
    await asyncio.gather(*tasks)
    logging.info("All tasks completed.")

async def process_and_save_example(field, topic, subtopic, chapter, subunit, text, total_examples):
    global questions_created, processed_examples
    qa_text = await generate_qa_pairs(text)
    
    if qa_text and qa_text.strip() != 'NONE':
        qa_pairs = extract_qa_pairs(qa_text)
        questions_created += len(qa_pairs)  # Increment the question counter
        qa_results = []
        for question, answer in qa_pairs:
            qa_results.append({
                'field': field,
                'topic': topic,
                'subtopic': subtopic,
                'chapter': chapter,
                'subunit': subunit,
                'question': question,
                'answer': answer
            })
        if qa_results:
            df = pd.DataFrame(qa_results)
            df.to_csv('patufet-QA/patufet-QA_2.csv', mode='a', index=False, header=not os.path.exists('patufet-QA/patufet-QA_2.csv'))
    
    processed_examples += 1
    progress_percentage = (processed_examples / total_examples) * 100
    print(f"Processed and saved QA pairs for: {field} - {topic} - {subtopic} - {chapter} - {subunit}. {progress_percentage:.2f}% done. Total questions created: {questions_created}")
    logging.info(f"Processed and saved QA pairs for: {field} - {topic} - {subtopic} - {chapter} - {subunit}")

async def main():
    logging.info("Starting main function.")
    await process_dataset()
    logging.info(f"Finished processing dataset. Total questions created: {questions_created}")

if __name__ == "__main__":
    asyncio.run(main())