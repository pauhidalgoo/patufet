import json
import asyncio
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
load_dotenv()


genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")

logging.basicConfig(filename='./hellaswag/error_log.txt', level=logging.ERROR, encoding="utf-8",
                    format='%(asctime)s - %(levelname)s - %(message)s')


async def translate_entry(entry, semaphore):
    async with semaphore:
        prompt = f"""
        Translate the following text from English to Catalan:
        1. activity_label: {entry['activity_label']}
        2. ctx_a: {entry['ctx_a']}
        3. ctx_b: {entry['ctx_b']}
        4. ctx: {entry['ctx']}
        5. endings: {entry['endings']}

        The output format should be:
        1:- TRANSLATION OF ACTIVITY LABEL
        2:- TRANSLATION OF CTX_A
        3:- TRANSLATION OF CTX_B
        4:- TRANSLATION OF CTX
        5:- TRANSLATION OF ENDING 1
        6:- TRANSLATION OF ENDING 2
        7:- TRANSLATION OF ENDING 3
        8:- TRANSLATION OF ENDING 4

        As you can see, each translation is separated by a new line and indicated using 1:- . Also, each of the 4 endings is on a separate line.
        Do NOT include activity_label, ctx_a, ctx_b... on your response, just 
        1:- Translation
        2:- Translation
        ...
        Remember: use ":-". Do not hallucinate. Return the whole translation, do not stop mid generation.
        """

        try:
            response = await model.generate_content_async(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                }
            )

            translated_text = response.text.split('\n')

            entry['activity_label'] = translated_text[0].split(':- ')[1]
            entry['ctx_a'] = translated_text[1].split(':- ')[1]
            entry['ctx_b'] = translated_text[2].split(':- ')[1]
            entry['ctx'] = translated_text[3].split(':- ')[1]
            ending1 = translated_text[4].split(':- ')[1]
            ending2 = translated_text[5].split(':- ')[1]
            ending3 = translated_text[6].split(':- ')[1]
            ending4 = translated_text[7].split(':- ')[1]
            entry['endings'] = [ending1, ending2, ending3, ending4]

            return entry
        except Exception as e:
            logging.error(f"{entry}.ERROR: {e}")
            return None

async def process_line(line, semaphore, output_file):
    entry = json.loads(line.strip())
    translated_entry = await translate_entry(entry, semaphore)
    if translated_entry is not None:
        with open(output_file, 'a', encoding="utf-8") as outfile:
            outfile.write(json.dumps(translated_entry) + '\n')


async def translate_dataset(input_file, output_file):
    semaphore = asyncio.Semaphore(10)
    tasks = []

    with open(input_file, 'r', encoding="utf-8") as infile:
        for line in infile:
            tasks.append(process_line(line, semaphore, output_file))

    await asyncio.gather(*tasks)

async def main():
    input_file = './hellaswag/hellaswag_val.jsonl'
    output_file = './hellaswag/hellaswag_cat.jsonl'
    await translate_dataset(input_file, output_file)

if __name__ == "__main__":
    asyncio.run(main())
