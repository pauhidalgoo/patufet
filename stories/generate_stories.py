import os
import asyncio
import csv
import google.generativeai as genai
import datasets
from tqdm.asyncio import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import random
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")

dataset =  datasets.load_dataset("oscar-corpus/OSCAR-2301", "ca", split="train", trust_remote_code=True)

sampled_dataset = dataset.shuffle(seed=42).select(range(200_000))

total_errors = 0

prompt_start = [
    "Write a short story (5-10 paragraphs), in catalan, and using only simple words suitable for a 3-5 year old child. The story should be based on the following text: ",
    "Create a fun and educational short story in catalan aimed at young children aged 3-5, using the following text as inspiration: ",
    "Develop a children's story in catalan (5-10 paragraphs) inspired by this text: ",
    "Write a short story (5-10 paragraphs) in catalan, that appeals to a 12-15 year old audience, using language and themes appropriate for this age group, somehow related to the text: ",
    "Create a short bedtime story (in catalan) for children aged 5-7. Take inspiration from: ",
    "Craft a story for chiltren aged 8-11 (in catalan), which should be 5-10 paragraphs, based on the following text: ",
    "Craft a short, mature and thought-provoking story in catalan suitable for an adult reader, incorporating complex themes, inspired on:"
]

story_style = [
    "The story should teach a simple moral lesson about honesty, kindness, or sharing.",
    "Include a character who learns something new about the world around them.",
    "It should aim to be an adventure story (example: the main character goes on a journey and learns something important.)",
    "Create a tale of exploration, where curiosity leads to an exciting discovery.",
    "Tell a mysterious story that keeps the reader guessing until the end.",
    "Write a short mystery where the protagonist solves a puzzle or uncovers a secret.",
    "Create a funny and lighthearted story that will make the reader smile.",
    "Write a humorous tale with a playful twist at the end.",
    "Write a heartfelt story that explores deep emotions and relationships.",
    "Create a whimsical fantasy story, which can feature mythical creatures, a quest to find something important, a dialogue with a wise, magical being...",
    "The story should try to be about science fiction, aiming to inspire curiosity."
]


prompt_constraints = [
    "Primarily be told through dialogue between the characters.",
    "Include at least one dialogue",
    "Focus on the internal thoughts of the characters.",
    "Develop a character who faces a significant decision, and explore the consequences.",
    "Include an unexpected twist at the end of the story that changes the reader's understanding.",
    "Have a surprising conclusion that leaves the reader thinking.",
    "Aim to get insights into the characters, the consequences of their actions, decision making and interactions.",
    "Focus on an everyday situation",
    "Include an animal as the main character",
    "Include a dialogue where characters ask questions about their surroundings.",
    "Involve magical elements like talking toys or enchanted forests.",
    "End with a life lesson"
]



async def generate_story(web_sample):
    web_sample = web_sample[:1500]

    start = random.choice(prompt_start)
    style = random.choice(story_style)
    constraint1, constraint2 = random.sample(prompt_constraints, 2)

    prompt = f"""{start}: 
    '{web_sample}'. 
    {style} The story should:
    - {constraint1}
    - {constraint2}
    Start the story without using 'Hi havia una vegada' or other typical starts, be creative. Also, don't include 'Fi.' or other typical phrases in stories, try to be diverse and creative. Remember: always in catalan, and use simple and correct words!"""
    
    try:
        response = await model.generate_content_async(prompt, safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
})
        return prompt, response.text
    except Exception as e:
        print(f"Error: {e}")
        return None, None

async def generate_stories(num_stories, csv_writer, csv_file):
    global total_errors
    tasks = []

    concurrency_limit = 40
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def generate_limited_story(web_sample):
        async with semaphore:
            prompt, story = await generate_story(web_sample)
            return prompt, story
    
    for i in range(num_stories):
        web_sample = sampled_dataset[i]['text']
        tasks.append(generate_limited_story(web_sample))
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        prompt, story = await task
        if prompt != None:
            csv_writer.writerow({'Prompt': prompt, 'Story': story})
            csv_file.flush()
        else:
            total_errors += 1

async def main():
    num_stories = 200_000
    
    with open('./stories/synthetic_stories_dataset.csv', 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Prompt', 'Story']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        await generate_stories(num_stories, csv_writer, csv_file)

    print(total_errors)

if __name__ == "__main__":
    asyncio.run(main())
