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
import json

load_dotenv()

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")

def load_topics(filename='./textbooks/topics.json'):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config

audiences = {
    "General": "provide a comprehensive explanation suitable for a general audience. Include examples and exercises where relevant.",
    "Kid": "explain the topic in simple language and include engaging stories or analogies. Make the content fun and easy to understand.",
    "High-School": "offer a detailed explanation with examples and exercises appropriate for high-school students. Ensure clarity and rigor in the explanations.",
    "College": "provide an in-depth and rigorous explanation with detailed examples and exercises suitable for college-level students. Include advanced concepts where relevant.",
    "Researcher": "engage a highly knowledgeable audience with deep expertise. Include analysis of recent research, advanced theories, and detailed examples."
}

def generate_content(field, topic, subtopic, chapters, current_chapter, current_subunits, current, audience):
    audience_description = audiences[audience]

    prompt = f"""Write a detailed, very long and comprehensive textbook chapter on the topic of '{topic}-{subtopic}' under '{field}'. The previous chapter(s) that have already been covered are: {chapters}. 
    The current chapter is called {current_chapter}, and we have written the following part(s) of it: {current_subunits}. You are going to be writing the sub-unit titled {current}. 
    Create it while trying to provide an in-depth explanation, be rigorous, engaging and avoiding incorrect information. You can use the knowledge you have in english, but the text must be in catalan. The content should be targeted to a {audience} audience, so {audience_description}.
    Include any examples, exercises, proofs, detailed analyses, equations, dates, key events, names... rellevant to the chapter.
    Do not include a headline, title, introduction nor indications, simply write it. Avoid too much formatting (don't use **, # ...) and make it more narrative and like a real life book. The language of the textbook must be in catalan: do not include any word in spanish and make sure what you write is correct. Do not include "[Continuarà]" or similar things.
"""
    
    
    try:
        response = model.generate_content(prompt)
        response = response.text
    except Exception as e:
        response = None

    return response

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

def generate_chapters(Field, Topic, Subtopic, Audience):
    units_prompt = f"""Create the units and subunits for an imaginary textbook for the topic "{Field} - {Topic}: {Subtopic}" intended for a {Audience} audience. Focus on this topic. The textbook is in catalan, but you can use your knowledge in english. The output should have the following format:
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
... You must only write this index, do not in any case provide any type of explanation. At most, the output should contain 25 sub-units, don't make the index too long. Rembember the audience you are aiming for, and to just output the units in Exactly the same format provided (same way of indexing)."""
    try:
        response = model.generate_content(units_prompt)
        response = response.text
        units = extract_units(response)
    except Exception as e:
        units = None
    print(response)
    print(units)
    return units


def save_results(data):
    df = pd.DataFrame(data)
    df.to_csv("./textbooks/synthetic_textbooks.csv", mode='a', index=False)

def main():
    topics = load_topics()
    for audience in audiences.keys():
        for field, topics in topics.items():
            for topic, subtopics in topics.items():
                for subtopic in subtopics:
                    chapters = generate_chapters(field, topic, subtopic, audience)
                    done_chapters = []
                    results = []
                    for chapter, subunits in chapters.items():
                        done_subunits = []
                        for subunit in subunits:

                            content = generate_content(field, topic, subtopic, done_chapters, chapter, done_subunits, subunit, audience)

                            # Save the generated content. You must save, in different columns: text (the content generated), field, topic, subtopic, chapter, subunit, audience

                            result = {
                                'text': content,
                                'field': field,
                                'topic': topic,
                                'subtopic': subtopic,
                                'chapter': chapter,
                                'subunit': subunit,
                                'audience': audience
                            }
                            
                            # Add the result to the list
                            results.append(result)
                            
                            done_subunits.append(subunit)
                        done_chapters.append(chapter)
                    save_results(results)
                    return

if __name__ == "__main__":
    main()





