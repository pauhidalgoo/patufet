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
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import google.generativeai as genai
import csv

load_dotenv()

topics = {
    "General Knowledge": ["Trivia", "facts", "definitions", "explanations"],
    "History": ["Events", "figures", "timelines", "historical analysis"],
    "Science": ["Biology", "chemistry", "physics", "astronomy", "earth science"],
    "Technology": ["Computers", "internet", "software", "AI", "gadgets"],
    "Geography": ["Countries", "cities", "landmarks", "maps", "physical features"],
    "Culture & Arts": ["Literature", "music", "film", "painting", "sculpture", "theatre"],
    "Current Events": ["News", "politics", "social issues", "global affairs"],
    "Health & Medicine": ["Diseases", "symptoms", "treatments", "nutrition", "fitness"],
    "Education": ["Learning", "teaching", "schools", "universities", "courses"],
    "Economics & Finance": ["Business", "markets", "investing", "personal finance", "personal budgeting"],
    "Philosophy & Ethics": ["Moral principles", "logic", "arguments", "philosophical theories"],
    "Psychology": ["Human behavior", "emotions", "mental health", "relationships", "helping people"],
    "Religion & Spirituality": ["Beliefs", "practices", "rituals", "mythology", "theology"],
    "Languages & Linguistics": ["Grammar", "vocabulary", "translation", "language learning"],
    "Law & Justice": ["Legal systems", "crimes", "punishments", "court proceedings"],
    "Basic Arithmetic": ["Addition", "subtraction", "multiplication", "division"],
    "Algebra": ["Equations", "inequalities", "functions", "graphing"],
    "Geometry": ["Shapes", "measurements", "proofs", "trigonometry"],
    "Calculus": ["Limits", "derivatives", "integrals", "applications"],
    "Statistics & Probability": ["Data analysis", "distributions", "hypothesis testing"],
    "Logic Puzzles & Riddles": ["Deductive reasoning", "problem-solving"],
    "Web navigation": ["places to shop online", "changing configuration", "educational web"],
    "Coding Challenges (Python)": ["Algorithms", "data structures", "programming languages", "solving errors"],
    "Creative Writing": ["Storytelling", "poetry", "fiction", "scripts", "screenplays", "speeches"],
    "Technical Writing": ["Manuals", "reports", "documentation", "instructions", "how-to", "recipes"],
    "Content Creation": ["Blog posts", "articles", "social media posts", "website content", "educational content"],
    "Copywriting": ["Advertising", "marketing", "persuasive writing", "sales copy"],
    "Email Writing": ["Professional communication", "formal emails", "informal emails"],
    "Resume & Cover Letter Writing": ["Job applications", "career advice"],
    "Academic Writing": ["Essays", "research papers", "dissertations", "citations"],
    "Translation": ["Converting text from one language to another"],
    "Summarization": ["Condensing information from a longer text."],
    "Paraphrasing": ["Rewriting text while maintaining the same meaning"],
    "Music Composition": ["Creating melodies", "harmonies", "rhythms"],
    "Art & Design": ["Generating images", "graphics", "logos", "illustrations"],
    "Travel Planning": ["Flights", "hotels", "transportation", "itineraries"],
    "Research & Information Gathering": ["Finding relevant sources", "data collection"],
    "Note-Taking & Organization": ["Outlining", "summarizing", "managing information"],
    "Project Management": ["Task lists", "timelines", "collaboration tools"],
    "Customer Service": ["Answering questions", "resolving issues", "providing support"],
    "Personal Assistant": ["Managing tasks", "errands", "appointments", "communications", "personal essays"],
    "Culture": ["song lyrics", "famous people", "pop culture", "famous books", "famous movies", "famous paintings"],
    "Language Learning": ["Vocabulary", "grammar", "pronunciation", "conversation practice"],
    "Code Debugging & Assistance": ["Identifying errors", "suggesting solutions"],
    "Document Editing & Proofreading": ["Grammar", "spelling", "punctuation", "style"],
    "Sentiment Analysis": ["Identifying emotions and opinions in text"],
    "Data Interpretation": ["Analyzing tables, statistics"],
    "Textual Analysis": ["Identifying themes", "characters", "plot", "literary devices", "named entities"],
    "Code Analysis": ["Understanding code structure", "functionality", "performance"],
    "Programming": ["Generating python code", "Commenting python code", "Solving coding challenges"],
    "Hate Speech Detection": ["Identifying offensive", "discriminatory language"],
    "Misinformation & Fake News Detection": ["Identifying false or misleading information"],
    "Cybersecurity Threats": ["Identifying phishing scams", "malware", "security vulnerabilities"],
    "Bias Detection": ["Identifying prejudice and discrimination in text or data"],
    "Harmful Content Filtering": ["Blocking inappropriate content", "violence", "pornography"],
    "Privacy & Data Security": ["Protecting personal information", "data breaches"],
    "Ethical Considerations": ["Identifying and mitigating ethical risks of AI"],
    "Jokes & Humor": ["Telling jokes", "understanding humor", "generating funny content"],
    "Games & Puzzles": ["Playing games", "solving puzzles", "creating interactive experiences"],
    "Trivia & Quizzes": ["Testing knowledge", "challenging users with questions"],
    "Storytelling & Roleplaying": ["Creating interactive narratives", "playing characters"],
    "Medical Diagnosis & Treatment": ["Analyzing symptoms", "suggesting diagnoses", "recommending treatments", "ethics"],
    "Legal Advice & Assistance": ["Providing legal information", "analyzing cases", "drafting documents"],
    "Financial Planning & Investment": ["Analyzing financial data", "providing investment advice"],
    "Educational Tutoring & Instruction": ["Providing personalized learning experiences", "answering questions"],
    "Scientific Research & Analysis": ["Analyzing data", "formulating hypotheses", "designing experiments"],
    "Reasoning & Inference": ["Drawing conclusions", "making predictions based on evidence", "correcting based on user response"],
    "Common Sense Reasoning": ["Understanding everyday knowledge and situations"],
    "Planning & Problem Solving": ["Problem solving", "Developing strategies", "solving complex problems", "simplifying problems"],
    "Learning & Adaptation": ["Improving performance over time based on experience"],
    "Explainability & Transparency": ["Explaining how decisions are made", "providing insights"],
    "Personalization & Customization": ["Adapting to individual user preferences and needs"],
    "Inventing New Products & Services": ["Generating ideas for innovative solutions"],
    "Designing Future Scenarios": ["Exploring possibilities and potential outcomes"],
    "Creating Fictional Worlds & Characters": ["Building immersive narratives and stories"],
    "Generating Art & Music in Different Styles": ["Exploring creative expression in various forms"],
    "Brainstorming & Idea Generation": ["Collaborating with humans to develop new ideas"],
    "Creative Writing & Storytelling": ["Co-creating stories and narratives with human authors"],
    "Scientific Discovery & Research": ["Assisting scientists in analyzing data and formulating hypotheses"],
    "Problem Solving & Decision Making": ["Working with humans to find solutions to complex problems", "asking more information"],
    "AI Bias & Fairness": ["Identifying and mitigating bias in AI systems"],
    "AI Safety & Security": ["Preventing malicious use of AI technology"],
    "Consciousness & Sentience in AI": ["Examining the philosophical implications of AI consciousness", "self consciousness", "own capabilties"],
    "Understanding User Intent": ["Identifying the underlying goal behind a user's request"],
    "Generating Different Types of Responses": ["Providing answers", "explanations", "summaries", "creative content", "lists", "code", "markdown", "chatting"],
    "Adapting to Different Communication Styles": ["Communicating in a way that is appropriate for the user"],
    "Learning and Improving from User Feedback": ["Continuously refining its abilities based on user interactions"],
    "Conversions": ["Converting units of measurement (e.g., length, weight, temperature)"],
    "Definitions": ["Providing definitions and explanations of words and phrases"],
    "Synonyms & Antonyms": ["Finding synonyms and antonyms for words"],
    "Spelling & Grammar Check": ["Checking for spelling and grammar errors in text"],
    "Simple Calculations": ["Performing basic arithmetic operations"],
    "Understanding limitations": ["Understanding things it can't do", "providing website resources as alternatives"],
    "Self-Awareness as a Model": ["Understanding its own nature as a large language model", "recognizing its capabilities and limitations"],
    "Responding to Questions About its Identity": ["Providing accurate and informative responses to questions about its creation", "purpose", "function"],
    "Differentiating Between Itself and Humans": ["Recognizing the differences between AI and human intelligence", "consciousness", "experience"],
    "Avoiding Anthropomorphism": ["Responding in a way that avoids intentions, or physical characteristics"],
    "Explaining its Decision-Making Process": ["Providing insights into how it generates responses, including the data and algorithms involved", "step by step", "examples"],
    "Acknowledging its Lack of Personal Beliefs and Opinions": ["Stating clearly that it does not have personal beliefs, opinions, or emotions like humans do", "avoiding polemic topics"],
    "Response formats": ["short answers", "paragraphs", "lists", "long answers", "summarizing previous response", "breaking down in steps"],
    "Adapting to user": ["skill level", "user not understanding", "different approaches"],
    "Starting conversations": ["Greetings and introduction", "Open-ended conversation starters", "Picking Up from previous interactions", "Responding to user initiated conversations"],
    "Mantaining a conversation": ["Asking clarifying questions", "Providing relevant follow-up information", "showing interest and engagement", "gracefully ending conversation"],
    "Understanding human emotion": ["Identifying emotions in text", "Responding empathetically", "generationg different emotional tones", "being sorry"],
    "Handling ambiguity": ["Ask for clarification", "Providing multiple possibilities"],
    "Personalization": ["Rembembering user preferences", "using conversational history"],
    "Multilingual": ["Translating languages", "Undersanding code in other programming languages"],
    "Unusual requests": ["Responding to an unusual request", "Edge cases", "Tricky questions", "Tricky problems", "Detecting when you are trying to trick it", "variations on common problems"],
    "Vague instructions": ["Understanding vague instructions", "similar things"],
    "Constrainds": ["Identifying constrains", "Format challenges", "Limited resources", "Code output", "JSON output"],
    "Creativity": ["Absurd scenarios", "Unexpected twists", "Humor detection", "Humor generation", "Sarcasm detection"],
    "Assistant learns from user": ["Meta-Learning from a small set of examples", "Counterfactual reasoning", "ethical dilemas", "collaborative creativity"]
}

c = 0
for k, v in topics.items():
    c += len(v)
print("Total subtopics: ", c)
instruction_styles = ["Direct Instructions", "Questions", "Scenarios", "Incomplete prompts (continuation task)", "Understanding text", "What if?...", "Analogies and metaphors", "Role-playing", "Constraints and challenges", "Step-by-step", "Clarification questions", "Back-and-forth dialogue", "Data interpretation", "Casual conversation", "Formal", "Informal", "Prediction", "Dilemmas", "Correcting the assistant", "User questions from input", "User questions from previous messages", "Start of conversation", "End of conversation"]
print("Total instruction styles: ", len(instruction_styles))

greetings_list = ["Include greetings", "Do not include greetings"]



logging.basicConfig(filename='./conversa/error_log.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding="utf-8")

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")




async def generate_conversa(topic, subtopic, instruction_style, greetings):

    prompt = f"""Create a multi-turn conversation, in catalan, between a User and an AI Assistant. The conversation should showcase {topic}, focusing on {subtopic}.
The intended instructions style for the user are {instruction_style}. Include 2-5 exchanges, (you can include an initial system prompt if needed using System:. {greetings}). The AI should write the text the user wants it to write. The output format should be:
User: [User utterance]
AI: [AI response]
User: [User utterance]
AI: [AI response]
...

Remember: follow the specified theme, and in catalan and without errors. Generate just the conversation.
"""
    
    try:
        response = await model.generate_content_async(prompt, safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
})
        return response.text
    except Exception as e:
        logging.error(f"Failed to generate content topic: {topic}, subtopic: {subtopic}, instruction style: {instruction_style}, greetings: {greetings}. Error: {e}")
        return None

total_errors = 0
async def generate_conversations(csv_writer, csv_file):
    global total_errors
    tasks = []

    concurrency_limit = 10
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def generate_limited_conversa(topic, subtopic, style, greet):
        async with semaphore:
            output = await generate_conversa(topic, subtopic, style, greet)
            return output, topic, subtopic, style, greet
        
    
    
    for topic, llista in topics.items():
        for subtopic in llista:
            for style in instruction_styles:
                for greet in greetings_list:
                    for i in range(2):
                        tasks.append(generate_limited_conversa(topic, subtopic, style, greet))
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        conversa, topic, subtopic, style, greet = await task
        if conversa != None:
            csv_writer.writerow({'Converse': conversa, 'Topic': topic, 'Subtopic': subtopic, 'Style': style, 'Greetings': greet})
            csv_file.flush()
        else:
            total_errors += 1

async def main():
 
    with open('./conversa/synthetic_conversations.csv', 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Converse', 'Topic', 'Subtopic', 'Style', 'Greetings']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        await generate_conversations(csv_writer, csv_file)

    print(total_errors)

if __name__ == "__main__":
    asyncio.run(main())
