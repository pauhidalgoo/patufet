import re
import csv

# Step 1: Read the text file
with open('patufet-summaries/empty_examples.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Step 2: Parse the prompts and summaries using regular expressions
# Updated pattern to correctly identify 'Prompt:' and remove 'Summary: ' from the extracted summaries
pattern = r'Prompt:(.*?)(Summary:)(.*?)(?=Prompt:|$)'
matches = re.findall(pattern, text, re.DOTALL)

# Step 3: Create a data structure to hold the prompts and summaries
data = []
for prompt, _, summary in matches:
    data.append({
        'prompt': prompt.strip(),
        'summary': summary.strip()
    })

# Step 4: Write the data to a CSV file
with open('patufet-summaries/empty_examples.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['prompt', 'summary']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in data:
        writer.writerow(row)

print("CSV file 'empty_examples.csv' has been created successfully.")