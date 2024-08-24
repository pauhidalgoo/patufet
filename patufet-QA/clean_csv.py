import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('patufet-QA/patufet-QA_old.csv')

# 1. Remove rows with "Pregunta x" or "Resposta x" in 'question' or 'answer'
pattern = re.compile(r'Pregunta \d+|Resposta \d+', re.IGNORECASE)
df = df[~df['question'].str.contains(pattern, na=False) & ~df['answer'].str.contains(pattern, na=False)]

# 2. Remove rows where 'question' or 'answer' is empty
df = df.dropna(subset=['question', 'answer'])
df = df[(df['question'].str.strip() != '') & (df['answer'].str.strip() != '')]

# 3. Remove rows where 'answer' contains 'NONE'
df = df[~df['answer'].str.contains('NONE', case=False, na=False)]

# 4. Format bullet points only in 'question' column
def format_bullets(text):
    # Split text into lines
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        if line.strip():  # Check if the line is not empty
            # If the line starts with a single space, format it as a bullet point
            if line.startswith(' '):  # Detects single space at the start of a line
                formatted_lines.append(f"- {line.strip()}")
            else:
                formatted_lines.append(line.strip())

    return '\n'.join(formatted_lines)

df['question'] = df['question'].apply(format_bullets)

# 5. Reorder columns to have 'question' first and 'answer' second
df = df[['question', 'answer'] + [col for col in df.columns if col not in ['question', 'answer']]]

# 6. Shuffle the rows randomly
df = df.sample(frac=1).reset_index(drop=True)

# Save the cleaned, formatted, and shuffled CSV back to file
df.to_csv('patufet-QA/patufet-QA.csv', index=False)

print("CSV file cleaned, shuffled, and saved as 'patufet-QA.csv'")