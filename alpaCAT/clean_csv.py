import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('alpaCAT/alpaCAT_old.csv')

# Define the regex pattern to identify the unwanted text
pattern = r'(please provide|translate)'

# Replace the entire 'input' content with an empty string if it matches the pattern
df['input'] = df['input'].apply(lambda x: '' if isinstance(x, str) and re.search(pattern, x, flags=re.IGNORECASE) else x)
df['output'] = df['output'].apply(lambda x: '' if isinstance(x, str) and re.search(pattern, x, flags=re.IGNORECASE) else x)
df['instruction'] = df['instruction'].apply(lambda x: '' if isinstance(x, str) and re.search(pattern, x, flags=re.IGNORECASE) else x)

# Save the cleaned DataFrame back to a CSV file
df.to_csv('alpaCAT/alpaCAT.csv', index=False)

print("Cleaning completed and saved to 'alpaCAT/alpaCAT.csv'")