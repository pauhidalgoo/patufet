from datasets import load_dataset

# Load The Stack dataset
dataset = load_dataset('bigcode/the-stack', data_dir='data/python', split='train')

# Filter for Python files and extract well-commented code
def is_well_commented(example):
    code = example['content']
    comments = sum(1 for line in code.splitlines() if line.strip().startswith("#"))
    return comments / max(1, len(code.splitlines())) > 0.1  # e.g., 10% of lines are comments

python_code = dataset.filter(is_well_commented)
python_code = python_code.shuffle(seed=42).select(range(0, 1_000_000))  # Adjust for token size
