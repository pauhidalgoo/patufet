import tkinter as tk
from tkinter import messagebox
import pandas as pd
import random

# Predefined lists for random selection
TYPES = ["Multiple Choice", "True/False", "Fill in the Blanks", "Short Answer"]
TOPICS = ["Math", "Science", "History", "Geography", "Language"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

class CSVEditor:
    def __init__(self, master, csv_file):
        self.master = master
        self.master.title("CSV Entry Creator")
        
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file)
        
        # Styling and layout adjustments
        self.frame = tk.Frame(master, padx=10, pady=10)
        self.frame.pack(expand=True, fill=tk.BOTH)
        
        self.label = tk.Label(self.frame, text="Add a New Entry", font=("Arial", 16))
        self.label.grid(row=0, columnspan=2, pady=(0, 20))

        # Input field for Prompt
        self.prompt_label = tk.Label(self.frame, text="Prompt", font=("Arial", 12))
        self.prompt_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        self.prompt_entry = tk.Text(self.frame, height=5, width=50, wrap=tk.WORD)
        self.prompt_entry.grid(row=1, column=1, pady=(0, 10))
        
        # Button to generate random values
        self.random_button = tk.Button(self.frame, text="Generate Random Values", font=("Arial", 12), command=self.generate_random_values)
        self.random_button.grid(row=2, columnspan=2, pady=(10, 20))

        # Display generated fields
        self.type_label = tk.Label(self.frame, text="Type", font=("Arial", 12))
        self.type_label.grid(row=3, column=0, sticky=tk.W)
        self.type_value = tk.Label(self.frame, text="Not set", font=("Arial", 12))
        self.type_value.grid(row=3, column=1, sticky=tk.W)
        
        self.topic_label = tk.Label(self.frame, text="Topic", font=("Arial", 12))
        self.topic_label.grid(row=4, column=0, sticky=tk.W)
        self.topic_value = tk.Label(self.frame, text="Not set", font=("Arial", 12))
        self.topic_value.grid(row=4, column=1, sticky=tk.W)

        self.difficulty_label = tk.Label(self.frame, text="Difficulty", font=("Arial", 12))
        self.difficulty_label.grid(row=5, column=0, sticky=tk.W)
        self.difficulty_value = tk.Label(self.frame, text="Not set", font=("Arial", 12))
        self.difficulty_value.grid(row=5, column=1, sticky=tk.W)

        # Save button
        self.save_button = tk.Button(self.frame, text="Save Entry", font=("Arial", 12), command=self.save_entry, bg="#4CAF50", fg="white")
        self.save_button.grid(row=6, columnspan=2, pady=(20, 0))
    
    def generate_random_values(self):
        """Generate random values for 'Type', 'Topic', and 'Difficulty'."""
        type_val = random.choice(TYPES)
        topic_val = random.choice(TOPICS)
        difficulty_val = random.choice(DIFFICULTIES)
        
        self.type_value.config(text=type_val)
        self.topic_value.config(text=topic_val)
        self.difficulty_value.config(text=difficulty_val)
    
    def save_entry(self):
        """Save a new entry with the current values."""
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        type_val = self.type_value.cget("text")
        topic_val = self.topic_value.cget("text")
        difficulty_val = self.difficulty_value.cget("text")

        if not prompt:
            messagebox.showwarning("Warning", "Prompt cannot be empty!")
            return
        
        if type_val == "Not set" or topic_val == "Not set" or difficulty_val == "Not set":
            messagebox.showwarning("Warning", "Generate random values for Type, Topic, and Difficulty before saving!")
            return
        
        # Append new row to the DataFrame
        new_row = {"Prompt": prompt, "Type": type_val, "Topic": topic_val, "Difficulty": difficulty_val}
        self.df = self.df.append(new_row, ignore_index=True)
        
        messagebox.showinfo("Info", "New entry saved!")
        self.clear_fields()
    
    def clear_fields(self):
        """Clear the fields after saving an entry."""
        self.prompt_entry.delete("1.0", tk.END)
        self.type_value.config(text="Not set")
        self.topic_value.config(text="Not set")
        self.difficulty_value.config(text="Not set")

    def save_to_csv(self):
        """Save the DataFrame back to the CSV file."""
        self.df.to_csv(self.csv_file, index=False)
        messagebox.showinfo("Info", "All changes saved to CSV!")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    editor = CSVEditor(root, "./rlhf/manual_prompts.csv")  # Replace with your desired CSV file path
    root.mainloop()
