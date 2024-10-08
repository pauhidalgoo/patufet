import tkinter as tk
from tkinter import messagebox
import pandas as pd

class CSVEditor:
    def __init__(self, master, csv_file):
        self.master = master
        self.master.title("CSV Editor")
        
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file)
        self.current_index = 0
        
        # Create UI elements
        self.label = tk.Label(master, text="Edit CSV Entry")
        self.label.pack()
        
        self.entries = []
        self.create_entry_fields()
        
        self.prev_button = tk.Button(master, text="Previous", command=self.prev_entry)
        self.prev_button.pack()
        
        self.save_button = tk.Button(master, text="Save", command=self.save_entry)
        self.save_button.pack()
        
        self.next_button = tk.Button(master, text="Next", command=self.next_entry)
        self.next_button.pack()

        self.save_all_button = tk.Button(master, text="Save All Changes", command=self.save_to_csv)
        self.save_all_button.pack()

        
        self.update_fields()

        master.bind("<Return>", lambda event: self.save_entry())  # Enter key to save
        master.bind("<Right>", lambda event: self.next_entry())  # Right arrow key to go next
        master.bind("<Left>", lambda event: self.prev_entry())  # Left arrow key to go previous
    
    def create_entry_fields(self):
        # Create entry fields based on the columns of the CSV
        for col in self.df.columns:
            frame = tk.Frame(self.master)
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            label = tk.Label(frame, text=col)
            label.pack()
            
            if col.lower() == "instructions":  # Assuming 'instructions' is the field needing more space
                entry = tk.Text(frame, height= 40, width=200)  # Bigger Text widget for instructions
            if col.lower() == "prompt" or col.lower() == "completion":
                entry = tk.Text(frame, height= 10, width=200)
            else:
                entry = tk.Entry(frame, width=200)  # Default Entry widget for other columns

            entry.pack( padx=5)
            self.entries.append(entry)
    
    def update_fields(self):
        # Update entry fields with current row data
        for i, entry in enumerate(self.entries):
            if isinstance(entry, tk.Text):  # If the widget is a Text widget
                entry.delete(1.0, tk.END)  # Clear the current entry
                entry.insert(tk.END, str(self.df.iloc[self.current_index, i]))  # Insert new value
            else:
                entry.delete(0, tk.END)  # Clear the current entry
                entry.insert(0, str(self.df.iloc[self.current_index, i])) # Insert new value

    def save_entry(self):
        # Save the current entry back to the DataFrame
        for i, entry in enumerate(self.entries):
            if isinstance(entry, tk.Text):  # If the widget is a Text widget
                self.df.iloc[self.current_index, i] = entry.get(1.0, tk.END).strip()  # Get multiline text
            else:
                self.df.iloc[self.current_index, i] = entry.get()  # Get single-line text
        messagebox.showinfo("Info", "Entry saved!")
        
    def next_entry(self):
        # Move to the next entry
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
            self.update_fields()
        else:
            messagebox.showinfo("Info", "You are at the last entry!")
    
    def prev_entry(self):
        # Move to the previous entry
        if self.current_index > 0:
            self.current_index -= 1
            self.update_fields()
        else:
            messagebox.showinfo("Info", "You are at the first entry!")

    def save_to_csv(self):
        # Save the modified DataFrame back to CSV
        self.df.to_csv("./premium-instructions/prompts_instructs_ultra_revised.csv", index=False)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    editor = CSVEditor(root, "./premium-instructions/prompts_instructs_ultra.csv")  # Replace with your CSV file
    root.mainloop()
