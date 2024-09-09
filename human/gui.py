import tkinter as tk
from tkinter import simpledialog, messagebox
import json
import random

class ConversationCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Conversation Creator")
        
        self.conversations = []

        # System prompt
        self.system_label = tk.Label(root, text="System Prompt (Optional):")
        self.system_label.grid(row=0, column=0, sticky="w")
        self.system_entry = tk.Text(root, height=2, width=50)
        self.system_entry.grid(row=0, column=1, padx=10, pady=10)

        # User question
        self.user_label = tk.Label(root, text="User Question:")
        self.user_label.grid(row=1, column=0, sticky="w")
        self.user_entry = tk.Text(root, height=2, width=50)
        self.user_entry.grid(row=1, column=1, padx=10, pady=10)

        # Expected answer
        self.assistant_label = tk.Label(root, text="Assistant Answer:")
        self.assistant_label.grid(row=2, column=0, sticky="w")
        self.assistant_entry = tk.Text(root, height=2, width=50)
        self.assistant_entry.grid(row=2, column=1, padx=10, pady=10)

        # Buttons
        self.add_turn_button = tk.Button(root, text="Add Turn", command=self.add_turn)
        self.add_turn_button.grid(row=3, column=0, padx=10, pady=10)

        self.add_conversation_button = tk.Button(root, text="Add Conversation", command=self.add_conversation)
        self.add_conversation_button.grid(row=3, column=1, padx=10, pady=10)

        self.theme_button = tk.Button(root, text="Generate Random Theme", command=self.generate_random_theme)
        self.theme_button.grid(row=4, column=0, padx=10, pady=10)

        self.save_button = tk.Button(root, text="Save Dataset", command=self.save_dataset)
        self.save_button.grid(row=4, column=1, padx=10, pady=10)

        # Conversation List
        self.conversation_list_label = tk.Label(root, text="Current Conversation:")
        self.conversation_list_label.grid(row=5, column=0, sticky="w")
        self.conversation_listbox = tk.Listbox(root, width=70, height=10)
        self.conversation_listbox.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        self.current_conversation = []

    def add_turn(self):
        user_text = self.user_entry.get("1.0", tk.END).strip()
        assistant_text = self.assistant_entry.get("1.0", tk.END).strip()

        if user_text and assistant_text:
            self.current_conversation.append({"role": "user", "content": user_text})
            self.current_conversation.append({"role": "assistant", "content": assistant_text})

            # Update listbox
            self.conversation_listbox.insert(tk.END, f"User: {user_text}")
            self.conversation_listbox.insert(tk.END, f"Assistant: {assistant_text}")
            
            # Clear entries
            self.user_entry.delete("1.0", tk.END)
            self.assistant_entry.delete("1.0", tk.END)
        else:
            messagebox.showwarning("Warning", "User question and assistant answer cannot be empty.")

    def add_conversation(self):
        system_prompt = self.system_entry.get("1.0", tk.END).strip()
        conversation = []

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        
        conversation.extend(self.current_conversation)

        if conversation:
            self.conversations.append({"messages": conversation})
            self.conversation_listbox.delete(0, tk.END)
            self.current_conversation = []
            messagebox.showinfo("Success", "Conversation added to dataset.")
        else:
            messagebox.showwarning("Warning", "Conversation cannot be empty.")

    def generate_random_theme(self):
        themes = [
            "Planning a vacation", "Discussing a recipe", "Job interview preparation", 
            "Talking about a book", "Learning a new language", "Solving a technical issue",
            "Debating a historical event", "Giving relationship advice", "Fitness and health tips"
        ]
        theme = random.choice(themes)
        messagebox.showinfo("Random Theme", f"Use this theme for inspiration: {theme}")

    def save_dataset(self):
        if not self.conversations:
            messagebox.showwarning("Warning", "No conversations to save.")
            return
        
        file_path = simpledialog.askstring("Save Dataset", "Enter the file name (with .json extension):")
        
        if file_path:
            with open(file_path, 'w') as file:
                json.dump(self.conversations, file, indent=4)
            messagebox.showinfo("Success", f"Dataset saved as {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ConversationCreator(root)
    root.mainloop()
