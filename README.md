# cat-dataset

The idea is to have 3B tokens of quality data.

5 parts:
- Common Crawl subset (majority of the data, +2.5B)
  - The idea is to filter it a bit, to have educational content,
    like fineweb-edu. Since the crawl contains 12.5B tokens in catalan, 4/5 of the data will be left out
    (in reality, I think we ended up with aprox 4B tokens)

- Python code (60.000 examples)
  - I would like the model to be able to generate some Python code. To do so, I have to include examples in the training. The problem in this part is that usually the code comments are in english, but we want them in catalan (since that is what the model understands)
    - Some of the code is going to be from synthetically generated data

- Stories (100M):
  - Stories are a great way for the model to grasp some real world understanding, and also a way to understand the structure of the language. This subset would be 
  synthetic generated data, from a bigger model. The idea is to imitate cosmopedia.

- Textbook (0.29B):
  - As shown with the phi-1.5 model, textbook quality data enhance the learning process of LLMs.
  
- Questions & answers (500.000 entries):
  - Made by @rogerbaiges