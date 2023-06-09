import torch
import os
import datetime
import argparse
import helper
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='Load a PyTorch model')
parser.add_argument('-m_path', type=str, required=True, help='path to saved model')
args = parser.parse_args()

model = helper.BigramLanguageModel()
model.load_state_dict(torch.load(args.m_path))


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
text_file_path = os.path.join(parent_dir, 'Texts', 'tinyShakespeare.txt')
with open(text_file_path, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
# create a mapping from characters to integers
itos = { i:ch for i,ch in enumerate(chars) }
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(context, max_new_tokens=4000)[0].tolist()
decoded_text = decode(generated_text)

# out to console
sentences = decoded_text.split(". ")
for sentence in sentences:
    sentence = sentence.strip() + "."
    for letter in sentence:
        print(letter, end="", flush=True)
        time.sleep(0.05)
    print()