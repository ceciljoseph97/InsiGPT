import os
import requests

folder_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),"Texts")
urls = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
]
modelText = [
    os.path.join(folder_path, "tinyShakespeare.txt"),
    os.path.join(folder_path, "janeAusPP.txt")
]

for i in range(len(urls)):
    if os.path.isfile(modelText[i]):
        print(f"{modelText[i]} already exists, skipping download.")
    else:
        response = requests.get(urls[i])
        if response.status_code == 200:
            with open(modelText[i], "w", encoding="utf-8") as f:
                f.write(response.content.decode())
                print(f"Downloaded {urls[i]} and saved to {modelText[i]}.")
        else:
            print(f"Error downloading {urls[i]}: {response.status_code}.")