from utils import load_doc, load_descriptions, clean_captions
from pickle import dump

doc = load_doc("data/Flickr8k.token.txt")
descriptions = load_descriptions(doc)
clean_captions(descriptions)
dump(descriptions, open("data/descriptions.pkl", "wb"))

print("✅ Descriptions cleaned and saved.")
