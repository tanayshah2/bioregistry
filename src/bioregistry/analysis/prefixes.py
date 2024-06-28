import json

with open("/Users/tanayshah/Documents/GitHub/bioregistry/src/bioregistry/data/bioregistry.json") as f:
    bioregistry = json.load(f)

prefixes = list(bioregistry.keys())

print(prefixes)