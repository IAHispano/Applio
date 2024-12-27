import sys, json

def compare_language_strings(from_language_file, to_language_file, differences_file):
    with open(from_language_file, "r", encoding='utf-8') as r: # Opening file contains an updated strings, for example en_US.json
        from_language_data = json.load(r)
    with open(to_language_file, "r", encoding='utf-8') as r: # Language JSON file you translating
        to_language_data = json.load(r)
    key_difference = {} # Difference between new keys and keys you translated in your language
    for key, value in from_language_data.items():
        if key not in to_language_data:
            key_difference[key] = value
    with open(differences_file, "w", encoding='utf-8') as w: # File to save differences between keys in updated language file, and language file you translating
        json.dump(key_difference, w, ensure_ascii=False, indent=2)
    print('Language comparison complete.')

if len(sys.argv) != 4:
    print(sys.argv[0], ' from_language_file to_language_file differences_file')
    sys.exit()
compare_language_strings(sys.argv[1], sys.argv[2], sys.argv[3])
