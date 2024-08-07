from fuzzywuzzy import process

def load_dictionary(filename):
    with open(filename) as f:
        return set(word.strip().lower() for word in f)

dictionary = load_dictionary('dictionary.txt')

def fuzzy_match(word, dictionary, threshold=20):
    if not word:
        return []
    matches = process.extractBests(word, dictionary, score_cutoff=threshold)
    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
    return sorted_matches

def fuzzy_match_best(word, dictionary, threshold=90):
    if not word:
        return None
    best_match = process.extractOne(word, dictionary, score_cutoff=threshold)
    print(best_match)
    return best_match

def find_words(input_text):
    words = []
    length = len(input_text)
    start = 0

    while start < length:
        best_match = ("", 0)  
        end = start + 1

        while end <= length:
            segment = input_text[start:end]
            matches = fuzzy_match(segment, dictionary)
            if matches:
                for match in matches:
                    word, score = match
                    if score > best_match[1]:
                        best_match = (word, score)

            end += 1

        if best_match[0]:
            words.append(best_match[0])
            start += len(best_match[0])
        else:
            start += 1  

    return words

def form_sentence(words):
    return ' '.join(words)

def main():
    print("Enter 'exit' to quit.")
    while True:
        input_text = input("Enter letters: ").strip().lower()
        if input_text == 'exit':
            break

        words = find_words(input_text)
        sentence = form_sentence(words)
        print("Formed sentence:", sentence)

if __name__ == "__main__":
    main()
