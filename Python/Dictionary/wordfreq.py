from pprint import pprint
preamble = "We the People of the United States, in Order to form a more perfect Union, establish Justice, insure domestic Tranquility, provide for the common defence, promote the general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity, do ordain and establish this Constitution for the United States of America."


char_frequency = {}
for char in preamble:
    if char in char_frequency:
        char_frequency[char] += 1
    else:
        char_frequency[char] = 1

words = preamble.split()
# Count words manually
word_frequency = {}
for word in words:
    if word in word_frequency:
        word_frequency[word] += 1
    else:
        word_frequency[word] = 1

# print(sorted(char_frequency)) (ouch -- cannot sort)
# print(sorted(char_frequency.items())) (close but not quite)

# Store in frequency of characters however dictionaries are unordered collections
# pull out key value pairs from dictionary, convert into a tuple,
# then put them into a list for sorting
char_frequency_sorted = sorted(
    char_frequency.items(), key=lambda kv: kv[1], reverse=True)
word_frequency_sorted = sorted(
    word_frequency.items(), key=lambda kv: kv[1], reverse=True)

pprint(char_frequency_sorted[0:5], width=1)

# Show the top 10
print("\nTop 10 words:")
for word, freq in word_frequency_sorted[:10]:
    print(f"{word}: {freq}")
