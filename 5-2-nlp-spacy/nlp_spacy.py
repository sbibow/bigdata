# Natural Language Processing with spacy.io
import spacy
import csv
from datetime import datetime
from pprint import pprint
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

# this is the earliest date we consider, CAUTION: processing takes ages if we use too many tweets
EARLIEST_DATE = datetime.strptime("2020-01-01", "%Y-%m-%d")

full_text = ''
with open('trump_tweets_20201017.csv', newline='') as csvfile:
    records = csv.DictReader(csvfile)
    # filter out older tweets and retweets (starting with 'RT')
    for row in records:
        row_date = datetime.strptime(row['date'], "%Y-%m-%d %H:%M:%S")
        # TODO check if row_date is after EARLIEST_DATE
        if row_date > EARLIEST_DATE:
            # TODO check if text in row does not start with 'RT'
            if row['isRetweet'] == "f":
                # TODO add this row to full_text
                full_text += row['text']

# TODO run the actual Natural Language Processing (see exercise 5.2.3, simply call "nlp(...)" on the text)
doc = nlp(full_text)

# print all mentioned persons
# for entity in doc.ents:
#     if entity.label_ == 'PERSON':
#         # TODO print this entry
#         print(entity)

# TODO count the number of 'joe' and 'biden' in the tweets
def filter_bidens(text):
    return "joe" in str(text).lower() or "biden" in str(text).lower()

bidens = list(filter(filter_bidens, doc.ents))
print('Number of Bidens in Trump\'s tweets since ' + str(EARLIEST_DATE) + ': ' + str(len(bidens)))

labels = {}
texts = {}
for ent in bidens:
    labels[ent.label_] = labels.get(ent.label_,0)+1
    texts[str(ent)] = texts.get(str(ent),0)+1
pprint(labels)
most_used_bidens = sorted(texts, key=lambda e: texts[e], reverse=True)[:10]
most_used_bidens_hist = {text: texts[text] for text in most_used_bidens}

plt.bar(most_used_bidens_hist.keys(), most_used_bidens_hist.values())
plt.show()

