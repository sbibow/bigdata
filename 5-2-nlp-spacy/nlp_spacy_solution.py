# Natural Language Processing with spacy.io
import spacy
import csv
from datetime import datetime

nlp = spacy.load("en_core_web_sm")

# this is the earliest date we consider, CAUTION: processing takes ages if we use too many tweets
EARLIEST_DATE = datetime.strptime("2020-01-01", "%Y-%m-%d")

full_text = ''
with open('trump_tweets_20201017.csv', newline='') as csvfile:
    records = csv.DictReader(csvfile)
    # filter out older tweets and retweets (starting with 'RT')
    for row in records:
        row_date = datetime.strptime(row['date'], "%Y-%m-%d %H:%M:%S")
        # check if row_date is after EARLIEST_DATE
        if row_date > EARLIEST_DATE:
            # check if text in row does not start with 'RT'
            if not row['text'].startswith('RT'):
                # add this row to full_text
                full_text += row['text']

# run the actual Natural Language Processing (see exercise 5.2.3, simply call "nlp(...)" on the text)
doc = nlp(full_text)

# print all mentioned persons
for entity in doc.ents:
    if entity.label_ == 'PERSON':
        # print this entry
        print(entity.text, entity.label_)

# count the number of Bidens in the tweets
num_bidens = len(
    [e for e in doc.ents if e.label_ == 'PERSON' and ('biden' in e.text.lower() or 'joe' in e.text.lower())])
print('Number of Bidens in Trump\'s tweets since ' + str(EARLIEST_DATE) + ': ' + str(num_bidens))
