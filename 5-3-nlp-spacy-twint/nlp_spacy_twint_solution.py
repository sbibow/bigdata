# Natural Language Processing with spacy.io
import twint
import spacy
import csv
import os

nlp = spacy.load("en_core_web_sm")

# this is the earliest date we consider, CAUTION: processing takes ages if we use too many tweets
EARLIEST_DATE = "2020-10-01"

output_filename = "trump_tweets.csv"

# remove old output file if it exists
if os.path.exists(output_filename):
    os.remove(output_filename)

# scrape tweets
c = twint.Config()
c.Username = "realDonaldTrump"
c.Since = EARLIEST_DATE
c.Store_csv = True
c.Custom_csv = ["tweet"]
# save the scraped tweets into this file
c.Output = "trump_tweets.csv"
# run the actual twitter search
twint.run.Search(c)

full_text = ''
# open the saved tweets
with open('trump_tweets.csv', newline='') as csvfile:
    records = csv.DictReader(csvfile)
    for row in records:
        # filter retweets (starting with 'RT')
        if not row['tweet'].startswith('RT'):
            full_text += row['tweet']

doc = nlp(full_text)

# print all mentioned persons
for entity in doc.ents:
    if entity.label_ == 'PERSON':
        print(entity.text, entity.label_)

# count the number of Bidens in the tweets
num_bidens = len(
    [e for e in doc.ents if e.label_ == 'PERSON' and ('biden' in e.text.lower() or 'joe' in e.text.lower())])
print('Number of Bidens in Trump\'s tweets since ' + str(EARLIEST_DATE) + ': ' + str(num_bidens))
