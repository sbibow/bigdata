# NLP Pipeline using Luigi
import twint
import spacy
import csv
from datetime import datetime, date
import os
from luigi.parameter import DateParameter
from luigi import Task, LocalTarget
import luigi

nlp = spacy.load("en_core_web_sm")

# this is the earliest date we consider, CAUTION: processing takes ages if we use too many tweets
EARLIEST_DATE = "2020-10-01"


class Scrape(Task):
    earliest_date = DateParameter(default=date.today())

    def run(self):
        output_filename = 'tweets_since_%s.csv' % self.earliest_date

        # remove old output file if it exists
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # scrape tweets
        c = twint.Config()
        c.Username = "realDonaldTrump"
        c.Since = str(self.earliest_date)
        c.Store_csv = True
        c.Custom_csv = ["tweet"]
        # save the scraped tweets into this file
        c.Output = output_filename
        # run the actual twitter search
        twint.run.Search(c)

    def output(self):
        return LocalTarget('tweets_since_%s.csv' % self.earliest_date)


class Filter(Task):
    earliest_date = DateParameter(default=date.today())

    def requires(self):
        return Scrape(self.earliest_date)

    def run(self):
        # open the saved tweets
        with self.output().open('w') as output_file:
            with self.input().open('r') as input_file:
                records = csv.DictReader(input_file)
                for row in records:
                    # filter out retweets (starting with 'RT')
                    if not row['tweet'].startswith('RT'):
                        output_file.write(row['tweet'])

    def output(self):
        return LocalTarget('tweets_filtered_since_%s.csv' % self.earliest_date)


class GetBidens(Task):
    earliest_date = DateParameter(default=date.today())

    def requires(self):
        return Filter(self.earliest_date)

    def run(self):
        with self.input().open('r') as file:
            doc = nlp(file.read())

            # print all mentioned persons
            for entity in doc.ents:
                if entity.label_ == 'PERSON':
                    print(entity.text, entity.label_)

            # count the number of Bidens in the tweets
            num_bidens = len([e for e in doc.ents if
                              e.label_ == 'PERSON' and ('biden' in e.text.lower() or 'joe' in e.text.lower())])
            print('Number of Bidens in Trump\'s tweets since ' + str(EARLIEST_DATE) + ': ' + str(num_bidens))


if __name__ == '__main__':
    luigi.build([GetBidens(earliest_date=datetime.strptime(EARLIEST_DATE, "%Y-%m-%d"))], local_scheduler=True)
