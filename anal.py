import praw
import os
import csv
import spacy
from re import sub
from dotenv import load_dotenv
load_dotenv()

def kebab(s):
  return sub(
    r"(\s|_|-)+","-",
    sub(
      r"[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+",
      lambda mo: mo.group(0).lower(), s))

reddit = praw.Reddit(client_id=os.getenv('client_id'),
    client_secret=os.getenv('client_secret'),
    password=os.getenv('password'),
    username=os.getenv('username'),
    user_agent='wsb-anal v1')
reddit.read_only = True
wsb = reddit.subreddit('wallstreetbets')

# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
ruler = spacy.pipeline.EntityRuler(nlp)

patterns = []
# ticker, company_name, short_name, industry, description,
# website, logo, ceo, exchange, market cap, sector, tag_1, tag_2, tag_3
with open('companies.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        patterns.append({
            "label": "STOCK", 
            "pattern":row["ticker"], 
            "id":kebab(row["short_name"])})

ruler.add_patterns(patterns)
nlp.add_pipe(ruler)


def combined(submission):
    return submission.title + " " + submission.selftext

def u(url):
    return reddit.submission(url=url)

for submission in wsb.stream.submissions():
    submission.author
    submission.id
    submission.title
    submission.created_utc
    submission.permalink
    submission.score
    submission.selftext
    list(submission.comments)
    flatten_comments(submission.comments)