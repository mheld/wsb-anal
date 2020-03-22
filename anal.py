import praw
import os
import spacy
from dotenv import load_dotenv
load_dotenv()

reddit = praw.Reddit(client_id=os.getenv('client_id'),
    client_secret=os.getenv('client_secret'),
    password=os.getenv('password'),
    username=os.getenv('username'),
    user_agent='wsb-anal v1')
# don't want to actually be able to post anything
reddit.read_only = True

wsb = reddit.subreddit('wallstreetbets')

# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

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