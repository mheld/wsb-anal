import praw
from dotenv import load_dotenv
load_dotenv()

reddit = praw.Reddit(
    client_id=''
    client_secret='')