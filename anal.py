import praw
import os
import csv
import spacy
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import streamlit as st
from spacy import displacy

from dotenv import load_dotenv
load_dotenv()

#nltk.download('vader_lexicon')
#nltk.download('stopwords')
#nltk.download('punkt')

SPACY_MODEL_NAMES = ["en_core_web_sm", "en_core_web_md"]
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
ATTRS = ["line", "compound", "neg", "neu", "pos", "found_tickers"]

def kebab(s):
  return re.sub(
    r"(\s|_|-)+","-",
    re.sub(
      r"[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+",
      lambda mo: mo.group(0).lower(), s))

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

reddit = praw.Reddit(client_id=os.getenv('client_id'),
    client_secret=os.getenv('client_secret'),
    password=os.getenv('password'),
    username=os.getenv('username'),
    user_agent='wsb-anal v1')
reddit.read_only = True
wsb = reddit.subreddit('wallstreetbets')

easy_patterns = {}
patterns = []
# ticker, company_name, short_name, industry, description,
# website, logo, ceo, exchange, market cap, sector, tag_1, tag_2, tag_3
with open('companies.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader: #GILD AND $GILD
        tick = row["ticker"]
        big_tick = "$"+tick
        
        short_name = kebab(row["short_name"])

        easy_patterns[tick] = short_name
        easy_patterns[big_tick] = short_name

        patterns.append({
            "label": "STOCK", 
            "pattern":tick,
            "id": short_name})
        patterns.append({
            "label": "STOCK", 
            "pattern": big_tick,
            "id": short_name})

def force_sentence(text): #todo: could be cleaner
    return text + "."

def combined(submission):
    submission.comments.replace_more(limit=None) # make sure we get ALL comments
    header = [submission.title, cleanhtml(submission.selftext_html or "")]
    sub = header.copy()
    sub.extend(map(lambda x: cleanhtml(x.body_html), list(submission.comments) )) # clean of html 
    sub = map(lambda x: force_sentence(x), sub)
    return ("\n".join(header), " ".join(sub))

def u(url):
    return reddit.submission(url=url)

@st.cache(allow_output_mutation=True)
def url_to_text(url):
    return combined(u(url))

sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
stop_words.add("I")

def upsert(db, key):
    db[key] = db.get(key, 0) + 1
    return db

@st.cache(allow_output_mutation=True)
def process_text(text):

    tick_count = {}
    ret = []

    for line in nltk.tokenize.sent_tokenize(text):
        words = [word for word in nltk.tokenize.word_tokenize(line) if word not in stop_words]
        found_tickers = []

        for word in words:
            if easy_patterns.get(word, None):
                upsert(tick_count, word)
                found_tickers.append(word)

        ss = sid.polarity_scores(line)
        # compound: 0.8316, neg: 0.0, neu: 0.254, pos: 0.746,

        ret.append((line, ss['compound'], ss['neg'], ss['neu'], ss['pos'], found_tickers))

    return (tick_count, ret)


def st():
    st.sidebar.title("Interactive WSB sentiment visualizer")
    st.sidebar.markdown(
        """
    Process text with spacy models and visualize named entities,
    dependencies and more. Uses spaCy's built-in
    [displaCy](http://spacy.io/usage/visualizers) visualizer under the hood.
    """
    )

    url = st.text_area("Reddit URL to analyze", "https://www.reddit.com/r/wallstreetbets/comments/fmh129")
    (header, text) = url_to_text(url)
    (tickers, data) = process_text(text)
    sorted_t_count = {k: v for k, v in sorted(tickers.items(), key=lambda item: item[1], reverse=True)}

    st.header("TEEEXXXTTT")
    st.write(HTML_WRAPPER.format(header), unsafe_allow_html=True)
    df = pd.DataFrame(data, columns=ATTRS)
    st.dataframe(df)
    st.write(sorted_t_count)

    st.header("Winner")
    winning = list(sorted_t_count)[0]
    st.write(winning + " - " + str(sorted_t_count.get(winning))) #assumes we always get a ticker
    df_with = df[df['found_tickers'].str.contains(winning, regex=False)]

    st.header("Avg without filtering by ticker")
    st.dataframe(df.mean())

    st.header("Avg WITH filtering by ticker")
    st.dataframe(df_with.mean())




for submission in wsb.stream.submissions():
    print(submission.title + "("+ str(submission.score) +")" + " \n" + submission.permalink)
    (header, text) = combined(submission)
    (tickers, data) = process_text(text)
    sorted_t_count = {k: v for k, v in sorted(tickers.items(), key=lambda item: item[1], reverse=True)}
    df = pd.DataFrame(data, columns=ATTRS)
    try:
        winning = list(sorted_t_count)[0] # may not have found a ticker
        df_with = df[df['found_tickers'].str.contains(winning, regex=False)]
        print(winning + " - " + str(sorted_t_count.get(winning)))
        print(pd.concat((df.mean(), df_with.mean()), axis=1))
    except:
        pass
    print('\n')