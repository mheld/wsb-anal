import praw
import os
import csv
import spacy
import pandas as pd
import re
import streamlit as st
from spacy import displacy

from dotenv import load_dotenv
load_dotenv()

SPACY_MODEL_NAMES = ["en_core_web_sm", "en_core_web_md"]
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

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

patterns = []
# ticker, company_name, short_name, industry, description,
# website, logo, ceo, exchange, market cap, sector, tag_1, tag_2, tag_3
with open('companies.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader: #GILD AND $GILD
        patterns.append({
            "label": "STOCK", 
            "pattern":row["ticker"], 
            "id":kebab(row["short_name"])})
        patterns.append({
            "label": "STOCK", 
            "pattern": "$"+row["ticker"], 
            "id":kebab(row["short_name"])})

def combined(submission):
    return submission.title + " " + cleanhtml(submission.selftext_html)

def u(url):
    return reddit.submission(url=url)

@st.cache(allow_output_mutation=True)
def load_model(name):
    nlp = spacy.load(name)
    ruler = spacy.pipeline.EntityRuler(nlp)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler)
    return nlp

@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)

@st.cache(allow_output_mutation=True)
def url_to_text(url):
    return combined(u(url))


st.sidebar.title("Interactive WSB sentiment visualizer")
st.sidebar.markdown(
    """
Process text with spacy models and visualize named entities,
dependencies and more. Uses spaCy's built-in
[displaCy](http://spacy.io/usage/visualizers) visualizer under the hood.
"""
)

spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()

url = st.text_area("Reddit URL to analyze", "https://www.reddit.com/r/wallstreetbets/comments/fmh129")
text = url_to_text(url)
doc = process_text(spacy_model, text)


st.header("Named Entities")
st.sidebar.header("Named Entities")
label_set = nlp.get_pipe("ner").labels
labels = st.sidebar.multiselect(
    "Entity labels", options=label_set, default=list(label_set)
)
html = displacy.render(doc, style="ent", options={"ents": labels})
# Newlines seem to mess with the rendering
html = html.replace("\n", " ")
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
attrs = ["text", "label_", "ent_id_", "start", "end", "start_char", "end_char"]
if "entity_linker" in nlp.pipe_names:
    attrs.append("kb_id_")
data = [
    [str(getattr(ent, attr)) for attr in attrs]
    for ent in doc.ents
    #if ent.label_ in labels
]
df = pd.DataFrame(data, columns=attrs)
st.dataframe(df)


st.header("Token attributes")
attrs = [
    "idx",
    "text",
    "lemma_",
    "pos_",
    "tag_",
    "dep_",
    "head",
    "ent_type_",
    "ent_iob_",
    "shape_",
    "is_alpha",
    "is_ascii",
    "is_digit",
    "is_punct",
    "like_num",
]
data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
df = pd.DataFrame(data, columns=attrs)
st.dataframe(df)


st.header("JSON Doc")
if st.button("Show JSON Doc"):
    st.json(doc.to_json())

st.header("JSON model meta")
if st.button("Show JSON model meta"):
    st.json(nlp.meta)

'''
for submission in wsb.stream.submissions():
    submission.author
    submission.id
    submission.title
    submission.created_utc
    submission.permalink
    submission.score
    submission.selftext
    sub = nlp(combined(submission))
    list(submission.comments)
'''