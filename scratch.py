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
    
spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()


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