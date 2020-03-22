# Setup notes:
`
python -m venv env
source env/bin/activate
`

make sure .env is configured! (template .env.template included)
make sure nltk libs are setup!

`
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
`

# to run:
python anal.py

# when done modifying:
pip freeze > requirements.txt