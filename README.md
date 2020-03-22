# Setup notes:
You should be using venv:
```
python -m venv env
source env/bin/activate
```

make sure .env is configured (template .env.template included):
```
cp .env.template .env
```

make sure nltk libs are setup:
```
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
```

# To run:
```
python anal.py
```

# When done modifying:
Make sure requirements are updated!
```
pip freeze > requirements.txt
```