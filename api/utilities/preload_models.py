from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForTokenClassification

# Sentiment
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Zero-shot intent classification
# zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Named Entity Recognition
# ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
# ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
# ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
