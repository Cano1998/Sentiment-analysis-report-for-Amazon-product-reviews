import pandas as pd
import spacy
from textblob import TextBlob
from fpdf import FPDF

# Load NLP en_core_we_sm
nlp = spacy.load("en_core_web_sm")

# Load dataset with low_memory=False to handle mixed types
df = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Drop the 'reviews.text' column
reviews_column = df['reviews.text']
clean_column = reviews_column.dropna()

# Remove stopwords and clean the text
def clean_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

clean_column = clean_column.apply(clean_text)

# Sentiment analysis function
def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive", sentiment_score
    elif sentiment_score < 0:
        return "Negative", sentiment_score
    else:
        return "Neutral", sentiment_score

# Sentiment analysis on a number of sample product reviews
sample_reviews = clean_column.sample(7)
for review in sample_reviews:
    sentiment, score = sentiment_analysis(review)
    print(f"Review: {review}\nSentiment: {sentiment}\nScore: {score}\n")

# Report (PDF file)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Sentiment Analysis Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

# Creating the PDF report
pdf = PDF()
pdf.set_left_margin(10)
pdf.set_right_margin(10)

# Chapters into the PDF file
pdf.add_chapter("Description of the dataset used", "There are user reviews of Amazon products in this dataset.")
pdf.add_chapter("Details of the preprocessing steps", 
                "The following preprocessing steps were performed:\n1. Removed missing values.\n2. Tokenized and lemmatized text.\n3. Removed stopwords and punctuation.")
pdf.add_chapter("Evaluation of results", "To verify that the predictions were accurate, sentiment analysis was used to sample reviews. The following are the outcomes:\n")

# Sample reviews and sentiment analysis into the PDF file
for review in sample_reviews:
    sentiment, score = sentiment_analysis(review)
    pdf.chapter_body(f"Review: {review}\nSentiment: {sentiment}\nScore: {score}\n")

pdf.add_chapter("Insights into the model's strengths and limitations", 
                "The model separates neutral, negative, and positive reviewes with accuracy. It could be limited, though, when dealing with delicate words or complicated feelings.")

# Save the PDF file
pdf.output("sentiment_analysis_report.pdf")

print("The sentiment analysis report has been created.")