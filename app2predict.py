import pandas as pd
import re
import streamlit as st
import plotly.express as px
from textblob import TextBlob
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Load Datasets (Replace with actual dataset paths)
twitter_df = pd.read_csv(r"C:\Users\B.L\Desktop\Python\357\twitter_dataset.csv", encoding='ISO-8859-1')
reddit_df = pd.read_csv(r"C:\Users\B.L\Desktop\Python\357\reddit_dataset.csv", encoding='ISO-8859-1')
facebook_df = pd.read_csv(r"C:\Users\B.L\Desktop\Python\357\facebook_dataset.csv", encoding='ISO-8859-1')

# Text Cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

# Translate Text (Multilingual Support)
def translate_text(text, source='auto', target='en'):
    return GoogleTranslator(source=source, target=target).translate(text)

# Sentiment Analysis
def analyze_sentiment(text, lang='en'):
    if lang != 'en':
        text = translate_text(text, lang, 'en')
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

# Mental Health Detection
def detect_mental_health(text):
    text = text.lower()
    depression_keywords = ["depressed", "hopeless", "sad", "down", "unhappy", "suicidal"]
    anxiety_keywords = ["anxious", "nervous", "stressed", "worried", "panic"]
    burnout_keywords = ["exhausted", "overwhelmed", "tired", "burnout"]
    
    if any(word in text for word in depression_keywords):
        return "Depression Risk"
    elif any(word in text for word in anxiety_keywords):
        return "Anxiety Risk"
    elif any(word in text for word in burnout_keywords):
        return "Burnout Risk"
    else:
        return "No Concern"

# User Behavior Prediction
def predict_user_behavior(sentiment, mental_health):
    if mental_health in ["Depression Risk", "Anxiety Risk", "Burnout Risk"]:
        return "Support-Seeking"
    elif sentiment == "Positive":
        return "Positive Contributor"
    elif sentiment == "Neutral":
        return "Passive Observer"
    else:
        return "Highly Engaged"

# Streamlit App
st.set_page_config(page_title="Social Media Analysis", layout="wide")
st.title("📊 Social Media Sentiment & User Behavior Analysis")
st.markdown("---")

platform = st.sidebar.selectbox("Choose a platform:", ["Twitter", "Reddit", "Facebook"])
sentiment_filter = st.sidebar.multiselect("Filter by Sentiment:", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])
mental_health_filter = st.sidebar.multiselect("Filter by Mental Health Risk:", ["Depression Risk", "Anxiety Risk", "Burnout Risk", "No Concern"], default=["Depression Risk", "Anxiety Risk", "Burnout Risk", "No Concern"])
user_behavior_filter = st.sidebar.multiselect("Filter by User Behavior:", ["Support-Seeking", "Highly Engaged", "Positive Contributor", "Passive Observer"], default=["Support-Seeking", "Highly Engaged", "Positive Contributor", "Passive Observer"])

# Data Analysis
def analyze_data(platform):
    if platform == "Twitter":
        data = twitter_df
    elif platform == "Reddit":
        data = reddit_df
    else:
        data = facebook_df
    
    text_column = "body" if "body" in data.columns else "text" if "text" in data.columns else None
    if text_column is None:
        st.error("No valid text column found in dataset.")
        return []
    
    results = []
    for idx, row in data.iterrows():
        cleaned_text = clean_text(row[text_column])
        sentiment = analyze_sentiment(cleaned_text)
        mental_health = detect_mental_health(cleaned_text)
        user_behavior = predict_user_behavior(sentiment, mental_health)
        results.append({
            "id": idx,
            "text": row[text_column],
            "sentiment": sentiment,
            "mental_health": mental_health,
            "user_behavior": user_behavior
        })
    return pd.DataFrame(results)

if st.button("Analyze Sentiment & Behavior"):
    df = analyze_data(platform)
    if not df.empty:
        df = df[df["sentiment"].isin(sentiment_filter) & df["mental_health"].isin(mental_health_filter) & df["user_behavior"].isin(user_behavior_filter)]
        st.write(df)
        
        # Sentiment Pie Chart
        sentiment_colors = {"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"}
        fig1 = px.pie(df, names="sentiment", title="Sentiment Distribution", color="sentiment", color_discrete_map=sentiment_colors)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Mental Health Bar Chart
        fig2 = px.bar(df, x="mental_health", title="Mental Health Status", color="mental_health", color_discrete_sequence=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"])
        st.plotly_chart(fig2, use_container_width=True)
        
        # User Behavior Bar Chart
        fig3 = px.bar(df, x="user_behavior", title="User Behavior Trends", color="user_behavior", color_discrete_sequence=["#9467bd", "#8c564b", "#e377c2", "#17becf"])
        st.plotly_chart(fig3, use_container_width=True)
        
        # Word Cloud for Most Common Words
        all_text = ' '.join(df['text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        st.subheader("Word Cloud of Most Common Words")
        fig4, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig4)

# To Run: streamlit run this_script.py