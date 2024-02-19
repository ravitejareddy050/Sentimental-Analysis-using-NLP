import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Sentiment Analysis NLP App", page_icon=":guardsman:", layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df

def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)
        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)
    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

def main():
    st.title("Sentiment Analysis NLP App")
    st.write("Enter Text to Analyze Sentiment")
    st.write("This app uses TextBlob and VADER to analyze the sentiment of text input.")

    raw_text = st.text_area("", height=200)
    if st.button("Analyze"):
        st.write("Results")
        sentiment = TextBlob(raw_text).sentiment
        st.write(sentiment)

        if sentiment.polarity > 0:
            st.markdown("Sentiment: Positive 😊")
        elif sentiment.polarity < 0:
            st.markdown("Sentiment: Negative 😠")
        else:
            st.markdown("Sentiment: Neutral 😐")

        result_df = convert_to_df(sentiment)
        st.dataframe(result_df)

        c = alt.Chart(result_df).mark_bar().encode(
            x='metric',
            y='value',
            color='metric'
        ).properties(width=500, height=250)

        st.altair_chart(c, use_container_width=True)

        st.write("Token Sentiment")
        token_sentiments = analyze_token_sentiment(raw_text)
        st.write(token_sentiments)

       

if __name__ == '__main__':
    main()
