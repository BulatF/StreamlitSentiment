import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import base64

# Define the model and tokenizer
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def classify_review(review):
    inputs = tokenizer.encode_plus(review, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.logits[0].tolist()

def top_rating(scores):
    return scores.index(max(scores)) + 1  # add 1 because star ratings start from 1

def top_prob(scores):
    return max(scores)

def get_table_download_link(df):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'

def main():
    st.title('Sentiment Analysis with BERT')
    file = st.file_uploader("Upload an excel file", type=['xlsx'])

    if file is not None:
        # read the file into a DataFrame
        df = pd.read_excel(file)
        if 'reviews' in df.columns:
            # Perform the analysis
            df['raw_scores'] = df['reviews'].apply(classify_review)
            df['prob_scores'] = df['raw_scores'].apply(lambda scores: F.softmax(torch.tensor(scores), dim=0).tolist())
            
            for i in range(1, 6):
                df[f'{i} Star'] = df['prob_scores'].apply(lambda scores: scores[i-1]).round(2)

            df['Rating'] = df['prob_scores'].apply(top_rating)
            df['Probability'] = df['prob_scores'].apply(top_prob).round(2)

            # Remove the raw_scores and prob_scores columns
            df = df.drop(columns=['raw_scores', 'prob_scores'])

            # Rearrange columns
            cols = ['reviews', 'Rating', 'Probability'] + [f'{i} Star' for i in range(1, 6)]
            df = df.reindex(cols, axis=1)

            # Create a fancy bar chart of ratings using Plotly
            rating_counts = df['Rating'].value_counts().sort_index()
            fig = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values, labels={'x':'Star Rating', 'y':'Count'}, title="Review Counts by Star Rating")
            fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
            fig.update_layout(title_x=0.5, xaxis=dict(tickmode='linear', tick0=1, dtick=1))
            fig.update_layout(
                autosize=True,
                hovermode="closest",
                plot_bgcolor="white",
                showlegend=False,
                modebar=dict(orientation="v", bgcolor="rgba(0,0,0,0)", color="darkgray", activecolor="rgb(97, 97, 97)")
            )

            st.plotly_chart(fig)

            # Show the DataFrame in Streamlit
            st.write(df)

            # Add a link to download the DataFrame
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        else:
            st.write('No column named "reviews" found in the uploaded file.')
    
if __name__ == "__main__":
    main()
