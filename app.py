import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import io
import base64

# Define the model and tokenizer
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
st.set_page_config(layout="wide")

# streamlit part
@st.cache_data


#defs
def classify_reviews(reviews):
    inputs = tokenizer(reviews, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.logits.tolist()

def top_rating(scores):
    return scores.index(max(scores)) + 1  # add 1 because star ratings start from 1

def top_prob(scores):
    return max(scores)

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'

def main():
    st.title('Sentiment Analysis with BERT')
    st.markdown('Upload an Excel file with a column named "reviews" to get sentiment analysis.')

    file = st.file_uploader("Upload an excel file", type=['xlsx'])

    if file is not None:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            st.write("An error occurred while reading the uploaded file. Please make sure it's a valid Excel file.")
            return

        if 'reviews' in df.columns:
            with st.spinner('Performing sentiment analysis...'):
                df, df_display = process_reviews(df)

            display_ratings(df)
            display_dataframe(df, df_display)
        else:
            st.write('No column named "reviews" found in the uploaded file.')

def process_reviews(df):
    raw_scores = classify_reviews(df['reviews'].tolist())
    prob_scores = [F.softmax(torch.tensor(score), dim=0).tolist() for score in raw_scores]

    df['raw_scores'] = raw_scores
    df['prob_scores'] = prob_scores

    for i in range(1, 6):
        df[f'{i} Star'] = df['prob_scores'].apply(lambda scores: scores[i-1]).round(2)

    df['Rating'] = df['prob_scores'].apply(top_rating)
    df['Probability'] = df['prob_scores'].apply(top_prob).round(2)

    df = df.drop(columns=['raw_scores', 'prob_scores'])
    cols = ['reviews', 'Rating', 'Probability'] + [f'{i} Star' for i in range(1, 6)]
    df = df.reindex(cols, axis=1)

    df_display = df.copy()

    for i in range(1, 6):
        df_display[f'{i} Star'] = df_display[f'{i} Star'].apply(lambda x: f'{x*100:.0f}%')

    df_display['Probability'] = df_display['Probability'].apply(lambda x: f'{x*100:.0f}%')

    return df, df_display


def convert_df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')

def display_dataframe(df, df_display):
    csv = convert_df_to_csv(df)

    # Create two columns for the buttons
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    # Place the download button in the first column
    with col1:
        show_data = st.button('Show all data')

    # Show all data button in the second column
    with col2:
        st.download_button(
            "Download CSV",
            csv,
            "data.csv",
            "text/csv",
            key='download-csv'
        )
        

    # Call st.dataframe outside of the column blocks to display the dataframe across the full width of the page
    if show_data:
        st.dataframe(df_display)
    else:
        st.dataframe(df_display.head())





def display_ratings(df):
    cols = st.columns(5)  # create 5 columns

    for i in range(1, 6):
        rating_counts = df[df['Rating'] == i].shape[0]  # count of each rating
        cols[i-1].markdown(f"### {i} Star")
        cols[i-1].markdown(f"**Count:** {rating_counts}")



if __name__ == "__main__":
    main()
