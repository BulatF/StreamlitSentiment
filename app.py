import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import io
import base64
from stqdm import stqdm

# Define the model and tokenizer
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
st.set_page_config(layout="wide")

#defs
def classify_reviews(reviews):
    inputs = tokenizer(reviews, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1).tolist()  
    return probabilities

def top_rating(scores):
    return scores.index(max(scores)) + 1  

def top_prob(scores):
    return max(scores)

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'

def main():
    st.title('Sentiment Analysis')
    st.markdown('Upload an Excel file and select a column to get sentiment analysis.')

    file = st.file_uploader("Upload an excel file", type=['xlsx'])
    review_column = None
    df = None

    if file is not None:
        try:
            df = pd.read_excel(file)
            review_column = st.selectbox('Select the reviews column', df.columns)
            df[review_column] = df[review_column].astype(str)
        except Exception as e:
            st.write("An error occurred while reading the uploaded file. Please make sure it's a valid Excel file.")
            return

    start_button = st.button('Start Analysis')

    if start_button and df is not None:
        if review_column in df.columns:
            with st.spinner('Performing sentiment analysis...'):
                df, df_display = process_reviews(df, review_column)

            display_ratings(df)
            display_dataframe(df, df_display)
        else:
            st.write(f'No column named "{review_column}" found in the uploaded file.')
            

def process_reviews(df, review_column):
    with st.spinner('Classifying reviews...'):
        progress_bar = st.progress(0)
        total_reviews = len(df[review_column].tolist())
        review_counter = 0

        batch_size = 50
        raw_scores = []
        reviews = df[review_column].tolist()
        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i:i+batch_size]
            batch_scores = classify_reviews(batch_reviews)
            raw_scores.extend(batch_scores)
            review_counter += len(batch_reviews)
            progress_bar.progress(review_counter / total_reviews)

    df_new = df.copy()
    df_new['raw_scores'] = raw_scores
    scores_to_df(df_new)
    df_display = scores_to_percent(df_new.copy())

    # Get all columns excluding the created ones and the review_column
    remaining_columns = [col for col in df.columns if col not in [review_column, 'raw_scores', 'Weighted Rating', 'Rating', 'Probability', '1 Star', '2 Star', '3 Star', '4 Star', '5 Star']]

    # Reorder the dataframe with selected columns first, created columns next, then the remaining columns
    df_new = df_new[[review_column, 'Weighted Rating', 'Rating', 'Probability', '1 Star', '2 Star', '3 Star', '4 Star', '5 Star'] + remaining_columns]

    # Reorder df_display as well
    df_display = df_display[[review_column, 'Weighted Rating', 'Rating', 'Probability', '1 Star', '2 Star', '3 Star', '4 Star', '5 Star'] + remaining_columns]

    return df_new, df_display


def scores_to_df(df):
    for i in range(1, 6):
        df[f'{i} Star'] = df['raw_scores'].apply(lambda scores: scores[i-1]).round(2)

    df['Rating'] = df['raw_scores'].apply(top_rating)
    df['Probability'] = df['raw_scores'].apply(top_prob).round(2)
    # Compute the Weighted Rating
    df['Weighted Rating'] = sum(df[f'{i} Star']*i for i in range(1, 6))
    
    df.drop(columns=['raw_scores'], inplace=True)

def scores_to_percent(df):
    for i in range(1, 6):
        df[f'{i} Star'] = df[f'{i} Star'].apply(lambda x: f'{x*100:.0f}%')

    df['Probability'] = df['Probability'].apply(lambda x: f'{x*100:.0f}%')

    return df

def convert_df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')

def display_dataframe(df, df_display):
    csv = convert_df_to_csv(df)

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    with col1:
        st.download_button(
            "Download CSV",
            csv,
            "data.csv",
            "text/csv",
            key='download-csv'
        )

    st.dataframe(df_display)

def display_ratings(df):
    cols = st.columns(5)

    for i in range(1, 6):
        rating_counts = df[df['Rating'] == i].shape[0]
        cols[i-1].markdown(f"### {rating_counts}")
        cols[i-1].markdown(f"{'⭐' * i}")



if __name__ == "__main__":
    main()
