from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

ROLLING_MEAN_SENTIMENT = int(168 * 1)
MARKERS = True
POLITICO = "kamala_harris"

START_DATE = '2024-08-15'
END_DATE = '2026-11-15'

# Load data (you might need to adjust this based on how you're storing your data on Koyeb)
ig_comments_df = pd.read_csv('ig_comments.csv', low_memory=False)
users_df = pd.read_csv('users.csv')
markers_df = pd.read_csv('markers.csv')

comment_counts = ig_comments_df['comment_user'].value_counts()
users_with_few_comments = comment_counts[comment_counts <= 3].index
ig_comments_df = ig_comments_df[ig_comments_df['comment_user'].isin(users_with_few_comments)]

def process_user_data(ig_name, rolling_window):
    filtered_df = ig_comments_df.loc[
        (ig_comments_df['original_poster'] == ig_name) & 
        ig_comments_df['sentiment'].notna(),
        ['comment_datetime', 'sentiment', 'likes']
    ]
    filtered_df[['sentiment', 'likes']] = filtered_df[['sentiment', 'likes']].astype(float)
    
    filtered_df['comment_datetime'] = pd.to_datetime(filtered_df['comment_datetime'], format='mixed')
    hourly_total_likes = filtered_df.set_index('comment_datetime').resample('1h')['likes'].sum()
    
    filtered_df['likes_relativity'] = filtered_df.apply(lambda row: row['likes'] / hourly_total_likes[row['comment_datetime'].floor('h')] if row['comment_datetime'].floor('h') in hourly_total_likes.index else 0, axis=1)
    
    filtered_df['weighted_sentiment'] = filtered_df['sentiment'] * filtered_df['likes_relativity']
    
    hourly_sentiment = filtered_df.set_index('comment_datetime').resample('1h').agg({
        'sentiment': 'mean',
        'weighted_sentiment': 'mean'
    })
    hourly_sentiment = hourly_sentiment.ffill()
    
    hourly_sentiment = 50 * (hourly_sentiment + 1)
    
    rolling_mean = hourly_sentiment.rolling(window=rolling_window, min_periods=1).mean()
    
    SMOOTHING_WINDOW = 2
    
    return rolling_mean.rolling(SMOOTHING_WINDOW).mean()

markers_df['date'] = pd.to_datetime(markers_df['date'])
ig_comments_df['comment_datetime'] = pd.to_datetime(ig_comments_df['comment_datetime'], format='mixed')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    show_markers = request.args.get('show_markers', 'true').lower() == 'true'
    rolling_window = int(request.args.get('rolling_window', ROLLING_MEAN_SENTIMENT))
    start_date = request.args.get('start_date', START_DATE)
    end_date = request.args.get('end_date', END_DATE)

    user_data = {user_row['full_name']: process_user_data(user_row['ig_user'], rolling_window)
                 for _, user_row in users_df[users_df['full_name'] == POLITICO].iterrows()}

    chart_data = []
    min_y, max_y = float('inf'), float('-inf')
    for user, data in user_data.items():
        filtered_data = data.loc[start_date:end_date]
        unweighted = filtered_data['sentiment'].reset_index().values.tolist()
        weighted = filtered_data['weighted_sentiment'].reset_index().values.tolist()
        
        min_y = min(min_y, filtered_data['sentiment'].min(), filtered_data['weighted_sentiment'].min())
        max_y = max(max_y, filtered_data['sentiment'].max(), filtered_data['weighted_sentiment'].max())
        
        chart_data.append({
            'name': f"{user} (Unweighted)",
            'data': [[int(d.timestamp() * 1000), float(s) if not np.isnan(s) else None] for d, s in unweighted]
        })
        chart_data.append({
            'name': f"{user} (Weighted)",
            'data': [[int(d.timestamp() * 1000), float(s) if not np.isnan(s) else None] for d, s in weighted]
        })

    markers = []
    if show_markers:
        markers = markers_df[
            (markers_df['date'] >= start_date) & 
            (markers_df['date'] <= end_date)
        ].to_dict('records')

    return jsonify({
        'series': chart_data,
        'markers': [{
            'x': int(m['date'].timestamp() * 1000),
            'text': m['description']
        } for m in markers],
        'yaxis': {
            'min': max(0, min_y - 5) if not np.isinf(min_y) else 0,
            'max': min(100, max_y + 5) if not np.isinf(max_y) else 100
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)