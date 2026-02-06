import pandas as pd
import numpy as np
import pickle

def calculate_es_accuracy(all_results):
    """
    Find the proportion of results from ES that have the correct title in the results.

    Think about this as a ceiling on the ranker's accuracy: if the correct article
    is not in the ES results, it's impossible for the ranker to get it right.

    Handle Nones as well: if the correct title is None, it's always possible.
    """
    possible_200 = []
    possible_100 = []
    possible_50 = []
    error = []
    for df in all_results:
        if df.shape[0] == 0:
            error.append(True)
            continue
        correct_title = df.iloc[0]['correct_title']
        # replace "", None, "None", and nan with None
        if correct_title is None or correct_title == "None" or correct_title == "" or correct_title == "nan" or pd.isna(correct_title):
            correct_title = None
        if correct_title is None:
            possible_200.append(True)
            possible_100.append(True)
            possible_50.append(True)
        else:
            if df.iloc[0]['correct_title'] in df['title'].values:
                possible_200.append(True)
            if correct_title in df['title'].values[0:100]:
                possible_100.append(True)
            if correct_title in df['title'].values[0:50]:
                possible_50.append(True)
    mean_possible_200 = np.sum(possible_200) / len(all_results)
    mean_possible_100 = np.sum(possible_100) / len(all_results)
    mean_possible_50 = np.sum(possible_50) / len(all_results)
    return (mean_possible_200, mean_possible_100, mean_possible_50), error

import pickle
with open("wiki_results.pkl", "rb") as f:
    all_results = pickle.load(f)

with open("wiki_scores.pkl", "rb") as f:
    score_dfs = pickle.load(f)

possible, error = calculate_es_accuracy(all_results)
print(possible[2])
# 0.863 (50 results)
# 0.883 (100 results)
# 0.868 (fixed correct titles whitespace)
# 0.898 (more minor fixes)
# 0.902 (new data)
# 0.946 (final data cleaning)


# find the queries that had no results
no_results = []
for n, df in enumerate(all_results):
    if df.shape[0] == 0:
        no_results.append(df)
        continue
    correct_title = df.iloc[0]['correct_title']
    # replace "", None, "None", and nan with None
    if correct_title is None or correct_title == "none" or correct_title == "None" or correct_title == "" or correct_title == "nan" or pd.isna(correct_title):
        correct_title = None

    if correct_title is None:
        continue
    elif df.iloc[0]['correct_title'] not in df['title'].values: 
        no_results.append(df.iloc[0])
    else:
        continue

no_results_df = pd.concat(no_results, axis=1).T
no_results_df[['query_term', 'correct_title']].to_markdown()

formatted = []
for i in no_results:
    d = f"Query term orig: {i['query_term_orig']}\nQuery term cleaned: {i['query_term']}\ncorrect title: {i['correct_title']}\n"
    formatted.append(d)
formatted = "\n\n".join(formatted)

# Save the results to a txt file
with open("no_results.txt", "w") as f:
    f.write(formatted)
