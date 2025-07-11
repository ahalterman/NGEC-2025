### TODO
# - then also use the original, unmodified search term for similarity
# - (also look for exact matches for the original??)


import logging
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import io
from contextlib import contextmanager
import os

from actor_resolution import WikiMatcher
from actor_resolution import TextPreProcessor, CountryDetector

import spacy
nlp = spacy.load("en_core_web_lg")


# Initialize WikiMatcher after logger setup
wiki_matcher = WikiMatcher(device='cuda:0',
                           nlp=nlp)
# Initialize TextPreProcessor
text_preprocessor = TextPreProcessor()
country_detector = CountryDetector()


# This list will store logs for each example
all_example_logs = []
# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Disable specific loggers you don't want to see
logging.getLogger('elasticsearch').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('elasticsearch.trace').setLevel(logging.WARNING)

# Clear all existing handlers from the root logger to avoid duplicates
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Get only the loggers you want to capture (don't include root logger)
loggers_to_capture = [
    logging.getLogger('actor_resolution'),  # The main module
    logging.getLogger('wiki_lookup'),       # Your custom logger
    logging.getLogger('__main__')           # Main module logger
]

@contextmanager
def capture_logs_for_example():
    """Context manager to capture logs for a single example"""
    # Create a new string buffer for this example
    log_stream = io.StringIO()
    
    # Create a handler that writes to this buffer
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to ALL relevant loggers
    handlers_added = []
    for logger in loggers_to_capture:
        logger.addHandler(handler)
        handlers_added.append((logger, handler))
    
    try:
        # Yield control back to the caller
        yield
    finally:
        # Get the log contents
        log_contents = log_stream.getvalue()
        
        # Add to our collection of example logs
        all_example_logs.append(log_contents)
        
        # Remove the handler from all loggers
        for logger, h in handlers_added:
            logger.removeHandler(h)
        log_stream.close()

def filter_log(log_text, exclude_patterns=None, max_line_length=200):
    """
    Filter a log string to remove unwanted content
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    # Split by lines
    lines = log_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip lines matching exclude patterns
        if any(re.search(pattern, line) for pattern in exclude_patterns):
            continue
            
        # Truncate long lines
        if len(line) > max_line_length:
            line = line[:max_line_length] + "... [truncated]"
            
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

if 'wiki_gold_standard.csv' in os.listdir():
    wiki_df = pd.read_csv("wiki_gold_standard.csv")
else:
    # Load the CSV from Google sheets
    url = "https://docs.google.com/spreadsheets/d/1wWKi3-7kb8J1bZbfqbdlsjjI4VpxSNLRX6KEffoPPQk/export?format=csv"
    wiki_df = pd.read_csv(url)
    wiki_df.to_csv("wiki_gold_standard.csv", index=False)

wiki_df2 = pd.read_csv("wiki_resolution_tests3.csv")
wiki_df = pd.concat([wiki_df, wiki_df2])
wiki_df['Correct Wiki Title'] = wiki_df['Correct Wiki Title'].str.strip()
wiki_df['Correct Wiki Title'] = wiki_df['Correct Wiki Title'].replace({'none': 'None'})

# Define patterns to exclude (customize these based on your log output)
exclude_patterns = [
    r"Top 30 intro paras",  # Long paragraphs
    r"Similarity scores:", # Raw similarity scores
    r"intro_para.*?", # Long intro paragraphs from Wikipedia
]

def get_context(text_input, actor_text_for_wiki):
    # Get the 200 characters from each side of the query from the document
    actor_text_for_wiki = re.sub(r'\s+', ' ', actor_text_for_wiki).strip()
    
    # Find position in text (handle case where query might not be in text)
    query_pos = text_input.find(actor_text_for_wiki)
    if query_pos >= 0:
        start_index = max(0, query_pos - 200)
        end_index = min(len(text_input), query_pos + len(actor_text_for_wiki) + 200)
        passage_context = text_input[start_index:end_index]
    else:
        passage_context = text_input[:400]  # Just take first 400 chars
    return passage_context


output_list = []
filtered_logs = []
all_example_logs = []
parsed_input_list = []

PREPROCESS = True

for index, row in tqdm(wiki_df.iterrows(), total=wiki_df.shape[0]):
    # Extract the relevant columns
    text_input = row['Document']
    actor_text_for_wiki = row['Query']
    correct_title = row['Correct Wiki Title']
    
    passage_context = get_context(text_input, actor_text_for_wiki)
    country_context = None
    country_context, _ = country_detector.search_nat(passage_context, use_name=True)
    orig = actor_text_for_wiki
    if PREPROCESS:
        # Preprocess the text input
        parsed_input = text_preprocessor.extract_entity_components(actor_text_for_wiki, nlp)
        parsed_input['orig'] = actor_text_for_wiki
        orig = actor_text_for_wiki
        parsed_input_list.append(parsed_input)
        if parsed_input['core_entity'] is not None:
            actor_text_for_wiki = parsed_input['core_entity']
            #text_input = f"{text_input}\n\nMOST IMPORTANT INFO: {parsed_input['role']} {parsed_input['core_entity']}"

    # Query the wiki matcher with logging
    with capture_logs_for_example():
        # Add a debug message to verify logging is working
        logging.debug(f"Starting WikiMatcher query for: {actor_text_for_wiki}")
        
        try:
            if PREPROCESS:
                if parsed_input['role'] is not None:
                    result = wiki_matcher.query_wiki(actor_text_for_wiki, 
                                                     context=text_input, 
                                                     actor_desc=parsed_input['role'],
                                                     country=country_context,
                                                     #context=None,
                                                     #actor_desc=None,
                                                     max_results=50)
                else:
                    result = wiki_matcher.query_wiki(actor_text_for_wiki, 
                                                     context=text_input, 
                                                     country=country_context,
                                                     #context=None,
                                                     max_results=50)
            else:
                result = wiki_matcher.query_wiki(actor_text_for_wiki, 
                                                 context=text_input, 
                                                 country=country_context,
                                                 #context=None,
                                                 max_results=50)
            if result:
                pred_title = result['title']
                wiki_reason = result['wiki_reason']
                logging.debug(f"Found Wikipedia article: {pred_title}")
            else:
                pred_title = None
                wiki_reason = "No good match found"
                logging.debug("No Wikipedia article found")
            error = None
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            pred_title = None
            error = str(e)
            logging.debug(f"Error occurred during lookup: {error}")

    # Verify we captured logs
    last_log = ""
    filtered_log = ""
    if len(all_example_logs) > 0:
        last_log = all_example_logs[-1]
        #print(f"Log captured: {len(last_log)} characters")
        
        # Filter the logs for this example
        filtered_log = filter_log(last_log, exclude_patterns=exclude_patterns, max_line_length=150)
        filtered_logs.append(filtered_log)
    else:
        print("No logs captured for this example")
        filtered_logs.append("")
    
    # Store result
    d = {
        'orig_span': orig,
        'actor_text_for_wiki': actor_text_for_wiki,
        'passage_context': passage_context,
        'pred_title': pred_title,
        'correct_title': correct_title,
        'error': error,
        'log': last_log,
        'wiki_reason': wiki_reason,
        'filtered_log': filtered_log,
        'index': index
    }
    output_list.append(d)


# Create output dataframe
output_df = pd.DataFrame(output_list)
# mix of Nans and Nones--replace all with "None"
output_df['correct_title'] = output_df['correct_title'].replace({np.nan: "None"})
output_df['pred_title'] = output_df['pred_title'].replace({None: "None"})
output_df['pred_title'] = output_df['pred_title'].replace({np.nan: "None"})
output_df['correct_title'] = output_df['correct_title'].replace({None: "None"})

output_df['correct'] = output_df['correct_title'] == output_df['pred_title']
print(f"\nAccuracy: {output_df['correct'].mean()}")

# calculate accuracy by wiki_reason (and record the number of examples)
# sort by size
accuracy_by_reason = output_df.groupby('wiki_reason').agg(
    accuracy=('correct', 'mean'),
    count=('correct', 'size')
).sort_values('count', ascending=False)
print(accuracy_by_reason)

picked_by_score = output_df[output_df['wiki_reason'] == "Best overall combined score"]
overall_score = []
for index, row in picked_by_score.iterrows():
    log = row['filtered_log']
    try:
        score = re.findall(r"Returning best overall combined score\: [^0-9]+(\d+\.\d+)\n", log)[0]
    except:
        score = np.nan
    overall_score.append(score)
picked_by_score['overall_score'] = overall_score
picked_by_score['overall_score'] = picked_by_score['overall_score'].replace({None: np.nan})
picked_by_score['overall_score'] = picked_by_score['overall_score'].astype(float)
# calculate average by correct
picked_by_score.groupby(picked_by_score['correct']).agg(
    accuracy=('overall_score', 'mean'),
    count=('overall_score', 'size')
).sort_values('count', ascending=False)

picked_by_score = picked_by_score[['correct', 'pred_title', 'correct_title', 'overall_score']]
picked_by_score.to_csv("picked_by_score.csv", index=False)


# delete 'log' column to save space
output_df = output_df.drop(columns=['log'])
# reorder columns
output_df = output_df[['passage_context', 'actor_text_for_wiki', 'pred_title', 'correct_title', 'correct', 'wiki_reason', 'filtered_log', 'error']]

output_df.to_csv("wiki_lookup_results.csv", index=False)

# Sample incorrect prediction
if len(output_df[output_df['correct'] == False]) > 0:
    print("\nSample incorrect prediction:")
    print(output_df[output_df['correct'] == False].iloc[0][['actor_text_for_wiki', 'pred_title', 'correct_title']])


### Get the scoring dataframe for the best match

#### SCRATCH TESTING

context = """device translates into Braille for him. But for major games, he calls on de Araujo and others like him. The technique is so good that Junior even knew in previous games when Neymar fell down, or when Brazil coach Tite hurt himself while celebrating the team's win over Costa Rica.

“Even though they (deaf and blind people) don’t have access to lots of information, that doesn’t hinder their lives,” de Araujo said."""

query_term = "Tite"
query_term = wiki_matcher._expand_query(query_term, context)

results = results = wiki_matcher.wiki_searcher.search_wiki(
            query_term, 
            limit_term=None, 
            fuzziness="AUTO", 
            max_results=50,
        )

titles = [i['title'] for i in results]

np.argmax([i['title'] == "Tite (football manager)" for i in results])

scores = wiki_matcher._create_scoring_dataframe(results, 
                                                query_term,
                                                context=context,
                                                country="Brazil",
                                                actor_desc=None)

best = wiki_matcher._apply_selection_rules(scores, 
                                 results,
                                 context=context)
best['title']

best = wiki_matcher.query_wiki(query_term,
                                 context=context,
                                 country="Brazil",
                                 actor_desc=None,
                                 max_results=50)



#### Get results and raw score dfs for all examples ###

score_dfs = []
all_results = []
for index, row in tqdm(wiki_df.iterrows(), total=wiki_df.shape[0]):
    text_input = row['Document']
    actor_text_for_wiki = row['Query']
    correct_title = row['Correct Wiki Title']
    
    passage_context = get_context(text_input, actor_text_for_wiki)
    country_context = None
    actor_desc = None
    country_context, _ = country_detector.search_nat(passage_context, use_name=True)
    orig = actor_text_for_wiki
    if PREPROCESS:
        # Preprocess the text input
        parsed_input = text_preprocessor.extract_entity_components(actor_text_for_wiki, nlp)
        parsed_input['orig'] = actor_text_for_wiki
        orig = actor_text_for_wiki
        parsed_input_list.append(parsed_input)
        if parsed_input['core_entity'] is not None:
            actor_text_for_wiki = parsed_input['core_entity']
        if parsed_input['role'] is not None:
            actor_desc = parsed_input['role']
            #text_input = f"{text_input}\n\nMOST IMPORTANT INFO: {parsed_input['role']} {parsed_input['core_entity']}"
    if text_input:
        actor_text_for_wiki = wiki_matcher._expand_query(actor_text_for_wiki, text_input)
    try:
        results = wiki_matcher.wiki_searcher.search_wiki(
                                        actor_text_for_wiki, 
                                        limit_term=None, 
                                        fuzziness='AUTO', 
                                        max_results=50,
                                    )
        score_df = wiki_matcher._create_scoring_dataframe(results, 
                                            actor_text_for_wiki,
                                            context=text_input,
                                            country=country_context,
                                            actor_desc=actor_desc)
    
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        pred_title = None
        error = str(e)
        logging.debug(f"Error occurred during lookup: {error}")
        continue
    # Save the raw results
    results_df = pd.DataFrame(results)
    results_df['correct_title'] = correct_title
    results_df['query_term'] = actor_text_for_wiki
    results_df['query_term_orig'] = orig
    all_results.append(results_df)
    # save the score_df
    score_df['correct_title'] = correct_title
    score_df['query_term'] = actor_text_for_wiki
    score_df['query_term_orig'] = orig
    if 'title' not in score_df.columns:
        score_df['title'] = None
    score_df['y'] = score_df['title'] == score_df['correct_title'] 
    score_df['task'] = index
    score_df['passage'] = passage_context
    score_dfs.append(score_df)

# res = wiki_matcher.wiki_searcher.wiki_client.conn.query(query)[0:5].execute()

def calculate_es_accuracy(all_results):
    """
    Find the proportion of results from ES that have the correct title in the results.

    Think about this as a ceiling on the ranker's accuracy: if the correct article
    is not in the ES results, it's impossible for the ranker to get it right.

    Handle Nones as well: if the correct title is None, it's always possible.
    """
    possible = []
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
            possible.append(True)
        else:
            if df.iloc[0]['correct_title'] in df['title'].values:
                possible.append(True)
            else:
                possible.append(False)
    return possible, error

possible, error = calculate_es_accuracy(all_results)
print(np.mean(possible))
# 0.863 (50 results)
# 0.883 (100 results)
# 0.868 (fixed correct titles whitespace)
# 0.898 (more minor fixes)
# 0.902 (new data)


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


no_results[33]

formatted = []
for i in no_results:
    d = f"Query term orig: {i['query_term_orig']}\nQuery term cleaned: {i['query_term']}\ncorrect title: {i['correct_title']}\n"
    formatted.append(d)
formatted = "\n\n".join(formatted)

# Save the results to a txt file
with open("no_results.txt", "w") as f:
    f.write(formatted)


all_scores = pd.concat(score_dfs)
#all_scores.to_csv("wiki_scores.csv", index=False)



# Now we want to fit a model to predict the y value
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier

# Get unique task IDs and split them into train and test sets
task_ids = all_scores['task'].unique()
train_tasks, test_tasks = train_test_split(task_ids, test_size=0.2, random_state=42)
train_df_list, test_df_list = [], []
for task in train_tasks:
    train_df_list.append(all_scores[all_scores['task'] == task])
for task in test_tasks:
    test_df_list.append(all_scores[all_scores['task'] == task])

# Filter the dataframe based on the task assignments
train_df = all_scores[all_scores['task'].isin(train_tasks)]
test_df = all_scores[all_scores['task'].isin(test_tasks)]

# Verify the split worked as expected
print(f"Training set: {len(train_tasks)} tasks, {len(train_df)} rows")
print(f"Test set: {len(test_tasks)} tasks, {len(test_df)} rows")


omit_columns = ['y', 'correct_title', 'title', 'combined_score', # 'combined_score_norm',
                'task', 'passage', 'query_term_orig', 'query_term']#, 'title_sim', 'title_sim_norm']

X_train = train_df.drop(columns=omit_columns)
X_test = test_df.drop(columns=omit_columns)
y_train = train_df['y']
y_test = test_df['y']


# Now fit an XGBoost model
xgb_model = XGBClassifier(n_estimators=1000,
                    random_state=42,
                    #max_depth=5,
                    #min_samples_split=2,
                    #min_samples_leaf=1,
                    ####subsample=0.5,
                    ####learning_rate=0.1,
                    class_weight='balanced',
                    n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))

# save the model to disk (safe format)
model_file = 'xgb_model.json'
xgb_model.save_model(model_file)

# load the model from disk
loaded_model = XGBClassifier()
loaded_model.load_model(model_file)

def predict_xgboost(score_df, xgb_model, just_title=False,
                    score_threshold=0.5):
    X = score_df[xgb_model.feature_names_in_]
    y_proba = xgb_model.predict_proba(X)[:, 1]
    score_df['y_pred_proba'] = y_proba

    # Find the max probability for each task
    score_df = score_df.reset_index(drop=True)
    task_max_proba = score_df.groupby('task')['y_pred_proba'].transform('max')
    score_df['is_max_for_task'] = (score_df['y_pred_proba'] == task_max_proba)

    # Create a column where True is the max predicted proba (over 0.5) and False otherwise
    score_df['is_predicted_match'] = score_df['is_max_for_task'] & (score_df['y_pred_proba'] > score_threshold)
    if score_df['is_predicted_match'].sum() == 0:
        print("No predictions made")
        return None
    pred_row = score_df[score_df['is_predicted_match']].copy()
    if just_title:
        return pred_row['title'].values[0]
    else:
        return pred_row
        

predict_xgboost(test_df_list[153], xgb_model, just_title=True)
predict_xgboost(test_df_list[153], xgb_model)

# better eval
# first, get the true titles for each task as a list
def get_true_titles(test_df_list):
    true_titles = []
    for i in test_df_list:
        if i.empty:
            true_titles.append(None)
        else:
            title = i.iloc[0]['correct_title']
            # Replace nan with None
            if pd.isna(title):
                true_titles.append(None)
            elif title == 'none':
                true_titles.append(None)
            else:
                true_titles.append(title)
    return true_titles

true_titles = get_true_titles(test_df_list)
#pred_titles = [predict_xgboost(i, xgb_model, just_title=True) for i in tqdm(test_df_list)]
predictions = [predict_xgboost(i, xgb_model) for i in tqdm(test_df_list)]
pred_titles = []
for i in predictions:
    if i is not None:
        pred_titles.append(i['title'].values[0])
    else:
        pred_titles.append(None)

pred_probas = []
for i in predictions:
    if i is not None:
        pred_probas.append(i['y_pred_proba'].values[0])
    else:
        pred_probas.append(None)


# calculate accuracy
pred_titles = np.array(pred_titles)
true_titles = np.array(true_titles)
correct = pred_titles == true_titles
np.mean(correct)
# np.float64(0.734)

for n, pred in enumerate(pred_titles):
    true = true_titles[n]
    correct = true == pred
    if correct:
        continue
    d = {"query_term": test_df_list[n]['query_term'].values[0],
         #"query_term_orig": test_df_list[n]['query_term_orig'].values[0],
         }
    print(f"Doc {n}\nQuery: {test_df_list[n]['query_term_orig'].values[0]:<40}\nPredicted: {str(pred):<40}\nProb: {pred_probas[n]}\nTrue: {true}\n\n")

test_df.iloc[235]


# get the top score for each task
top_scores = test_df[test_df['is_max_for_task']].copy()
top_scores.to_csv("top_scores.csv", index=False)

top_scores['pred_title'] = top_scores['title']
top_scores.loc[top_scores['is_predicted_match']==False, 'pred_title'] = ''
# fix NAN in correct_title
top_scores['correct_title'] = top_scores['correct_title'].replace({None: ""})
top_scores['correct_title'] = top_scores['correct_title'].replace({np.nan: ""})
top_scores['correct'] = top_scores['correct_title'] == top_scores['pred_title']
top_scores['correct'].mean()

def split_dataframe_by_task(df):
    split_dfs = []
    for task in df['task'].unique():
        task_df = df[df['task'] == task].copy()
        split_dfs.append(task_df)
    return split_dfs

def run_ablation(drop_vars: list, 
                 train_n: int=-1,
                 train_df_list=train_df_list, 
                 test_df_list=test_df_list,
                 seed=1):
    omit_columns = ['y', 'correct_title', 'title', 'combined_score', # 'combined_score_norm',
                    'task', 'passage', 'query_term', 'query_term_orig']#, 'title_sim', 'title_sim_norm']
    omit_columns += drop_vars
    print(f"Running ablation with dropped variables: {drop_vars}")
    if train_n > 0:
        # randomly sample train_n tasks
        random.seed(seed)
        random.shuffle(train_df_list)
        train_df_list = train_df_list[:train_n]
    print(f"Training on {len(train_df_list)} tasks, testing on {len(train_df_list)} tasks")
    train_df = pd.concat(train_df_list)
    X_train = train_df.drop(columns=omit_columns)
    y_train = train_df['y']

    xgb_model = XGBClassifier(n_estimators=1000,
                        random_state=42,
                        max_depth=5,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        #subsample=0.5,
                        #learning_rate=0.1,
                        class_weight='balanced',
                        n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    pred_titles = [predict_xgboost(i, xgb_model, just_title=True) for i in tqdm(test_df_list)]
    true_titles = get_true_titles(test_df_list)
    acc = np.mean(np.array(pred_titles) == np.array(true_titles))
    d = {
        'drop_vars': drop_vars,
        'acc': acc,
        'train_n': train_n,
        'seed': seed,
    }
    return d

# Run ablation study

# ['index', 'exact_title_match', 'redirect_match', 'alt_name_match',
# 'alt_names_count', 'redirect_names_count', 'num_es_results', 'intro_length',
# 'country_match', 'title_sim', 'context_sim_intro', 'context_sim_short',
# 'actor_desc_sim_intro', 'actor_desc_sim_short', 'levenshtein', 'lcs',
# 'best_subseq', 'best_lev', 'title_sim_norm', 'context_sim_intro_norm',
# 'context_sim_short_norm', 'actor_desc_sim_intro_norm',
# 'actor_desc_sim_short_norm', 'lcs_norm', 'levenshtein_norm',
# 'country_match_norm', 'exact_title_match_norm', 'alt_name_match_norm',
# 'redirect_match_norm']

# We want to focus on expensive features (mostly actor sim)
drop_vars_list = [
    [],
    ['index', 'num_es_results'],  # ES results
    ['intro_length', 'alt_names_count', 'redirect_names_count'], # basic "importance"
    # remove all the norms
    ['context_sim_intro_norm', 'context_sim_short_norm', 'actor_desc_sim_intro_norm', 'actor_desc_sim_short_norm', 'levenshtein_norm', 'lcs_norm', 'country_match_norm', 'exact_title_match_norm', 'alt_name_match_norm', 'redirect_match_norm'],
    ['title_sim', 'title_sim_norm'],
    ['context_sim_intro', 'context_sim_intro_norm'],
    ['context_sim_short', 'context_sim_short_norm'],
    ['actor_desc_sim_intro', 'actor_desc_sim_intro_norm'],
    ['actor_desc_sim_short', 'actor_desc_sim_short_norm'],
    ['levenshtein', 'levenshtein_norm'],
    ['lcs', 'lcs_norm'],
    ['best_subseq', 'best_lev'],
    ['best_subseq', 'best_lev', 'levenshtein_norm', 'lcs_norm', 'lcs', 'levenshtein'],
    ['country_match', 'country_match_norm'],
    ['exact_title_match', 'exact_title_match_norm'],
    ['alt_name_match', 'alt_name_match_norm'],
    ['redirect_match', 'redirect_match_norm'],
    # exclude all the context-dependent features
    ['context_sim_intro', 'context_sim_short', 'actor_desc_sim_intro', 'actor_desc_sim_short',
     'context_sim_intro_norm', 'context_sim_short_norm', 'actor_desc_sim_intro_norm', 'actor_desc_sim_short_norm'],
    # exclude just the context sim features
    ['context_sim_intro', 'context_sim_short', 'context_sim_intro_norm', 'context_sim_short_norm'],
    # exclude just the actor desc sim features
    ['actor_desc_sim_intro', 'actor_desc_sim_short', 'actor_desc_sim_intro_norm', 'actor_desc_sim_short_norm'],
]

ablation_results = []
for drop_vars in drop_vars_list:
    result = run_ablation(drop_vars)
    ablation_results.append(result)

ablation_df = pd.DataFrame(ablation_results)
# sort by f1
ablation_df.sort_values('acc', ascending=False, inplace=True)
ablation_df['norm_acc'] = ablation_df['acc'] / np.mean(possible)
ablation_df

ablation_n_results = []
for seed in [1, 2, 3, 4, 5]:
    for train_n in [300, 500, 750, 800, 900, 1000]:
        for drop_vars in drop_vars_list:
            result = run_ablation(drop_vars = drop_vars,
                                  train_n=train_n,
                                  seed=seed)
            ablation_n_results.append(result)

ablation_n_df = pd.DataFrame(ablation_n_results)
ablation_n_df.sort_values('acc', ascending=False, inplace=True)
ablation_n_df.to_csv("ablation_n_results.csv", index=False)

import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb


def plot_xgb_feature_importance(xgb_model, feature_names, top_n=-1, filename='xgb_importance.png'):
    """
    Plot different types of XGBoost feature importance
    """
    plt.figure(figsize=(12, 15))
    
    # Plot 1: Weight importance (number of times a feature appears in the trees)
    plt.subplot(3, 1, 1)
    ax1 = plt.gca()
    xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=top_n, title='Feature Importance (Weight)', ax=ax1)
    
#    # Plot 2: Gain importance (improvement in accuracy brought by a feature)
#    plt.subplot(3, 1, 2)
#    ax2=plt.gca()
#    xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=top_n, title='Feature Importance (Gain)', ax=ax2)
#    ax2.set_xscale('log')
    # Plot 2: Gain importance (custom implementation with log scale)
    plt.subplot(3, 1, 2)
    # Get feature importance data
    importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
    tuples = [(k, importance_gain[k]) for k in importance_gain]
    tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
    if len(tuples) > top_n:
        tuples = tuples[:top_n]
        
    # Unpack the feature names and values
    features, values = zip(*tuples)
    # Create custom horizontal bar plot with log scale
    ax2 = plt.gca()
    ax2.barh(range(len(features)), values)
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features)
    ax2.set_xscale('log')  # This should now work correctly
    ax2.set_title('Feature Importance (Gain) - Log Scale')
    ax2.set_xlabel('Log Importance Score')
    ax2.grid(True)
    
    # Plot 3: Cover importance (relative quantity of observations affected by splits)
    plt.subplot(3, 1, 3)
    ax3 = plt.gca()
    xgb.plot_importance(xgb_model, importance_type='cover', 
                        max_num_features=top_n, 
                        title='Feature Importance (Cover)', 
                        ax=ax3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"XGBoost feature importance plots saved to {filename}")
    
    # Also return the importance values as a DataFrame
    importance_types = ['weight', 'gain', 'cover']
    importance_data = {}
    
    for imp_type in importance_types:
        # Get importance scores
        importance = xgb_model.get_booster().get_score(importance_type=imp_type)
        
        # Some features might not appear in the trees, so ensure all features are included
        all_importances = [importance.get(f, 0) for f in feature_names]
        
        # Normalize for easier comparison
        normalized_imp = np.array(all_importances) / np.sum(all_importances)
        importance_data[imp_type] = normalized_imp
    
    # Create a sorted list for easier viewing
    sorted_indices = np.argsort(-importance_data['gain'])  # Sort by gain importance
    sorted_features = [feature_names[i] for i in sorted_indices]
    
    result = {
        'Feature': sorted_features,
        'Weight': [importance_data['weight'][i] for i in sorted_indices],
        'Gain': [importance_data['gain'][i] for i in sorted_indices],
        'Cover': [importance_data['cover'][i] for i in sorted_indices]
    }
    
    # Also save as CSV
    import pandas as pd
    importance_df = pd.DataFrame(result)
    importance_df.to_csv(filename.replace('.png', '.csv'), index=False)
    
    return importance_df

# Example usage:
feature_names = X_train.columns.tolist()
feature_importance_df = plot_xgb_feature_importance(xgb_model, feature_names, top_n=-1)
print(feature_importance_df.head(20))  # Show top 10 features

# Visualize a specific tree from your XGBoost model
def plot_xgb_tree(xgb_model, tree_index=0):
    """
    Visualize a specific tree from an XGBoost model
    """
    filename = f"xgb_tree_{tree_index}.png"
    # Create a temporary graph representation of the tree
    graph_data = xgb.to_graphviz(xgb_model, num_trees=tree_index)
    
    # Save the visualization as a PNG
    graph_data.render(filename=filename.replace('.png', ''), format='png', cleanup=True)
    
    print(f"XGBoost tree visualization saved to {filename}")
    
    return graph_data

# Example usage:
plot_xgb_tree(xgb_model, tree_index=0)
plot_xgb_tree(xgb_model, tree_index=1)
plot_xgb_tree(xgb_model, tree_index=2)
plot_xgb_tree(xgb_model, tree_index=400)