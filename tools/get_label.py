import re
import csv
import json
import numpy as np
from tqdm import tqdm
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
lock = threading.Lock()

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)

def get_label(review, f):
    reviewerID = review['reviewerID']
    if 'vote' in review and int(review['vote'].replace(',','')) >= 5 and 'reviewText' in review and 'summary' in review:
        vote = int(review['vote'].replace(',',''))
        text = review['reviewText']
        summary = review['summary']
        blob_text = TextBlob(text)
        blob_summary = TextBlob(summary)
        text_list = []
        summary_list = []
        for sentence in blob_text.sentences:
            text_list.append(sentence.sentiment.polarity)
        for sentence in blob_summary.sentences:
            summary_list.append(sentence.sentiment.polarity)
        s = 0.3 * np.array(text_list).mean() + 0.7 * np.array(summary_list).mean()
        # s = np.array(summary_list).mean()
        text_summary = re.sub('[^\da-zA-Z\s\.,!?]', '', summary).replace('\n','')
        if len(text.split())>=3 and len(summary.split())>=1:
            if s<0 and review['overall']<=3.0:
                data = [reviewerID, str(text_summary), vote, 0.0]
            elif s>0 and review['overall']>=4.0:
                data = [reviewerID, str(text_summary), vote, 1.0]
            else:
                return
            lock.acquire()
            try:
                writer = csv.writer(f)
                writer.writerow(data)
            finally:
                lock.release()

with open('label.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(["ID","summary","vote","label"])

# for review in tqdm(read_json_file('./Books_5.json')):
#     get_label(review)

max_workers = 16
executor = ThreadPoolExecutor(max_workers=max_workers)
tasks = []
with open('label.csv', 'a') as f:
    for review in tqdm(read_json_file('../data/Books_5.json')):
        tasks.append(executor.submit(get_label, review, f))
    for futures in tqdm(as_completed(tasks), total=len(tasks)):
        pass
