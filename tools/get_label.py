import re
import csv
import os
import time
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

def get_label(review, f, img_list):
    reviewerID = review['reviewerID']
    if 'reviewText' in review and 'summary' in review:
        vote = int(review['vote'].replace(',',''))
        text = review['reviewText']
        summary = review['summary']
        # asin = review['asin']
        # reviewtime = int(review['unixReviewTime'])
        # reviewtime = time.localtime(reviewtime)
        # reviewtime = time.strftime("%Y-%m-%d %H", reviewtime)
        blob_text = TextBlob(text)
        blob_summary = TextBlob(summary)
        text_list = []
        summary_list = []
        for sentence in blob_text.sentences:
            text_list.append(sentence.sentiment.polarity)
        for sentence in blob_summary.sentences:
            summary_list.append(sentence.sentiment.polarity)
        s = 0.7 * np.array(text_list).mean() + 0.3 * np.array(summary_list).mean()
        # s = np.array(summary_list).mean()
        text_summary = re.sub('[^\da-zA-Z\s\.,!?]', '', summary + text).replace('\n','')
        if len(text.split())>=5 and len(summary.split())>=1:
        # if (reviewerID in img_list) and len(text.split())>=10 and len(summary.split())>=1:
            if s<=-0.01 and review['overall']<=3.0:
                data = [reviewerID, str(text_summary), 0.0]
                # data = [reviewerID, asin, reviewtime, 0.0]
            elif s>=0.01 and review['overall']>=4.0:
                data = [reviewerID, str(text_summary), 2.0]
                # data = [reviewerID, asin, reviewtime, 1.0]
            elif s<0.01 and s>-0.01 and review['overall']==4.0:
                data = [reviewerID, str(text_summary), 1.0]
        else:
            return
        lock.acquire()
        try:
            writer = csv.writer(f)
            writer.writerow(data)
        finally:
            lock.release()

with open('label_0.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(["ID","text","label"])
    # writer.writerow(["ID","asin","time","label"])

# for review in tqdm(read_json_file('./Books_5.json')):
#     get_label(review)

max_workers = 120
executor = ThreadPoolExecutor(max_workers=max_workers)
tasks = []

img_list = []
for i in os.listdir("../data/Books_5_images"):
    img_list.append(i[:-4])

with open('label_0.csv', 'a') as f:
    for review in tqdm(read_json_file('../data/S_0.json')):
    # for review in tqdm(read_json_file('../data/Books_5_img.json')):
        tasks.append(executor.submit(get_label, review, f, img_list))
    for futures in tqdm(as_completed(tasks), total=len(tasks)):
        pass
