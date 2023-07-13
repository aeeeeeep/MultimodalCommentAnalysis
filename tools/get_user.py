import gc
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

lock = threading.Lock()


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line), line


def get_user(review, line):
    reviewerID = review['reviewerID']
    if 'reviewText' in review and 'summary' in review:
        filename = f"./users/{reviewerID}.json"
        lock.acquire()
        try:
            with open(filename, 'a') as f:
                f.write(line)
        finally:
            lock.release()


gc.collect(generation=0)
max_workers = 16
executor = ThreadPoolExecutor(max_workers=max_workers)
tasks = []

for i in tqdm(range(5)[4:]):
    for review, line in tqdm(read_json_file(f'../data/{i + 1}.json')):
        tasks.append(executor.submit(get_user, review, line))
    for future in tqdm(as_completed(tasks), total=len(tasks)):
        pass
