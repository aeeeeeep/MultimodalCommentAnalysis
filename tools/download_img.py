import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import ProxyHandler, build_opener

from tqdm import tqdm

proxy_handler = ProxyHandler({'socks5': '127.0.0.1:7890'})
opener = build_opener(proxy_handler)


def download_image(image_url, reviewer_id):
    try:
        with opener.open(image_url) as url:
            image_data = url.read()
        if not os.path.exists('images'):
            os.makedirs('images')
        with open(f'images/{reviewer_id}.jpg', 'wb') as img_file:
            img_file.write(image_data)
    except Exception as e:
        print(f'Error downloading image {image_url}: {e}')
        with open('404.csv', 'a') as f:
            f.write(f'{reviewer_id},"{image_url}"\n')


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)


max_workers = 64
executor = ThreadPoolExecutor(max_workers=max_workers)

tasks = []

id_list = []
images_list = os.listdir('./images')
for image_list in images_list:
    id_list.append(image_list.split('.')[0])

for review in tqdm(read_json_file('../Books_5.json')):
    if ('image' in review) and (review['reviewerID'] not in id_list):
        images = review['image']
        for i, image_url in enumerate(images):
            image_url = image_url.replace("_SY88.", "")
            tasks.append(executor.submit(download_image, image_url, review['reviewerID']))

for future in tqdm(as_completed(tasks), total=len(tasks)):
    pass
