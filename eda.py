import csv
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

def hist(csv_file):
    source_len = []
    i=0
    with open(csv_file, 'r') as fp:
        reader = csv.reader(fp)
        samples = [row for row in reader]
        for sample in samples:
            if(i!=0):
                source = [x for x in sample[1].split()]
                source_len.append(len(source))
            i+=1
    print("The maximum input length is: ",np.array(source_len).max())
    print("The minimum input length is: ",np.array(source_len).min())
    print("The average input length is: ",np.array(source_len).mean())
    print("The mode of input length is: ",np.argmax(np.bincount(np.array(source_len))))
    plt.hist(source_len,bins=5,alpha=0.5)
    plt.show()

def count_numbers(csv_file):
    numbers = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for column in row[1:]:
                numbers += column.split()
    count = Counter(numbers)
    top_100 = count.most_common(100)
    print("The first 100 numbers with the most occurrences:")
    for number, frequency in top_100:
        print(f"{number}: {frequency}")
    print(f"There are {len(set(numbers))} kinds of numbers")

# count_numbers("./data/train.csv")
hist("./data/train.csv")
