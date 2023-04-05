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
    plt.hist(source_len,bins=43,alpha=0.5)
    plt.show()
def count_numbers(csv_file):
    cou = 0
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            numbers = len(row[1].split())
            if numbers >= 5 and numbers <= 10 and int(row[2]) > 5:
                cou+=1
    print(cou)

count_numbers("./data/label.csv")
hist("./data/label.csv")
