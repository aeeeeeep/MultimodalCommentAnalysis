import pandas as pd
import os
label = pd.read_csv("../data/label.csv")
img_list = []
for i in os.listdir("../data/Books_5_images"):
    img_list.append(i[:-4])
temp = list(set(label["ID"]) & set(img_list))
df = label.loc[(label['ID'].isin(temp))][:5000]
df.to_excel("label.xlsx")
