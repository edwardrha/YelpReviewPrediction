import pandas as pd


df = pd.read_json("dataset/attribute.json")

df1 = pd.read_json("dataset/business.json", lines=True)

df2 = pd.read_json("dataset/checkin.json", lines=True)

df3 = pd.read_json("dataset/tip.json", lines=True)

df.sort_values('date')
df.head()
df1.head()
df2.head()
df2['time'][1]
df1[df1['business_id'] == 'kREVIrSBbtqBhIYkTccQUg']['hours']

counter = 0
df1 = df1[[('Restaurants' in categories) for categories in df1['categories']]]
temp = df1['business_id'].as_matrix()
for i in range(1,7):
    df = pd.read_json("dataset/review_"+str(i)+".json")
    temp = df.join(df1.set_index('business_id'), on='business_id', how='inner', lsuffix='_caller', rsuffix='_other')
    counter += temp.shape[0]

counter
