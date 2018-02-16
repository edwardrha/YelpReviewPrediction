import pandas as pd

df = pd.read_json("dataset/review.json", lines = True)

df1 = pd.read_json("dataset/business.json", lines=True)

df2 = pd.read_json("dataset/checkin.json", lines=True)

df3 = pd.read_json("dataset/tip.json", lines=True)

df.head()
df1.head()
df2.head()
df2['time'][1]
df1[df1['business_id'] == 'kREVIrSBbtqBhIYkTccQUg']['hours']
