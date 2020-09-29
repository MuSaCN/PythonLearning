# # 15.2 综合案例：中医病症辩证关联规则分析
# # 数据读取与预处理
import pandas as pd
df = pd.read_excel('中医辨证.xlsx')
df.head()

df['病人症状'].tolist()

symptoms = []
for i in df['病人症状'].tolist():
    symptoms.append(i.split(','))
print(symptoms)

from apyori import apriori
rules = apriori(symptoms, min_support=0.1, min_confidence=0.7)
results = list(rules)

for i in results:
    for j in i.ordered_statistics:
        X = j.items_base
        Y = j.items_add
        x = ', '.join([item for item in X])
        y = ', '.join([item for item in Y])
        if x != '':
            print(x + ' → ' + y)

from mlxtend.preprocessing import TransactionEncoder
TE = TransactionEncoder()
data = TE.fit_transform(symptoms)
print(data)

import pandas as pd
df = pd.DataFrame(data, columns=TE.columns_)
df.head()

from mlxtend.frequent_patterns import apriori
items = apriori(df, min_support=0.1, use_colnames=True)
print(items)

print(items[items['itemsets'].apply(lambda x: len(x)) >= 2])


from mlxtend.frequent_patterns import association_rules
rules = association_rules(items, min_threshold=0.7)
print(rules)

for i, j in rules.iterrows():
    X = j['antecedents']
    Y = j['consequents']
    x = ', '.join([item for item in X])
    y = ', '.join([item for item in Y])
    print(x + ' → ' + y)

