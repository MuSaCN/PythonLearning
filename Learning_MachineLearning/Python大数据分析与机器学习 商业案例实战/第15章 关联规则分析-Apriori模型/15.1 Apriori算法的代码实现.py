# # 15.1.3 Apriori算法的代码实现
# # 1.apyori库代码实现关联规则
transactions = [['A', 'B', 'C'], ['A', 'B'], ['B', 'C'], ['A', 'B', 'C', 'D'], ['B', 'C', 'D']]

from apyori import apriori
rules = apriori(transactions, min_support=0.4, min_confidence=0.8)
results = list(rules)
print(results)
print(type(results[0].ordered_statistics))

for i in results:
    for j in i.ordered_statistics:
        X = j.items_base
        Y = j.items_add
        x = ', '.join([item for item in X])
        y = ', '.join([item for item in Y])
        if x != '':
            print(x + ' → ' + y)

# # 2.mlxtend库代码实现关联规则
transactions = [['A', 'B', 'C'], ['A', 'B'], ['B', 'C'], ['A', 'B', 'C', 'D'], ['B', 'C', 'D']]
from mlxtend.preprocessing import TransactionEncoder
TE = TransactionEncoder()
data = TE.fit_transform(transactions)
print(data)

import pandas as pd
df = pd.DataFrame(data, columns=TE.columns_)
print(df)

from mlxtend.frequent_patterns import apriori
items = apriori(df, min_support=0.4, use_colnames=True)
print(items)

items['itemsets'].apply(lambda x: len(x))
print(items[items['itemsets'].apply(lambda x: len(x)) >= 2])

from mlxtend.frequent_patterns import association_rules
rules = association_rules(items, min_threshold=0.8)
print(rules)

for i, j in rules.iterrows():
    X = j['antecedents']
    Y = j['consequents']
    x = ', '.join([item for item in X])
    y = ', '.join([item for item in Y])
    print(x + ' → ' + y)

