# --------------------------------------------
# Apriori Algorithm Implementation using mlxtend
# --------------------------------------------

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Define sample transaction data
transactions = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'eggs'],
    ['milk', 'bread', 'eggs']
]

# Step 2: Encode transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
encoded_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(encoded_data, columns=te.columns_)

print("Encoded Transaction Data:\n")
print(df)
print("-" * 50)

# Step 3: Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

print("Frequent Itemsets (min_support = 0.6):\n")
print(frequent_itemsets)
print("-" * 50)

# Step 4: Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Sort rules by lift (to find strongest associations)
rules = rules.sort_values(by='lift', ascending=False)

print("Association Rules (min_confidence = 0.7):\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print("-" * 50)

# Step 5 (Optional): Nicely format antecedents & consequents for readability
rules_formatted = rules.copy()
rules_formatted['antecedents'] = rules_formatted['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_formatted['consequents'] = rules_formatted['consequents'].apply(lambda x: ', '.join(list(x)))

print("Formatted Association Rules:\n")
print(rules_formatted[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
