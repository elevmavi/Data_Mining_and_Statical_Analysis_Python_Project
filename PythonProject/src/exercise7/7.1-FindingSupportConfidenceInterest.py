# Define the transaction data
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = {
    'Transaction': [1, 2, 3, 4, 5],
    'Milk': [True, False, False, True, False],
    'Bread': [True, False, False, True, True],
    'Butter': [False, True, False, True, False],
    'Beer': [False, False, True, False, False],
    'Diapers': [False, False, True, False, False]
}


# Create a DataFrame from the transaction data
df = pd.DataFrame(data)

# Set 'Transaction' column as the index
df.set_index('Transaction', inplace=True)

print(df)

# Calculate support for item pairs (e.g., Milk -> Bread)
support = pd.DataFrame(index=df.columns, columns=df.columns)

for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2:
            sup = sum(df[col1] & df[col2]) / len(df)
            support.loc[col1, col2] = sup
print("Support for item pairs:")
print(support)

# Calculate confidence for item pairs (e.g., Milk -> Bread)
confidence = pd.DataFrame(index=df.columns, columns=df.columns)

for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2:
            conf = sum(df[col1] & df[col2]) / sum(df[col1])
            confidence.loc[col1, col2] = conf

print("Confidence for item pairs:")
print(confidence)

# Calculate lift for item pairs
lift = pd.DataFrame(index=df.columns, columns=df.columns)

for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2:
            supp_both = sum(df[col1] & df[col2]) / len(df)
            supp_col1 = sum(df[col1]) / len(df)
            supp_col2 = sum(df[col2]) / len(df)
            lift.loc[col1, col2] = supp_both / (supp_col1 * supp_col2)

print("Lift for item pairs:")
print(lift)

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Generate association rules based on frequent itemsets
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Filter rules based on minimum confidence
filtered_rules = rules[rules['confidence'] >= 0.5]

# Display the filtered rules without truncation
with pd.option_context('display.max_columns', None):
    print("Association Rules with minsup=0.3 and minconf=0.5:")
    print(filtered_rules)


# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Generate association rules based on frequent itemsets
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)

# Filter rules based on minimum confidence
filtered_rules = rules[rules['confidence'] >= 0.6]

# Display the filtered rules without truncation
with pd.option_context('display.max_columns', None):
    print("Association Rules with minsup=0.2 and minconf=0.6:")
    print(filtered_rules)
