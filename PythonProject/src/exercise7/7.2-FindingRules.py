import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

from src.shared.data_processing import etl_mat_file_fill_na

# Load the Karate dataset
karate = etl_mat_file_fill_na('resources/data/Karate.mat', 'Karate')

# Apply the apriori algorithm to find frequent itemsets
minSup = 0.1
minConf = 0.7

# Convert the adjacency matrix to a format suitable for apriori
Karate_boolean = karate.applymap(lambda x: True if x == 1 else False)

# Find frequent itemsets
freq_itemsets = apriori(Karate_boolean, min_support=minSup, use_colnames=True)

# Generate the rules
rules = association_rules(freq_itemsets, metric="confidence", min_threshold=minConf)

# Limit the number of rules
rules = rules.nlargest(100, 'confidence')

# Add labels
rules['antecedents'] = rules['antecedents'].apply(lambda x: frozenset([f'Student{i + 1}' for i in x]))
rules['consequents'] = rules['consequents'].apply(lambda x: frozenset([f'Student{i + 1}' for i in x]))

# Create a 'Rule' column for better readability in the output file
rules['Rule'] = rules.apply(lambda row: f"{', '.join(row['antecedents'])} -> {', '.join(row['consequents'])}", axis=1)

# Save the rules to a file
rules.to_csv('Result.txt', columns=["Rule", "support", "confidence"], index=False, header=["Rule", "Support", "Confidence"])

# Display the rules
print(rules[["Rule", "support", "confidence"]])

# Visualize the adjacency matrix
plt.spy(karate, markersize=5)
plt.title('Karate Club Adjacency Matrix')
plt.show()
