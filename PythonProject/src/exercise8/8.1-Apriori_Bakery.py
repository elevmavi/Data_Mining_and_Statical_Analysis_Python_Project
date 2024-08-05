from mlxtend.frequent_patterns import apriori, association_rules

from src.shared.data_processing import etl_mat_file_fill_na, etl_text_file


def generate_association_rules(data, min_sup, min_conf, output_filename):
    """
    Generate association rules from transaction data using the Apriori algorithm and save the results to a file.

    Parameters:
        data (DataFrame): Input transaction data where each row represents a transaction, and each column represents an item.
        min_sup (float): Minimum support threshold for frequent itemsets.
        min_conf (float): Minimum confidence threshold for association rules.
        output_filename (str): Name of the output file to save the association rules.

    Returns:
        None
    """
    # Find frequent itemsets
    frequent_itemsets = apriori(data, min_support=min_sup, use_colnames=True)

    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

    # Sort and limit the number of rules
    sorted_rules = rules.sort_values(by=['confidence', 'support'], ascending=[False, False])

    # Create the output DataFrame
    output = sorted_rules[['antecedents', 'consequents', 'support', 'confidence']].copy()
    output['antecedents'] = output['antecedents'].apply(lambda x: ', '.join(map(str, list(x))))
    output['consequents'] = output['consequents'].apply(lambda x: ', '.join(map(str, list(x))))
    output['Rule'] = output['antecedents'] + ' -> ' + output['consequents']

    # Convert support and confidence to percentages
    output['support'] = ((output['support'] * 100).round(2)).astype(str) + '%'
    output['confidence'] = ((output['confidence'] * 100).round(2)).astype(str) + '%'

    output = output[['Rule', 'support', 'confidence']]

    # Save to a text file
    output.to_csv(output_filename, sep='\t', index=False)

    # Print the number of rules found
    num_rules = len(rules)
    print(f"Number of rules found: {num_rules}")
    print(f"See the file named {output_filename} for the association rules")


bakery1000 = etl_mat_file_fill_na('resources/data/Bakery1000.mat', 'xV')
bakery1000 = bakery1000.astype(bool)
bakery1000 = bakery1000.iloc[:, 1:]

bakery7500 = etl_text_file('resources/data/Bakery75000.csv', ',')
bakery7500 = bakery7500.astype(bool)
bakery7500 = bakery7500.iloc[:, 1:]

# Generate labels
labels = [f'product{i}' for i in range(1, bakery1000.shape[1] + 1)]
bakery1000.columns = labels

# Define minimum support and confidence
min_sup = 0.05
min_conf = 0.1

# Generate association rules for min support 0.05 and min confidence 0.1
generate_association_rules(bakery1000, min_sup, min_conf, 'BakeryRules-1000-1.txt')
generate_association_rules(bakery7500, min_sup, min_conf, 'BakeryRules-7500-1.txt')

# Change min_sup and min_conf
min_sup = 0.05
min_conf = 0.05

# Generate association rules for min support 0.05 and min confidence 0.05
generate_association_rules(bakery1000, min_sup, min_conf, 'BakeryRules-1000-2.txt')
generate_association_rules(bakery7500, min_sup, min_conf, 'BakeryRules-7500-2.txt')

# Change min_sup and min_conf
min_sup = 0.005
min_conf = 0.005

# Generate association rules for min support 0.0005 and min confidence 0.0005
generate_association_rules(bakery1000, min_sup, min_conf, 'BakeryRules-1000-3.txt')
generate_association_rules(bakery7500, min_sup, min_conf, 'BakeryRules-7500-3.txt')
