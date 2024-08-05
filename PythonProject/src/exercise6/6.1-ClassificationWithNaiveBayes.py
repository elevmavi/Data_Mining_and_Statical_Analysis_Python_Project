import numpy as np
from scipy.stats import norm


def calculate_mean_std(attribute, gender):
    """
    Calculate the mean and standard deviation for a given attribute and gender.

    Parameters:
    attribute (numpy.ndarray): Array of attribute values (height, weight, or shoe size).
    gender (str): Gender to filter the attribute values ('M' for male, 'F' for female).

    Returns:
    tuple: Mean and standard deviation of the filtered attribute values.
    """
    values = attribute[genders == gender]
    mean = np.mean(values)
    std = np.std(values)
    return mean, std


def calculate_probability(mean, std, x):
    """
    Calculate the Gaussian probability density function for a given value.

    Parameters:
    mean (float): Mean of the attribute.
    std (float): Standard deviation of the attribute.
    x (float): The value for which to calculate the probability.

    Returns:
    float: Probability of x given the mean and standard deviation.
    """
    return norm.pdf(x, mean, std)


# Training dataset
data = {
    'gender': ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'],
    'height': [182, 180, 170, 180, 152, 167, 165, 175],
    'weight': [81, 86, 77, 74, 45, 68, 58, 68],
    'shoe_size': [45, 42, 45, 40, 30, 35, 32, 37]
}

# Convert data to numpy arrays
genders = np.array(data['gender'])
heights = np.array(data['height'])
weights = np.array(data['weight'])
shoe_sizes = np.array(data['shoe_size'])

# Calculate mean and standard deviation for each attribute and gender
male_height_mean, male_height_std = calculate_mean_std(heights, 'M')
male_weight_mean, male_weight_std = calculate_mean_std(weights, 'M')
male_shoe_size_mean, male_shoe_size_std = calculate_mean_std(shoe_sizes, 'M')

female_height_mean, female_height_std = calculate_mean_std(heights, 'F')
female_weight_mean, female_weight_std = calculate_mean_std(weights, 'F')
female_shoe_size_mean, female_shoe_size_std = calculate_mean_std(shoe_sizes, 'F')

# Print results
print(f"Males - Height: mean={male_height_mean:.2f}, std={male_height_std:.2f}")
print(f"Males - Weight: mean={male_weight_mean:.2f}, std={male_weight_std:.2f}")
print(f"Males - Shoe Size: mean={male_shoe_size_mean:.2f}, std={male_shoe_size_std:.2f}")

print(f"Females - Height: mean={female_height_mean:.2f}, std={female_height_std:.2f}")
print(f"Females - Weight: mean={female_weight_mean:.2f}, std={female_weight_std:.2f}")
print(f"Females - Shoe Size: mean={female_shoe_size_mean:.2f}, std={female_shoe_size_std:.2f}")

# Test data
test_data = {
    'height': 182,
    'weight': 58,
    'shoe_size': 35
}

# Calculate probabilities for males
p_male_height = calculate_probability(male_height_mean, male_height_std, test_data['height'])
p_male_weight = calculate_probability(male_weight_mean, male_weight_std, test_data['weight'])
p_male_shoe_size = calculate_probability(male_shoe_size_mean, male_shoe_size_std, test_data['shoe_size'])

# Total probability for males
p_male = p_male_height * p_male_weight * p_male_shoe_size * 0.5

# Calculate probabilities for females
p_female_height = calculate_probability(female_height_mean, female_height_std, test_data['height'])
p_female_weight = calculate_probability(female_weight_mean, female_weight_std, test_data['weight'])
p_female_shoe_size = calculate_probability(female_shoe_size_mean, female_shoe_size_std, test_data['shoe_size'])

# Total probability for females
p_female = p_female_height * p_female_weight * p_female_shoe_size * 0.5

# Print results
print(f"Probability of being male: {p_male:.20f}")
print(f"Probability of being female: {p_female:.20f}")

# Classification
if p_male > p_female:
    print("The person is more likely to be male.")
else:
    print("The person is more likely to be female.")
