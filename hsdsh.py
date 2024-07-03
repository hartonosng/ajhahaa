import Levenshtein

# Extended list of values
values = [
    'aku', 'nomor rekening', '2000222', 'saldo', '1000000', 'nomor rekening', '23444', 'account number', '20009', 'nama', 'bambang', 'alamat', 'jl. kemang',
    'nama', 'hartono', 'kode pos', '12730', 'phone number', '08123456789', 'amount', 'Rp239', 'bank', 'bca', 'email', 'bambang@gmail.com', 'jenis kelamin', 'pria'
]

# Keywords dictionary
keywords_dict = {
    'account number': ['no rekening', 'account no'],
    'name': ['name', 'nama'],
    'amount': ['amount'],
    'phone number': ['phone number', 'no hp'],
    'email': ['email', 'e-mail']
}

def find_closest_matches(values, category, threshold_probability=0.8):
    # Check if the category exists in the keywords dictionary
    if category not in keywords_dict:
        return []

    matched_values = []
    matched_indices = set()

    # Normalize the values to lower case
    normalized_values = [value.lower() for value in values]

    # Get the list of keywords for the specified category
    keywords = keywords_dict[category]

    for keyword in keywords:
        # Normalize the keyword
        normalized_keyword = keyword.lower()

        # Compute the similarity ratio for each value
        similarities = [(Levenshtein.ratio(value.lower(), normalized_keyword), i) for i, value in enumerate(values)]

        # Filter values based on the threshold probability
        filtered_indices = [i for similarity, i in similarities if similarity >= threshold_probability]

        # Add the values following each matched keyword occurrence if not already added
        for idx in filtered_indices:
            if idx not in matched_indices and idx + 1 < len(values):
                matched_value = values[idx + 1]
                matched_values.append(matched_value)
                matched_indices.add(idx)

    return matched_values

# Find and print the matched values for each category with a specified threshold probability
matched_values_account = find_closest_matches(values, 'account number', threshold_probability=0.8)
matched_values_name = find_closest_matches(values, 'name', threshold_probability=0.8)
matched_values_amount = find_closest_matches(values, 'amount', threshold_probability=0.8)
matched_values_phone = find_closest_matches(values, 'phone number', threshold_probability=0.8)
matched_values_email = find_closest_matches(values, 'email', threshold_probability=0.8)

print("Account Numbers:", matched_values_account)
print("Names:", matched_values_name)
print("Amounts:", matched_values_amount)
print("Phone Numbers:", matched_values_phone)
print("Emails:", matched_values_email)
