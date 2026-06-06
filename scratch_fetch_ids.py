import openml

# Existing IDs from build_memory.py
existing_ids = [
    61, 31, 153, 44, 1504, 1494, 1462, 37, 1464, 40945, 1049, 40983, 54, 181, 1510, 40668, 23, 1489, 1120, 38,
    46, 182, 300, 4534, 1067, 41021, 507, 531, 422, 41540, 560, 574, 589, 1199, 42092, 42165, 42705, 42726, 42727,
    42728, 1590, 151, 11, 14, 16, 18, 22, 50, 188, 307, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 19, 20, 21,
    24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 45, 47, 48, 49, 51, 52, 53, 55, 56, 57,
    58, 59, 60, 62, 63, 64, 65, 189, 197, 201, 214, 225, 227, 228, 229, 230, 549, 564, 1027, 1028, 1029, 1030,
    23381, 40691, 1468, 1475, 1478, 1480, 1485, 1486, 1487, 1488, 4134, 6332, 23517, 40670, 40701, 179, 184, 554,
    772, 917, 1019, 1020, 1021, 1040, 1053, 1063, 1068, 4538, 6956, 40536, 41702, 42225, 43071, 43439, 43551, 41278,
    42563, 41980, 43928, 44027, 1169, 1170, 1442, 1443, 1444, 1446, 1447, 1448
]

# Fetch all active datasets
datasets = openml.datasets.list_datasets(output_format='dataframe')

# Filter
filtered = datasets[
    (datasets['NumberOfInstances'] > 50) & 
    (datasets['NumberOfInstances'] < 100000) & 
    (datasets['NumberOfFeatures'] < 200) &
    (datasets['NumberOfClasses'].fillna(0) <= 20) & # Not too many classes, or regression
    (~datasets['did'].isin(existing_ids))
]

# Get 150 IDs
new_ids = filtered['did'].tolist()[:150]

print("Fetched", len(new_ids), "IDs")
print(new_ids)
