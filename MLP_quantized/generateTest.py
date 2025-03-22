import numpy as np
import pandas as pd

def convert_csv_to_hpp(csv_file, hpp_file, array_name="test_data", label_name="labels"):
    # Label mapping dictionary
    label_mapping = {
        "badminton": 0, "boxing": 1, "end": 2, "fencing": 3, "golf": 4,
        "reload": 5, "shield": 6, "snowbomb": 7, "walking": 8, "wearing": 9
    }
    
    # Read CSV file, ignoring the first row
    df = pd.read_csv(csv_file, header=None, skiprows=1)
    
    # Extract test data and labels
    test_data = df.iloc[:, :-1].values  # All columns except last
    labels = df.iloc[:, -1].map(label_mapping).values  # Convert labels to integers
    
    # Open HPP file for writing
    with open(hpp_file, "w") as f:
        f.write("#ifndef TEST_DATA_HPP\n#define TEST_DATA_HPP\n\n")
        
        # Get number of samples and features
        num_samples, num_features = test_data.shape
        
        # Write test_data array
        f.write(f"const float test_data[{num_samples}][{num_features}] = {{\n")
        for row in test_data:
            row_str = "{ " + ", ".join(map(str, row)) + " }"
            f.write(f"    {row_str},\n")
        f.write("};\n\n")
        
        # Write labels array
        f.write(f"const int test_labels[{num_samples}] = {{\n")
        labels_str = ", ".join(map(str, labels))
        f.write(f"    {labels_str}\n")
        f.write("};\n\n")
        
        f.write("#endif // TEST_DATA_HPP\n")

# Example usage
convert_csv_to_hpp("data/processed_data_scaled.csv", "test_data.hpp")