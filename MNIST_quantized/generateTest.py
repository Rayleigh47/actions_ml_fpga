import os
import pandas as pd
import numpy as np

# Path to your test CSV file
csv_path = os.path.join("data", "test.csv")
df = pd.read_csv(csv_path)

df = df.iloc[:10000]

# Extract features: drop 'pixel0' and 'pixel1' to get 782 features (pixel2 to pixel783)
# (Adjust this if you wish to drop different columns.)
feature_cols = [f"pixel{i}" for i in range(0, 784)]
features = df[feature_cols].to_numpy(dtype=np.float32)

# Extract labels
labels = df["label"].to_numpy(dtype=np.int32)

# Create the header file content
header_lines = []
header_lines.append("#ifndef TEST_DATA_HPP")
header_lines.append("#define TEST_DATA_HPP")
header_lines.append('const int test_length = 10000;')
header_lines.append("")
header_lines.append("const float test_data[test_length][784] = {")
for row in features:
    # Format each float with 8 decimal places; join with commas.
    row_str = ", ".join(map(str, row))
    header_lines.append("    {" + row_str + "},")
header_lines.append("};")
header_lines.append("")
header_lines.append("const int test_labels[test_length] = {")
labels_str = ", ".join(str(x) for x in labels)
header_lines.append("    " + labels_str)
header_lines.append("};")
header_lines.append("")
header_lines.append("#endif // TEST_DATA_HPP")

# Write the header file to disk
with open("test_data.hpp", "w") as f:
    f.write("\n".join(header_lines))

print("test_data.hpp has been created.")
