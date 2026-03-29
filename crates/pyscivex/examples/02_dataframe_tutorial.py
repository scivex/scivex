"""DataFrame tutorial — creation, filtering, groupby, sorting."""
import pyscivex as sv

# Create from dict
df = sv.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [30.0, 25.0, 35.0, 28.0, 32.0],
    "score": [85.0, 92.0, 78.0, 95.0, 88.0],
    "dept": ["eng", "eng", "sales", "sales", "eng"],
})
print("DataFrame:")
print(df)
print("Shape:", df.shape())
print("Columns:", df.column_names())

# Select columns
ages = df.column("age")
print("Ages:", ages)

# Sort by score descending
sorted_df = df.sort_by("score", ascending=False)
print("\nSorted by score:")
print(sorted_df)

# Filter rows
filtered = df.filter("score", ">", 85.0)
print("\nScore > 85:")
print(filtered)

# GroupBy aggregation
grouped = df.groupby("dept")
print("\nMean by dept:")
print(grouped.mean())

# Descriptive stats
print("\nDescribe:")
print(df.describe())

# Add computed column
print("\nHead(3):", df.head(3))
print("Tail(2):", df.tail(2))
