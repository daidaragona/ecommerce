import pandas as pd

# Read the products data from the given URL
df_products = pd.read_json(
    "https://raw.githubusercontent.com/anyoneai/e-commerce-open-data-set/master/products.json"
)
# Select only the required columns
df_products = df_products[["name", "category", "description"]]

# Create a histogram of categories
category_counts = (
    df_products["category"].explode().apply(lambda x: x["name"]).value_counts()
)

# Filter categories with fewer than 100 products and assign them the category "Other"
threshold = 100
filtered_category_counts = category_counts[category_counts >= threshold]
filtered_categories = filtered_category_counts.index.tolist()
df_products["category"] = df_products["category"].apply(
    lambda x: [
        cat["name"] if cat["name"] in filtered_categories else "Other" for cat in x
    ]
)


# Create the new columns for each level
for i in range(1, 7 + 1):
    level_name = "level_" + str(i)
    df_products[level_name] = df_products["category"].apply(
        lambda x: x[i - 1] if len(x) >= i else "NA"
    )
# Drop the original "category" column
df_products.drop("category", axis=1, inplace=True)


def get_categories():
    """Returns a dictionary mapping each category at each level to a unique index.

    Returns:
        dict: A dictionary with level names as keys and nested dictionaries as values.
              The nested dictionaries map each category to a unique index.
    """
    categories = dict()
    for i in range(1, 8):
        cat = df_products["level_" + str(i)].unique().tolist()
        categories["level_" + str(i)] = dict(list(zip(cat, list(range(len(cat))))))
    return categories


def get_df_products(df_products):
    """Adds a new 'text' column to the DataFrame by concatenating 'name' and 'description' columns.

    Args:
        df_products (pd.DataFrame): The DataFrame containing the product data.
    """
    df_products["text"] = df_products["name"] + " " + df_products["description"]
