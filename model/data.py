import pandas as pd

df_products = pd.read_json(
    "https://raw.githubusercontent.com/anyoneai/e-commerce-open-data-set/master/products.json"
)

df_products = df_products[["name", "category", "description"]]

# Crear un histograma de categorías
category_counts = (
    df_products["category"].explode().apply(lambda x: x["name"]).value_counts()
)

# Filtrar categorías con menos de 100 productos y asignarles la categoría "other"
threshold = 100
filtered_category_counts = category_counts[category_counts >= threshold]
filtered_categories = filtered_category_counts.index.tolist()
df_products["category"] = df_products["category"].apply(
    lambda x: [
        cat["name"] if cat["name"] in filtered_categories else "Other" for cat in x
    ]
)

# Define the number of levels/columns you want to create
num_levels = 7

# Create the new columns
for i in range(1, num_levels + 1):
    level_name = "level_" + str(i)
    df_products[level_name] = df_products["category"].apply(
        lambda x: x[i - 1] if len(x) >= i else "NA"
    )
df_products.drop("category", axis=1, inplace=True)


def get_categories():
    categories = dict()
    for i in range(1, 8):
        cat = df_products["level_" + str(i)].unique().tolist()
        categories["level_" + str(i)] = dict(list(zip(cat, list(range(len(cat))))))
    return categories


def get_df_products(df_products):
    df_products["text"] = df_products["name"] + " " + df_products["description"]
