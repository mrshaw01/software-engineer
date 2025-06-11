import random

from product import Product
from product import Products

fantasy_names = [
    "Elixir of Eternal Youth",
    "Dragonfire Sword",
    "Phoenix Feather Wand",
    "Mermaid's Tears Necklace",
    "Elven Cloak of Invisibility",
    "Potion of Flying",
    "Amulet of Wisdom",
    "Crystal Ball of Fortune",
    "Enchanted Mirror",
    "Unicorn Horn Ring",
]

currencies = list(Product.conversion_rates.keys())
products = Products()

# Add randomized products
for name in fantasy_names:
    price = random.uniform(70, 1000)
    products.add_product(Product(name, price, price * 0.03, currency=random.choice(currencies)))

# View products in USD
print("\nViewing products in USD:")
products.view_products(currency="USD")

# View products in CHF without adapting data
print("\nViewing products in CHF (no adaptation):")
products.view_products(currency="CHF")

# Show saved internal state
print("\nSaved internal data view:")
products.view_products_as_saved_data()

# View products in CHF with data adaptation
print("\nViewing products in CHF (with adapt_data=True):")
products.view_products(currency="CHF", adapt_data=True)

# View final saved data
print("\nFinal saved internal state (after adaptation):")
products.view_products_as_saved_data()
