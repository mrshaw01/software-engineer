# 08. Dynamic Data Transformation

Dynamic data transformation refers to the process of modifying data on-the-fly based on specific rules or conditions, rather than applying a fixed static transformation. This allows for greater flexibility in processing, formatting, and analyzing data without permanently altering the underlying values.

## Analogy

Think of a saxophone player adjusting pitch using finger holes. The sound changes dynamically, but the instrument itself remains unaltered. Similarly, dynamic transformations change the data view, not the actual data.

## ✅ Use Cases

- Currency conversions without modifying original financial records
- On-the-fly formatting in ETL pipelines
- Adapting views of metrics in dashboards

## Example: Product Pricing

The provided Python example simulates a system where a `Product` class stores price and shipping cost in a base currency. You can dynamically convert to other currencies with or without altering the stored values.

## Key Concepts

- `currency`: What the user wants to view the data in
- `_used_currency`: What the data is actually stored in
- `adapt_data`: Boolean flag to indicate if the underlying data should be transformed or just the view

## Code Files

### `product.py`

Defines:

- `Product` class with real-time currency conversion
- `Products` class to handle collections of Product

### `products_demo.py`

Creates random product instances and demonstrates dynamic viewing and transformation behavior.

## Example Run

```bash
python products_demo.py
```

You’ll see output like:

```
Viewing products in USD:
Product name                      Price   Shipping
Elixir of Eternal Youth          371.85   11.16
...

Viewing products in CHF (with adapt_data=True):
Product name                      Price   Shipping
Elixir of Eternal Youth          334.67   10.04
...

Saved internal data view:
Saved Data: self.name='Elixir of Eternal Youth', self.currency='CHF', self._used_currency='CHF' ...
```

## Bonus

This example also illustrates class composition (`Products` uses `Product`) and property-based design to separate internal state from presentation.
