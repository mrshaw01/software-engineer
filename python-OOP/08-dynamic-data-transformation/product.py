import random


class Product:
    conversion_rates = {"USD": 1, "EUR": 0.92, "CHF": 0.90, "GBP": 0.79}

    def __init__(self, name, price, shipping_cost, currency="USD"):
        self.name = name
        self._price = price
        self._shipping_cost = shipping_cost
        self.currency = currency
        self._used_currency = currency

    def set_currency(self, new_currency, adapt_data=False):
        if self.currency != new_currency:
            self.currency = new_currency
        if adapt_data:
            self._price = self.price
            self._shipping_cost = self.shipping_cost
            self._used_currency = new_currency

    @property
    def price(self):
        return self._convert_currency(self._price)

    @property
    def shipping_cost(self):
        return self._convert_currency(self._shipping_cost)

    def _convert_currency(self, amount):
        factor = Product.conversion_rates[self.currency] / Product.conversion_rates[self._used_currency]
        return round(amount * factor, 2)

    def __str__(self):
        return f"{self.name:35s}   {self.price:7.2f}   {self.shipping_cost:6.2f}"

    def show_saved_data(self):
        print(
            f"Saved Data: self.name='{self.name}', self.currency='{self.currency}', self._used_currency='{self._used_currency}' self._price={self._price}, self._shipping_cost={self._shipping_cost}"
        )


class Products:

    def __init__(self):
        self.product_list = []

    def add_product(self, product):
        self.product_list.append(product)

    def view_products(self, currency="USD", adapt_data=False):
        print(f"{'Product name':39s} Price Shipping")
        for product in self.product_list:
            product.set_currency(currency, adapt_data)
            print(product)

    def view_products_as_saved_data(self):
        for product in self.product_list:
            product.show_saved_data()
