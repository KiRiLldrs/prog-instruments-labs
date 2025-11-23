from abc import ABC, abstractmethod

from model_objects import Discount, Offer


class OfferStrategy(ABC):
    @abstractmethod
    def calculate_discount(self, offer: Offer, quantity: int, unit_price: float, product)->Discount | None:
        pass


class ThreeForTwoStrategy(OfferStrategy):
    def calculate_discount(self, offer, quantity, unit_price, product)->Discount | None:
        if quantity<3: return None

        x = 3
        number_of_x = quantity//x
        discount_amount = quantity * unit_price - ((number_of_x * 2 * unit_price) + quantity % 3 * unit_price)
        return Discount(product, "3 for 2", -discount_amount)
    

class TwoForAmountStrategy(OfferStrategy):
    def calculate_discount(self, offer, quantity, unit_price, product)->Discount | None:
        if quantity<2: return None

        x = 2
        number_of_x = quantity//x
        total = offer.argument * (quantity // number_of_x) + quantity % 2 * unit_price
        discount_n = unit_price * quantity - total
        return Discount(product, "2 for " + str(offer.argument), -discount_n)
    

class FiveForAmountStrategy(OfferStrategy):
    def calculate_discount(self, offer, quantity, unit_price, product)->Discount | None:
        if quantity<5: return None

        x = 5
        number_of_x = quantity//x
        discount_total = unit_price * quantity - (
            offer.argument * number_of_x + quantity % 5 * unit_price)
        return Discount(product, str(x) + " for " + str(offer.argument), -discount_total)
    

class TenPercentDiscountStrategy(OfferStrategy):
    def calculate_discount(self, offer, quantity, unit_price, product)->Discount | None:
        return Discount(product, str(offer.argument) + "% off",
                                -quantity * unit_price * offer.argument / 100.0)