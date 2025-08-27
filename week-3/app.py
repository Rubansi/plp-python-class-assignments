def calculate_discount(price, discount_percent):
    if discount_percent >= 20:
        discount_amount = price * (discount_percent / 100)
        final_price = price - discount_amount
        return final_price
    else:
        return price

# Prompt user input
original_price = float(input("Enter the original price of the item: "))
discount_percentage = float(input("Enter the discount percentage: "))

# Calculate final price
final_price = calculate_discount(original_price, discount_percentage)

# Print the result
if discount_percentage >= 20:
    print(f"Final price after {discount_percentage}% discount: {final_price:.2f}")
else:
    print(f"No discount applied. Price remains: {original_price:.2f}")
