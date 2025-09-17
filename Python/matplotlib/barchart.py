import matplotlib.pyplot as plt

# Data
categories = ["A", "B", "C", "D"]
values = [3, 7, 5, 2]

plt.bar(categories, values, color="orange")
plt.title("Bar Chart Example")
plt.xlabel("Category")
plt.ylabel("Value")
plt.savefig('barchart.png')
plt.show()
