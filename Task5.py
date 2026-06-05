import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

data = {
    "Food_Item": [
        "Pizza", "Burger", "Pasta", "Salad", "IceCream",
        "Sandwich", "FrenchFries", "Dosa", "Idli", "Biryani",
        "Paneer", "FriedRice", "Noodles",
        "Apple", "Banana", "Orange", "Mango",
        "Cake", "Donut", "Samosa"
    ],

    "Calories": [
        285, 295, 220, 150, 207,
        250, 312, 168, 58, 290,
        350, 330, 280,
        95, 105, 62, 135,
        371, 452, 262
    ]
}

df = pd.DataFrame(data)

print("\nDataset:\n")
print(df)

encoder = LabelEncoder()
df["Food_Label"] = encoder.fit_transform(df["Food_Item"])

X = df[["Calories"]]
y = df["Food_Label"]


model = KNeighborsClassifier(n_neighbors=1)

model.fit(X, y)


cal = int(input("\nEnter Calories: "))

input_data = pd.DataFrame([[cal]], columns=["Calories"])

prediction = model.predict(input_data)

food_name = encoder.inverse_transform(prediction)

print("\nPredicted Food Item:", food_name[0])
print("Estimated Calories:", cal, "kcal")