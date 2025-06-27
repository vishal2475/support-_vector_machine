import pandas as pd
from sklearn.svm import SVC

# Step 1: Sample hotel data
data = {
    'Hotel': ['Royal Palace', 'Sunview Resort', 'Budget Inn', 'Ocean Pearl', 'City Lodge'],
    'Rating': [4.5, 3.8, 3.0, 4.2, 2.5],
    'Price': [8000, 5000, 3000, 7000, 2500]
}
df = pd.DataFrame(data)

salary = int(input("Enter your Monthly Salary: "))
hotel = input("Select a Hotel (Royal Palace, Sunview Resort, Budget Inn, Ocean Pearl, City Lodge): ")

# Step 3: Label generation
max_price = salary * 0.25
df['Affordable'] = df['Price'].apply(lambda x: 1 if x <= max_price else 0)

# Step 4: Train SVM model
X = df[['Rating', 'Price']]
y = df['Affordable']
model = SVC(kernel='linear')
model.fit(X, y)

# Step 5: Predict selected hotel
selected = df[df['Hotel'] == hotel].iloc[0]
rating = selected['Rating']
price = selected['Price']

prediction = model.predict([[rating, price]])[0]

# Step 6: Print result
if prediction == 1:
    print(f"\n{hotel} is affordable for you!")
else:
    print(f"\n {hotel} is NOT affordable based on your salary.")
