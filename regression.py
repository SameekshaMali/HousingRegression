from utils import load_data, get_models, evaluate_model
from sklearn.model_selection import train_test_split

df = load_data()
X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = get_models(X_train, y_train)
for name, model in models.items():
    model.fit(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"{name} → MSE: {mse:.2f}, R²: {r2:.2f}")
