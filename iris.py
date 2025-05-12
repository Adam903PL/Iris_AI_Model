from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

iris_data = load_iris()
features = iris_data.data
labels = iris_data.target

normalizer = StandardScaler()
features_scaled = normalizer.fit_transform(features)

labels_one = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels_one, test_size=0.2, random_state=42
)

classifier = Sequential([
    Input(shape=(4,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

# Kompilacja modelu
classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback wczesnego zatrzymania
stop_early = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=1e-3,
    mode='min',
    restore_best_weights=True
)

# Trenowanie modelu
classifier.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[stop_early]
)

# Ocena modelu na zbiorze testowym
test_loss, test_accuracy = classifier.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
