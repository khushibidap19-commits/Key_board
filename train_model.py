import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_PATH = "gesture_data.csv"
MODEL_PATH = "gesture_model.h5"
ENCODER_PATH = "label_encoder.npy"

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype("float32")
    y_raw = df["label"].values
    return X, y_raw

def build_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    print("[INFO] Loading data...")
    X, y_raw = load_data(DATA_PATH)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"[INFO] Found {num_classes} gesture classes: {list(le.classes_)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    print("[INFO] Training...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)
    model.save(MODEL_PATH)
    np.save(ENCODER_PATH, le.classes_)
    print(f"[DONE] Model saved to {MODEL_PATH}")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test Accuracy: {test_acc*100:.2f}%")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history.history["accuracy"], label="Train Acc")
    axs[0].plot(history.history["val_accuracy"], label="Val Acc")
    axs[0].set_title("Accuracy over Epochs")
    axs[0].legend()
    axs[1].plot(history.history["loss"], label="Train Loss")
    axs[1].plot(history.history["val_loss"], label="Val Loss")
    axs[1].set_title("Loss over Epochs")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax2, cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("[INFO] Saved training_history.png and confusion_matrix.png")

if __name__ == "__main__":
    main()
