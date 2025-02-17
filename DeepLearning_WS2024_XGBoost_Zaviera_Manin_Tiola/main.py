####################################################################
####################################################################
# @author Zaviera Manin Tiola
# @date 01.02.2025
#
# Dieser Code wertet die Datei creditcard.csv mit Kreditkartendaten aus.
# Hierbei wird die Funktionalitaet von XGBoost und logistischer Regression
# verglichen. Es wird zudem eine Visualisierung mit matplotlib erstellt.
#
#
#
#####################################################################
#####################################################################

#importiere alle wichtigen Bibliotheken, pandas, sklear, xgboost etc.
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Delimiter definiert die Trennung zwischen den Werten
def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=';', skiprows=[0,1])
    #data.astype(object).mask(data <= 6)
    return data.astype(int)


# Daten vorbereiten
def prepare_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.75, train_size=0.25, random_state=777)


# Modell trainieren und evaluieren
def train_xgboost_classifier(X_train, X_test, y_train, y_test):
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Classification Accuracy: {accuracy:.2f}")

    # Visualisierung der Konfusionsmatrix für XGBoost
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for XGBoost Classifier")
    plt.show()
    class_report = classification_report(y_test, predictions,output_dict=True)
    
    # Entfernen der Durchschnittswerte für eine reine Klassenübersicht
    class_report.pop("accuracy", None)
    class_report.pop("macro avg", None)
    class_report.pop("weighted avg", None)

    # Extrahieren der Metriken und Klassen
    labels = list(class_report.keys())
    metrics = ["precision", "recall", "f1-score"]
    values = np.array([[class_report[label][metric] for metric in metrics] for label in labels])

    # Erstellen des Balkendiagramms
    x = np.arange(len(labels))
    width = 0.2  # Breite der Balken

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[:, i], width, label=metric)

    # Hinzufügen der Accuracy als separaten Balken
    ax.bar(len(labels), accuracy, width, label="Accuracy", color="red")

    # Achsentitel setzen
    ax.set_xlabel("Klassen")
    ax.set_ylabel("Wert")
    ax.set_title("Klassifikationsbericht XGBoost (Precision, Recall, F1-Score)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()

    # Werte über den Balken anzeigen
    for i in range(len(labels)):
        for j in range(len(metrics)):
            ax.text(x[i] + j * width, values[i, j] + 0.02, f"{values[i, j]:.2f}", ha='center', fontsize=8)

    # Diagramm anzeigen
    plt.show()
    return model
    
# XGBoost Regressor wird hier nicht angewendet aber als If-else verfuegbar
def train_xgboost_regressor(X_train, X_test, y_train, y_test):
    model = XGBRegressor(eval_metric='rmse')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Regression Mean Squared Error: {mse:.2f}")

    # Visualisierung der Konfusionsmatrix
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for XGBoost Regressor")
    plt.show()
    return model


def logistic_regression_analysis(X_train, X_test, y_train, y_test):

    # Daten normalisieren
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistische Regression trainieren
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Vorhersagen treffen
    y_pred = model.predict(X_test)

    # Modellbewertung
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred,output_dict=True)

    print(f'Genauigkeit: {accuracy:.2f}')
    print('Konfusionsmatrix:')
    print(conf_matrix)
    print('Klassifikationsbericht:')
    print(class_report)

    # Konfusionsmatrix visualisieren für logistische Regression
    plt.figure(figsize=(6, 4))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Vorhergesagte Labels')
    plt.ylabel('Tatsächliche Labels')
    plt.title('Konfusionsmatrix')
    plt.show()

    # Entfernen der Durchschnittswerte für eine reine Klassenübersicht
    class_report.pop("accuracy", None)
    class_report.pop("macro avg", None)
    class_report.pop("weighted avg", None)

    # Extrahieren der Metriken und Klassen
    labels = list(class_report.keys())
    metrics = ["precision", "recall", "f1-score"]
    values = np.array([[class_report[label][metric] for metric in metrics] for label in labels])

    # Erstellen des Balkendiagramms
    x = np.arange(len(labels))
    width = 0.2  # Breite der Balken

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[:, i], width, label=metric)

    # Hinzufügen der Accuracy als separaten Balken
    ax.bar(len(labels), accuracy, width, label="Accuracy", color="red")

    # Achsentitel setzen
    ax.set_xlabel("Klassen")
    ax.set_ylabel("Wert")
    ax.set_title("Klassifikationsbericht logistische Regression (Precision, Recall, F1-Score)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()

    # Werte über den Balken anzeigen
    for i in range(len(labels)):
        for j in range(len(metrics)):
            ax.text(x[i] + j * width, values[i, j] + 0.02, f"{values[i, j]:.2f}", ha='center', fontsize=8)

    # Diagramm anzeigen
    plt.show()


def main():
    # Datei-Pfad zur CSV-Datei eingeben
    file_path = "creditcard.csv"
    target_column = "25"  # Spaltenname des Zielwerts Kreditwuerdigkeit

    # Daten laden und vorbereiten
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(data, target_column)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # Prüfen, ob die Zielvariable kategorisch (Klassifikation) oder numerisch (Regression) ist
    if (data[target_column].dtype == 'object' or len(data[target_column].unique()) < 10):
        print("Performing Classification...")
        train_xgboost_classifier(X_train, X_test, y_train, y_test)
    else:
        print("Performing Regression...")
        train_xgboost_regressor(X_train, X_test, y_train, y_test)

    # Funktion aufrufen logistische Regression zum Vergleich
    logistic_regression_analysis(X_train, X_test, y_train, y_test)

# Catche warnings die nerven
if __name__ == "__main__":
    with warnings.catch_warnings(action="ignore"):main()
