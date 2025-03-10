from langdetect import detect
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np

RUTA_TRAIN = "../train_limpio.csv"
RUTA_TEST = "../test.csv"

def detectar_idioma(texto):
    try:
        return detect(texto)
    except:
        return "desconocido"

def filtrar_idioma(df):

    df["idioma_detectado"] = df["review_es"].apply(detectar_idioma)

    return df[df["idioma_detectado"] == "es"]

def normalizar_texto(df, columna):
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
    df[columna] = df[columna].str.lower()
    for clave, valor in reemplazos.items():
        df[columna] = df[columna].str.replace(clave, valor)
    return df[columna]

def reemplazar_no(df):
    def reemplazo(texto):
        if isinstance(texto, str):
            return texto.replace(" película ", "")
        return texto

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(reemplazo)

    return df

def imprimir_resultados(y, y_pred):

    labels = [0, 1]

    f1 = f1_score(y, y_pred, pos_label=0, labels=labels)
    precision = precision_score(y, y_pred, pos_label=0, labels=labels)
    recall = recall_score(y, y_pred, pos_label=0, labels=labels)
    accuracy = accuracy_score(y, y_pred)

    cm = confusion_matrix(y, y_pred)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    print(f"F1: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")

def guardar_resultado_en_csv(prediccion, df_test, ruta):

    prediccion = np.where(prediccion == 0, "negativo", "positivo")

    df_predicciones = pd.DataFrame({"ID": df_test["ID"], "prediccion": prediccion})

    df_predicciones.to_csv(ruta, header=["ID", "sentimiento"], index=False)