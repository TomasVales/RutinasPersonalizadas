import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk

# Cargar datos CSV
def cargar_datos(csv_file):
    try:
        # Intentar con la codificación Latin-1
        data = pd.read_csv(csv_file, encoding='latin-1')
        return data
    except UnicodeDecodeError:
        # Intentar con la codificación Windows-1252 si Latin-1 falla
        data = pd.read_csv(csv_file, encoding='windows-1252')
        return data
    except FileNotFoundError:
        # Si no existe el archivo, generar uno inicial con datos básicos
        data = pd.DataFrame({
            "edad": [],
            "peso": [],
            "altura": [],
            "horas_sueño": [],
            "desgaste_fisico": [],
            "medicamentos": [],
            "sexo": [],
            "tipo_rutina": [],
            "rutina": []
        })
        data.to_csv(csv_file, index=False)
        return data

# Preprocesar datos
def preprocesar_datos(data):
    desgaste_encoder = LabelEncoder()
    medicamentos_encoder = LabelEncoder()
    sexo_encoder = LabelEncoder()
    tipo_rutina_encoder = LabelEncoder()

    data["desgaste_fisico"] = desgaste_encoder.fit_transform(data["desgaste_fisico"])
    data["medicamentos"] = medicamentos_encoder.fit_transform(data["medicamentos"])
    data["sexo"] = sexo_encoder.fit_transform(data["sexo"])
    data["tipo_rutina"] = tipo_rutina_encoder.fit_transform(data["tipo_rutina"])

    return data, desgaste_encoder, medicamentos_encoder, sexo_encoder, tipo_rutina_encoder

# Entrenar el modelo
def entrenar_modelo(data):
    X = data.drop("rutina", axis=1)
    y = data["rutina"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = DecisionTreeClassifier()
    modelo.fit(X_scaled, y)

    return modelo, scaler

# Guardar preprocesadores y modelo
def guardar_componentes(modelo, scaler, encoders):
    joblib.dump(modelo, "modelo_rutinas.pkl")
    joblib.dump(scaler, "scaler.pkl")
    for nombre, encoder in encoders.items():
        joblib.dump(encoder, f"{nombre}_encoder.pkl")

# Predecir rutina
def predecir_rutina(edad, peso, altura, horas_sueño, desgaste_fisico, medicamentos, sexo, tipo_rutina):
    try:
        modelo = joblib.load("modelo_rutinas.pkl")
        scaler = joblib.load("scaler.pkl")
        desgaste_encoder = joblib.load("desgaste_encoder.pkl")
        medicamentos_encoder = joblib.load("medicamentos_encoder.pkl")
        sexo_encoder = joblib.load("sexo_encoder.pkl")
        tipo_rutina_encoder = joblib.load("tipo_rutina_encoder.pkl")

        desgaste_fisico = desgaste_encoder.transform([desgaste_fisico])[0]
        medicamentos = medicamentos_encoder.transform([medicamentos])[0]
        sexo = sexo_encoder.transform([sexo])[0]
        tipo_rutina = tipo_rutina_encoder.transform([tipo_rutina])[0]

        entrada = [[edad, peso, altura, horas_sueño, desgaste_fisico, medicamentos, sexo, tipo_rutina]]
        entrada_scaled = scaler.transform(entrada)

        rutina = modelo.predict(entrada_scaled)
        return rutina[0]

    except Exception as e:
        return f"Error en la predicción: {e}"

# Interfaz de usuario
def mostrar_formulario():
    def calcular_rutina():
        try:
            edad = int(entry_edad.get())
            peso = float(entry_peso.get())
            altura = float(entry_altura.get())
            horas_sueño = int(entry_sueño.get())
            desgaste_fisico = desgaste_var.get()
            medicamentos = medicamentos_var.get()
            sexo = sexo_var.get()
            tipo_rutina = tipo_rutina_var.get()

            rutina_seleccionada = data[data["tipo_rutina"] == tipo_rutina]

            rutina = predecir_rutina(edad, peso, altura, horas_sueño, desgaste_fisico, medicamentos, sexo, tipo_rutina)
            messagebox.showinfo("Rutina recomendada", f"La rutina recomendada es: {rutina}")

        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar los datos: {e}")

    # Configuración de estilo global de customtkinter
    ctk.set_appearance_mode("dark")  # Modo oscuro
    ctk.set_default_color_theme("blue")  # Tema azul

    ventana = ctk.CTk()
    ventana.title("EagleAI - Rutinas")
    ventana.geometry("450x550")

    # Cargar el ícono
    icon_path = "icon.png"  # Ruta de tu ícono (asegúrate de tenerlo en el directorio)
    icon_image = ctk.CTkImage(dark_image=Image.open(icon_path), size=(50, 50))

    # Crear un label para el ícono
    icon_label = ctk.CTkLabel(ventana, image=icon_image, text="")
    icon_label.pack(pady=(10, 0))  # Espaciado superior e inferior

    # Título
    titulo_label = ctk.CTkLabel(
        ventana, text="- Eagle AI - ",
        font=ctk.CTkFont(size=18, weight="bold"), text_color="white"
    )
    titulo_label.pack(pady=10)

    # Marco principal
    frame = ctk.CTkFrame(ventana, corner_radius=10)
    frame.pack(fill="both", expand=True, padx=20, pady=20)

    # Función para crear etiquetas y entradas
    def crear_label_input(texto, row, values=None):
        label = ctk.CTkLabel(frame, text=texto, font=ctk.CTkFont(size=14))
        label.grid(row=row, column=0, sticky="w", pady=5, padx=5)

        if values:
            entry = ctk.CTkComboBox(frame, values=values)
        else:
            entry = ctk.CTkEntry(frame)

        entry.grid(row=row, column=1, pady=5, padx=5)
        return entry

    # Campos del formulario
    tipo_rutina_var = crear_label_input("Tipo de rutina", 0, ["Hipertrofia", "Fuerza", "Running", "Flexibilidad", "Potencia", "Descenso de peso"])
    entry_edad = crear_label_input("Edad", 1)
    entry_peso = crear_label_input("Peso (kg)", 2)
    entry_altura = crear_label_input("Altura (m)", 3)
    entry_sueño = crear_label_input("Horas de sueño", 4)
    desgaste_var = crear_label_input("Desgaste físico", 5, ["Bajo", "Medio", "Alto"])
    medicamentos_var = crear_label_input("¿Usa medicamentos?", 6, ["Sí", "No"])
    sexo_var = crear_label_input("Sexo", 7, ["Hombre", "Mujer"])

    # Botón para generar rutina
    boton_generar = ctk.CTkButton(
        frame, text="Generar Rutina", command=calcular_rutina, font=ctk.CTkFont(size=14, weight="bold")
    )
    boton_generar.grid(row=8, column=0, columnspan=2, pady=20, padx=(100, 0))

    ventana.mainloop()

# Cargar datos, preprocesar y entrenar
csv_file = "rutinas.csv"
data = cargar_datos(csv_file)
data, desgaste_encoder, medicamentos_encoder, sexo_encoder, tipo_rutina_encoder = preprocesar_datos(data)
modelo, scaler = entrenar_modelo(data)

guardar_componentes(modelo, scaler, {
    "desgaste": desgaste_encoder,
    "medicamentos": medicamentos_encoder,
    "sexo": sexo_encoder,
    "tipo_rutina": tipo_rutina_encoder
})

mostrar_formulario()
