import pandas as pd
import plotly.graph_objects as go
import os # Optional: Für das Erstellen eines Beispiel-CSVs

def plot_accelerometer_data(csv_filepath):
    """
    Liest Beschleunigungsdaten aus einer CSV-Datei und plottet x, y, z
    gegen den Timestamp.

    Reihen: timestamp, x, y, z.
    Es wird außerdem angenommen, dass der Timestamp in Millisekunden (ms) und die
    Beschleunigungswerte (x, y, z) in der Einheit 10⁻³ m/s² (Milli-g oder ähnliches)
    vorliegen.

    Args:
        csv_filepath (str): Der Pfad zur CSV-Datei.
    """
    try:
        column_names = ['timestamp', 'x', 'y', 'z']

        df = pd.read_csv(csv_filepath, header=None, names=column_names)


        if len(df.columns) != len(column_names):
             print(f"Warnung: Unerwartete Anzahl von Spalten ({len(df.columns)}) in '{csv_filepath}'. Erwartet wurden {len(column_names)}.")


        for col in column_names:

             df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_rows = len(df)
        df.dropna(subset=column_names, inplace=True)
        if len(df) < initial_rows:
            print(f"Info: {initial_rows - len(df)} Zeilen wurden wegen fehlender/ungültiger Werte entfernt.")


        if df.empty:
            print(f"Error: Nach der Verarbeitung enthält die Datei '{csv_filepath}' keine gültigen Daten mehr.")
            return


        df.sort_values(by='timestamp', inplace=True)

    except FileNotFoundError:
        print(f"Error: Die Datei '{csv_filepath}' wurde nicht gefunden.")
        return
    except pd.errors.EmptyDataError:
         print(f"Error: Die CSV-Datei '{csv_filepath}' ist leer.")
         return
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist beim Lesen oder Verarbeiten der CSV-Datei aufgetreten: {e}")
        return


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['x'], mode='lines', name='X', line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['y'], mode='lines', name='Y', line=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['z'], mode='lines', name='Z', line=dict(color='blue')
    ))


    fig.update_layout(
        title=f"Accelerometer Data: {os.path.basename(csv_filepath)} (Net)",
        xaxis_title="Zeit in ms",
        yaxis_title="Beschleunigung in  m/s<sup>2</sup>*10<sup>-3</sup>",
        margin=dict(l=50, r=20, t=60, b=40),
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        hovermode="x unified"
    )

    fig.show()

file_to_plot = "../collected_data/net/453.csv"
plot_accelerometer_data(file_to_plot)
