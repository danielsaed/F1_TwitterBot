import numpy as np
from scipy import interpolate
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go
import numpy as np
from scipy import interpolate
from pyproj import Transformer
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import re
from scipy.optimize import least_squares
from scipy.interpolate import interp1d, splprep, splev
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def interpolate_telemetry(telemetry_df, new_distance_step=1.0):
    """
    Interpolate telemetry data to have points at regular intervals with specific
    interpolation methods for different data types.
    """
    original_distance = telemetry_df['Distance'].values
    
    # Create new distance points with regular intervals
    new_distance = np.arange(
        original_distance[0], 
        original_distance[-1], 
        new_distance_step
    )
    
    # Create a new DataFrame to hold interpolated data
    interpolated_df = pd.DataFrame({'Distance': new_distance})
    
    # Define appropriate interpolation types for different columns
    discrete_columns = ['nGear', 'DRS']  # Columns that should use 'nearest'
    
    # Interpolate each column
    for column in telemetry_df.columns:
        if column != 'Distance':
            # Choose interpolation type based on column
            if column in discrete_columns:
                interp_kind = 'nearest'
            elif column in ['Speed', 'RPM', 'Throttle', 'Brake']:
                interp_kind = 'linear'
            else:
                interp_kind = 'linear'
                
            # Create an interpolation function for this column
            f = interpolate.interp1d(
                original_distance,
                telemetry_df[column].values,
                kind=interp_kind,
                bounds_error=False,
                fill_value=(telemetry_df[column].iloc[0], telemetry_df[column].iloc[-1])
            )
            
            # Apply the interpolation function to get new values
            interpolated_df[column] = f(new_distance)
            
            # Redondear valores enteros si es necesario
            if column in discrete_columns:
                interpolated_df[column] = interpolated_df[column].round().astype(int)
    
    return interpolated_df

def get_lapFromObject(laptime):
    
    l = []
    characters = "days "
    for i in range(len(characters)):
        laptime = laptime.replace(characters[i],"")
    
    laptime = re.sub(r'.', '', laptime, count = 5)
    if "000" in laptime:
        laptime = laptime.replace("000","")
    l = laptime.split()
    lap = l[0]
   

    while True:
        if(len(lap) <= 8):
            if lap[0] == ":" or lap[0] == "0" :
                if len(lap) == 1:
                    lap = '2:00.000'
                else:
                    lap= lap[1:]
            else:
                break
        else:
            lap= lap[1:]
    diferencia = 8- len(lap)
    arraDif = ".000"
    if len(lap) < 8:
        lap = lap + arraDif[-abs(diferencia):]
    return lap

def fill_fastest_values(delta_df):
    """
    Rellena los valores 0 en la columna 'Fastest' con el último valor no-cero (1 o -1).
    
    Args:
        delta_df: DataFrame con la columna 'Fastest' que contiene valores -1, 0, 1
        
    Returns:
        DataFrame con los valores 0 en 'Fastest' reemplazados
    """
    # Crear una copia para no modificar el original
    df = delta_df.copy()
    
    # Asegurarse de que los datos estén ordenados
    df = df.sort_index()
    
    # Valor actual para rellenar (inicializar con el primer valor no-cero)
    first_non_zero_idx = df[df['Fastest'] != 0]['Fastest'].first_valid_index()
    
    # Si no hay valores no-cero, no hay nada que hacer
    if first_non_zero_idx is None:
        return df
    
    current_fill_value = df.loc[first_non_zero_idx, 'Fastest']
    
    # Rellenar los valores 0 con el último valor no-cero
    for idx in df.index:
        if df.loc[idx, 'Fastest'] == 0:
            df.loc[idx, 'Fastest'] = current_fill_value
        else:
            # Actualizar el valor de relleno cuando encontramos un valor no-cero
            current_fill_value = df.loc[idx, 'Fastest']
    
    return df

# Detectar crestas y valles en los datos de velocidad
def detect_peaks_and_valleys(x, y, prominence=10, distance=150):
    """
    Detecta crestas (máximos) y valles (mínimos) en los datos.
    
    Args:
        x: Array con las distancias
        y: Array con las velocidades
        prominence: Prominencia mínima para considerar un punto como cresta o valle
        distance: Distancia mínima entre picos consecutivos
    
    Returns:
        peaks_indices: Índices de las crestas (máximos)
        valleys_indices: Índices de los valles (mínimos)
    """
    
    
    # Encontrar crestas (máximos)
    peaks_indices, _ = find_peaks(y, prominence=prominence, distance=distance)
    
    # Encontrar valles (mínimos) (invirtiendo los datos)
    valleys_indices, _ = find_peaks(-y, prominence=prominence, distance=distance)
    
    return peaks_indices, valleys_indices

def add_speed_annotations(fig, x1, y1, x2, y2, name1, name2, color1, color2, row=1, col=1, offset=20, min_peak_speed=250):
    """
    Añade anotaciones para velocidades máximas (crestas) y mínimas (valles).
    - Para máximos: Solo muestra los que superen min_peak_speed
    - Para mínimos: Compara valores cercanos (hasta 100m de diferencia) y siempre muestra menor abajo
    
    Args:
        fig: Figura de Plotly
        x1, y1: Arrays con las distancias y velocidades del primer piloto (Sainz)
        x2, y2: Arrays con las distancias y velocidades del segundo piloto (Hamilton)
        name1, name2: Nombres de los pilotos
        color1, color2: Colores para las anotaciones
        row, col: Posición del subplot
        offset: Desplazamiento vertical para las anotaciones
        min_peak_speed: Velocidad mínima para mostrar picos (solo para crestas)
    """
    
    
    # Convertir a arrays numpy para operaciones más eficientes
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    
    # Detectar crestas (máximos)
    peaks1_idx, _ = find_peaks(y1, prominence=10, distance=100)
    peaks2_idx, _ = find_peaks(y2, prominence=10, distance=100)
    
    # Detectar valles (mínimos)
    valleys1_idx, _ = find_peaks(-y1, prominence=10, distance=100)
    valleys2_idx, _ = find_peaks(-y2, prominence=10, distance=100)
    
    # Crear listas de puntos con sus coordenadas
    # Para picos, solo incluir aquellos con velocidad > min_peak_speed
    peaks1 = [(x1[i], y1[i]) for i in peaks1_idx if i < len(x1) and y1[i] > min_peak_speed]
    peaks2 = [(x2[i], y2[i]) for i in peaks2_idx if i < len(x2) and y2[i] > min_peak_speed]
    
    # Para valles, incluir todos
    valleys1 = [(x1[i], y1[i]) for i in valleys1_idx if i < len(x1)]
    valleys2 = [(x2[i], y2[i]) for i in valleys2_idx if i < len(x2)]
    
    # Procesar los picos (velocidades máximas) - Comparar en el mismo punto
    all_peaks = []
    
    # Añadir los puntos de pico de Sainz
    for x, y in peaks1:
        all_peaks.append((x, 'peak', 1, y))
    
    # Añadir los puntos de pico de Hamilton
    for x, y in peaks2:
        all_peaks.append((x, 'peak', 2, y))
    
    # Ordenar y agrupar picos
    all_peaks.sort(key=lambda p: p[0])
    
    # Agrupar picos cercanos (dentro de 50m de distancia)
    grouped_peaks = []
    current_group = []
    
    for point in all_peaks:
        if not current_group or abs(point[0] - current_group[0][0]) < 50:
            current_group.append(point)
        else:
            grouped_peaks.append(current_group)
            current_group = [point]
    
    if current_group:
        grouped_peaks.append(current_group)
    
    # Para cada grupo de picos, añadir anotaciones comparando velocidades
    for group in grouped_peaks:
        distances = [p[0] for p in group]
        avg_distance = sum(distances) / len(distances)
        
        # Encontrar las velocidades de ambos pilotos en este punto
        idx1 = np.abs(x1 - avg_distance).argmin()
        idx2 = np.abs(x2 - avg_distance).argmin()
        
        speed1 = y1[idx1]
        speed2 = y2[idx2]
        
        # Para picos, verificar que al menos una de las velocidades sea > min_peak_speed
        if max(speed1, speed2) <= min_peak_speed:
            continue  # Saltar este punto si ninguna velocidad supera el umbral
        
        # Añadir un símbolo para indicar máximo
        fig.add_trace(
            go.Scatter(
                x=[avg_distance],
                y=[(speed1 + speed2) / 2],  # Punto medio entre ambas velocidades
                mode='markers',
                marker=dict(
                    symbol="triangle-up",
                    size=8,
                    color='gray',
                    line=dict(width=1, color='white')
                ),
                showlegend=False,
                hoverinfo='none'
            ),
            row=row+1, col=col
        )
        
        # Añadir anotaciones con la velocidad mayor arriba y menor abajo
        if speed1 >= speed2:
            # Sainz más rápido o igual, va arriba
            fig.add_annotation(
                x=avg_distance,
                y=speed1,
                text=f"{int(speed1)}",
                showarrow=False,
                font=dict(color=color1, size=10),
                xref=f"x",
                yref=f"y{row}",
                yshift=offset,  # Positivo = arriba
                bgcolor="rgba(0,0,0,0.5)",

            )
            
            fig.add_annotation(
                x=avg_distance,
                y=speed2,
                text=f"{int(speed2)}",
                showarrow=False,
                font=dict(color=color2, size=10),
                xref=f"x",
                yref=f"y{row}",
                yshift=-offset,  # Negativo = abajo
                bgcolor="rgba(0,0,0,0.5)",

            )
        else:
            # Hamilton más rápido, va arriba
            fig.add_annotation(
                x=avg_distance,
                y=speed2,
                text=f"{int(speed2)}",
                showarrow=False,
                font=dict(color=color2, size=10),
                xref=f"x",
                yref=f"y{row}",
                yshift=offset,  # Positivo = arriba
                bgcolor="rgba(0,0,0,0.5)",

            )
            
            fig.add_annotation(
                x=avg_distance,
                y=speed1,
                text=f"{int(speed1)}",
                showarrow=False,
                font=dict(color=color1, size=10),
                xref=f"x",
                yref=f"y{row}",
                yshift=-offset,  # Negativo = abajo
                bgcolor="rgba(0,0,0,0.5)",

            )
    
    # Procesar y emparejar valles cercanos (menos de 100m de distancia)
    # Combinar y ordenar todos los valles
    all_valleys = []
    for x, y in valleys1:
        all_valleys.append((x, 1, y))  # 1 = Sainz
    for x, y in valleys2:
        all_valleys.append((x, 2, y))  # 2 = Hamilton
    
    all_valleys.sort(key=lambda v: v[0])  # Ordenar por distancia
    
    # Emparejar valles cercanos
    i = 0
    while i < len(all_valleys):
        current = all_valleys[i]
        
        # Buscar otro valle cercano del otro piloto
        j = i + 1
        found_pair = False
        
        while j < len(all_valleys) and all_valleys[j][0] - current[0] <= 100:
            if all_valleys[j][1] != current[1]:  # Diferente piloto
                # Encontramos un par de valles cercanos
                valley1 = current if current[1] == 1 else all_valleys[j]
                valley2 = all_valleys[j] if current[1] == 1 else current
                
                # Obtener posiciones y velocidades
                x_val1, _, y_val1 = valley1
                x_val2, _, y_val2 = valley2
                
                # Calcular posición promedio para el marcador
                avg_x = (x_val1 + x_val2) / 2
                
                # Añadir un símbolo para indicar mínimo en la posición promedio
                fig.add_trace(
                    go.Scatter(
                        x=[avg_x],
                        y=[(y_val1 + y_val2) / 2],  # Punto medio entre ambas velocidades
                        mode='markers',
                        marker=dict(
                            symbol="triangle-down",
                            size=8,
                            color='gray',
                            line=dict(width=1, color='white')
                        ),
                        showlegend=False,
                        hoverinfo='none'
                    ),
                    row=row+1, col=col
                )
                
                # Determinar cuál velocidad es menor y cuál es mayor
                # Siempre mostrar la menor abajo y la mayor arriba
                if y_val1 <= y_val2:
                    # Sainz más lento, va abajo
                    fig.add_annotation(
                        x=avg_x,
                        y=y_val1,
                        text=f"{int(y_val1)}",
                        showarrow=False,
                        font=dict(color=color1, size=10),
                        xref=f"x",
                        yref=f"y{row}",
                        yshift=-offset,  # Negativo = abajo
                        bgcolor="rgba(0,0,0,0.5)",

                    )
                    
                    fig.add_annotation(
                        x=avg_x,
                        y=y_val2,
                        text=f"{int(y_val2)}",
                        showarrow=False,
                        font=dict(color=color2, size=10),
                        xref=f"x",
                        yref=f"y{row}",
                        yshift=offset,  # Positivo = arriba
                        bgcolor="rgba(0,0,0,0.5)",

                    )
                else:
                    # Hamilton más lento, va abajo
                    fig.add_annotation(
                        x=avg_x,
                        y=y_val2,
                        text=f"{int(y_val2)}",
                        showarrow=False,
                        font=dict(color=color2, size=10),
                        xref=f"x",
                        yref=f"y{row}",
                        yshift=-offset,  # Negativo = abajo
                        bgcolor="rgba(0,0,0,0.5)",

                    )
                    
                    fig.add_annotation(
                        x=avg_x,
                        y=y_val1,
                        text=f"{int(y_val1)}",
                        showarrow=False,
                        font=dict(color=color1, size=10),
                        xref=f"x",
                        yref=f"y{row}",
                        yshift=offset,  # Positivo = arriba
                        bgcolor="rgba(0,0,0,0.5)",

                    )
                
                # Marcar como encontrado y avanzar
                found_pair = True
                i = j + 1
                break
            j += 1
        
        if not found_pair:
            # No encontramos pareja cercana, mostrar solo este valle
            x_val, pilot, y_val = current
            
            # Añadir un símbolo para indicar mínimo
            fig.add_trace(
                go.Scatter(
                    x=[x_val],
                    y=[y_val],
                    mode='markers',
                    marker=dict(
                        symbol="triangle-down",
                        size=8,
                        color='gray',
                        line=dict(width=1, color='white')
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row=row+1, col=col
            )
            
            # Añadir etiqueta con la velocidad mínima
            if pilot == 1:  # Sainz
                fig.add_annotation(
                    x=x_val,
                    y=y_val,
                    text=f"{int(y_val)}",
                    showarrow=False,
                    font=dict(color=color1, size=10),
                    xref=f"x",
                    yref=f"y{row}",
                    yshift=-offset/2,  # Ligeramente abajo
                    bgcolor="rgba(0,0,0,0.5)",

                )
            else:  # Hamilton
                fig.add_annotation(
                    x=x_val,
                    y=y_val,
                    text=f"{int(y_val)}",
                    showarrow=False,
                    font=dict(color=color2, size=10),
                    xref=f"x",
                    yref=f"y{row}",
                    yshift=-offset/2,  # Ligeramente abajo
                    bgcolor="rgba(0,0,0,0.5)",

                )
            i += 1







def interpolate_xy_coordinates(df, num_points_between=1, position_cols=['X', 'Y'], 
                               time_col='Time', distance_col='Distance', method='linear'):
    """
    Interpola los puntos X,Y para aumentar la densidad de puntos, lo que mejora el cálculo de radios.
    Permite especificar cuántos puntos interpolados añadir entre cada par de puntos originales.
    
    Args:
        df: DataFrame con datos de posición.
        num_points_between: Número de puntos a interpolar entre cada par de puntos originales (default: 1).
        position_cols: Lista de columnas que contienen coordenadas (por defecto ['X', 'Y']).
        time_col: Nombre de la columna que contiene el tiempo (por defecto 'Time').
        distance_col: Nombre de la columna que contiene la distancia (por defecto 'Distance').
        method: Método de interpolación ('linear', 'cubic', 'spline', etc.). Por defecto 'linear'.
        
    Returns:
        DataFrame con puntos adicionales interpolados entre los puntos originales.
    """
    import pandas as pd
    import numpy as np
    from scipy.interpolate import interp1d
    
    # Crear una copia del DataFrame original para no modificarlo
    original_df = df.copy().sort_values(by=distance_col).reset_index(drop=True)
    
    # Crear un array de nuevas distancias intercaladas entre las originales
    original_distances = original_df[distance_col].values
    
    # Crear un conjunto de nuevos puntos entre cada par de puntos originales
    new_distances = []
    # Guardar un índice para saber cuáles son puntos originales y cuáles interpolados
    is_original_point = []
    
    for i in range(len(original_distances) - 1):
        # Añadir el punto original
        new_distances.append(original_distances[i])
        is_original_point.append(True)
        
        # Añadir 'num_points_between' puntos interpolados equidistantes
        start_dist = original_distances[i]
        end_dist = original_distances[i + 1]
        segment_length = end_dist - start_dist
        
        for j in range(1, num_points_between + 1):
            # Calcular distancia en proporción j/n+1 entre start y end
            interp_dist = start_dist + segment_length * (j / (num_points_between + 1))
            new_distances.append(interp_dist)
            is_original_point.append(False)
    
    # Añadir el último punto original
    new_distances.append(original_distances[-1])
    is_original_point.append(True)
    
    # Convertir a array de numpy para procesar más rápido
    new_distances = np.array(new_distances)
    
    # Crear un nuevo DataFrame para los datos interpolados
    interpolated_df = pd.DataFrame({
        distance_col: new_distances,
        'is_original': is_original_point
    })
    
    # Interpolar solo las columnas especificadas
    columns_to_interpolate = position_cols + [time_col]
    
    for col in columns_to_interpolate:
        if col in original_df.columns:
            # Crear función de interpolación
            f = interp1d(
                original_distances,
                original_df[col].values,
                kind=method,
                bounds_error=False,
                fill_value="extrapolate"
            )
            
            # Aplicar la interpolación
            interpolated_df[col] = f(new_distances)
    
    # Para las demás columnas, copiar solo los valores originales y dejar NaN en interpolados
    other_columns = [col for col in original_df.columns if col not in columns_to_interpolate + [distance_col]]
    
    # Inicializar las columnas restantes con NaN
    for col in other_columns:
        interpolated_df[col] = np.nan
        
    # Copiar los valores originales para las filas que corresponden a puntos originales
    original_index = 0
    for i, is_original in enumerate(is_original_point):
        if is_original:
            for col in other_columns:
                if col in original_df.columns:
                    interpolated_df.loc[i, col] = original_df.loc[original_index, col]
            original_index += 1
    
    # Eliminar la columna auxiliar 'is_original'
    interpolated_df = interpolated_df.drop('is_original', axis=1)
    
    print(f"Puntos originales: {len(original_df)}")
    print(f"Puntos interpolados: {len(interpolated_df)}")
    print(f"Aumento: {len(interpolated_df) / len(original_df):.2f}x")
    
    return interpolated_df

def calibrate_xy_coordinates(telemetry_df, distance_col='Distance'):
    """
    Calibra las coordenadas X,Y para que sean proporcionales a la distancia real recorrida.
    
    Args:
        telemetry_df: DataFrame con columnas 'X', 'Y' y 'Distance'
        distance_col: Nombre de la columna que contiene la distancia real en metros
    
    Returns:
        DataFrame con las columnas 'X_scaled' y 'Y_scaled' añadidas
    """
    # Crear una copia para no modificar el original
    df = telemetry_df.copy()
    
    # Asegurarse de que el DataFrame esté ordenado por tiempo/distancia
    df = df.sort_values(by=distance_col).reset_index(drop=True)
    
    # Calcular distancias euclidianas entre puntos consecutivos en el espacio X,Y
    xy_distances = [0]  # Primer punto tiene distancia 0
    for i in range(1, len(df)):
        xy_dist = euclidean(
            [df['X'].iloc[i-1], df['Y'].iloc[i-1]],
            [df['X'].iloc[i], df['Y'].iloc[i]]
        )
        xy_distances.append(xy_dist)
    
    # Calcular distancia euclidiana acumulada
    df['XY_CumulativeDist'] = np.cumsum(xy_distances)
    
    # Calcular el factor de escala global
    # Divide la distancia real total entre la distancia euclidiana total
    real_dist_total = df[distance_col].iloc[-1] - df[distance_col].iloc[0]
    xy_dist_total = df['XY_CumulativeDist'].iloc[-1]
    
    scale_factor = real_dist_total / xy_dist_total if xy_dist_total > 0 else 1.0
    
    print(f"Factor de escala calculado: {scale_factor:.4f}")
    print(f"Distancia real total: {real_dist_total:.2f} metros")
    print(f"Distancia euclidiana total: {xy_dist_total:.2f} unidades")
    
    # Aplicar la transformación de escala
    # Primero centramos las coordenadas en el origen
    x_center = df['X'].iloc[0]
    y_center = df['Y'].iloc[0]
    
    # Escalar relativamente al punto inicial
    df['X_scaled'] = (df['X'] - x_center) * scale_factor + x_center
    df['Y_scaled'] = (df['Y'] - y_center) * scale_factor + y_center
    
    return df

def verify_scaling(scaled_df, distance_col='Distance'):
    """
    Verifica que la distancia euclidiana entre puntos consecutivos sea proporcional
    a la distancia real recorrida.
    """
    # Calcular distancias euclidianas entre puntos consecutivos en el espacio X,Y escalado
    xy_scaled_distances = [0]  # Primer punto tiene distancia 0
    for i in range(1, len(scaled_df)):
        xy_dist = euclidean(
            [scaled_df['X_scaled'].iloc[i-1], scaled_df['Y_scaled'].iloc[i-1]],
            [scaled_df['X_scaled'].iloc[i], scaled_df['Y_scaled'].iloc[i]]
        )
        xy_scaled_distances.append(xy_dist)
    
    # Calcular distancia euclidiana acumulada en espacio escalado
    scaled_df['XY_Scaled_CumulativeDist'] = np.cumsum(xy_scaled_distances)
    
    # Comparar distancias
    real_dist_total = scaled_df[distance_col].iloc[-1] - scaled_df[distance_col].iloc[0]
    xy_scaled_dist_total = scaled_df['XY_Scaled_CumulativeDist'].iloc[-1]
    
    print(f"Después de escalar:")
    print(f"Distancia real total: {real_dist_total:.2f} metros")
    print(f"Distancia euclidiana escalada total: {xy_scaled_dist_total:.2f} unidades")
    print(f"Error de escala: {abs(real_dist_total - xy_scaled_dist_total):.2f} metros ({abs(real_dist_total - xy_scaled_dist_total)/real_dist_total*100:.2f}%)")
    
    return scaled_df

'''def compute_radius(x, y):
    """
    Calcula el radio de curvatura a partir de coordenadas x, y (en metros).
    Retorna un array con los radios.
    """
    
    x = np.asarray(x)
    y = np.asarray(y)

    # Derivadas primeras
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Derivadas segundas
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Cálculo del radio
    numerator = (dx**2 + dy**2)**1.5
    denominator = np.abs(dx * ddy - dy * ddx)
    radius = np.where(denominator == 0, np.nan, numerator / denominator)

    return radius'''

def compute_radius(x, y, window_size=5, outlier_threshold=5000):
    """
    Calcula el radio de curvatura a partir de coordenadas x, y muestreadas por metro.
    
    Args:
        x: Coordenadas X en metros
        y: Coordenadas Y en metros
        window_size: Tamaño de la ventana para suavizar derivadas (impar recomendado)
        outlier_threshold: Umbral para considerar un radio como outlier (en metros)
    
    Returns:
        Array con los radios de curvatura en metros
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Asegurar que el tamaño de ventana sea impar
    if window_size % 2 == 0:
        window_size += 1
    
    # Aplicar suavizado a las coordenadas para reducir el ruido
    from scipy.signal import savgol_filter
    if len(x) > window_size:
        x_smooth = savgol_filter(x, window_size, 2)
        y_smooth = savgol_filter(y, window_size, 2)
    else:
        x_smooth = x
        y_smooth = y
    
    # Calcular derivadas con método central más preciso
    # Utilizamos un paso de diferenciación de 2 puntos para aprovechar que la data está por metro
    dx = np.gradient(x_smooth, 1.0)  # El 1.0 indica un paso de 1 metro
    dy = np.gradient(y_smooth, 1.0)
    
    # Derivadas segundas
    ddx = np.gradient(dx, 1.0)
    ddy = np.gradient(dy, 1.0)
    
    # Cálculo del radio usando la fórmula estándar
    numerator = (dx**2 + dy**2)**1.5
    denominator = np.abs(dx * ddy - dy * ddx)
    
    # Evitar divisiones por cero y valores muy pequeños que pueden causar radios extremos
    min_denominator = 1e-8
    denominator = np.maximum(denominator, min_denominator)
    
    # Calcular el radio y limitar valores extremos
    radius = numerator / denominator
    
    # Limitar radios extremadamente grandes (rectas)
    radius = np.where(radius > outlier_threshold, np.inf, radius)
    
    # Tratar los bordes: los primeros y últimos puntos suelen tener valores menos precisos
    edge_points = window_size // 2
    if len(radius) > 2 * edge_points:
        # Extender los valores válidos a los bordes
        radius[:edge_points] = radius[edge_points]
        radius[-edge_points:] = radius[-edge_points-1]
    
    return radius

def smooth_radius_with_savgol(df, radius_col='radio_curvatura', window_size=11, poly_order=3):
    """
    Aplica un filtro Savitzky-Golay para suavizar el radio de curvatura.
    
    Args:
        df: DataFrame con el radio de curvatura
        radius_col: Nombre de la columna con los valores de radio
        window_size: Tamaño de la ventana (debe ser impar)
        poly_order: Orden del polinomio para ajuste (típicamente 3 o 5)
        
    Returns:
        DataFrame con el radio suavizado
    """
    result_df = df.copy()
    
    # Asegurarse de que el tamaño de ventana sea impar
    if window_size % 2 == 0:
        window_size += 1
    
    # Aplicar el filtro SG
    try:
        result_df[f'{radius_col}_smooth'] = savgol_filter(
            result_df[radius_col], 
            window_size, 
            poly_order
        )
    except Exception as e:
        print(f"Error en el filtro Savitzky-Golay: {e}")
        # Si falla, intentar con un método más robusto
        result_df[f'{radius_col}_smooth'] = result_df[radius_col].rolling(
            window=window_size, center=True
        ).median().fillna(result_df[radius_col])
    
    return result_df

def detect_and_correct_radius_outliers(df, radius_col='radio_curvatura', threshold=2.0):
    """
    Detecta y corrige valores atípicos en el radio de curvatura.
    
    Args:
        df: DataFrame con el radio de curvatura
        radius_col: Nombre de la columna con los valores de radio
        threshold: Umbral para considerar un valor como atípico
                   (diferencia relativa con sus vecinos)
        
    Returns:
        DataFrame con valores atípicos corregidos
    """
    result_df = df.copy()
    
    # Calcular la media móvil y la desviación estándar
    rolling_mean = result_df[radius_col].rolling(window=5, center=True).mean()
    rolling_std = result_df[radius_col].rolling(window=5, center=True).std()
    
    # Identificar valores atípicos usando Z-score
    z_scores = np.abs((result_df[radius_col] - rolling_mean) / rolling_std)
    outliers = z_scores > threshold
    
    # Remplazar valores atípicos con la media de sus vecinos
    result_df[f'{radius_col}_corrected'] = result_df[radius_col].copy()
    
    for i in range(1, len(result_df)-1):
        if outliers.iloc[i]:
            # Usar la media de los puntos vecinos no atípicos como valor de reemplazo
            neighbors = [result_df[radius_col].iloc[i-1], result_df[radius_col].iloc[i+1]]
            # Filtrar cualquier vecino que también sea atípico
            valid_neighbors = [n for n in neighbors if n == n]  # filtra NaN
            if valid_neighbors:
                result_df.loc[result_df.index[i], f'{radius_col}_corrected'] = np.mean(valid_neighbors)
    
    return result_df








def merge_on_nearest_time(df_coords, df_telemetry, time_col='Time'):
    """
    Une dos DataFrames buscando el valor más cercano de 'Time' en df_telemetry
    para cada fila en df_coords. Similar a un left join aproximado.
    
    Parámetros:
        df_coords: DataFrame base (pocas filas), debe tener la columna 'Time'.
        df_telemetry: DataFrame con más filas, también con la columna 'Time'.
        time_col: Nombre de la columna de tiempo (por defecto 'Time').
        
    Retorna:
        Un DataFrame combinado con las columnas de ambos, alineados por el tiempo más cercano.
    """
    # Asegurar que ambos estén ordenados por tiempo
    df_coords_sorted = df_coords.sort_values(by=time_col).reset_index(drop=True)
    df_telemetry_sorted = df_telemetry.sort_values(by=time_col).reset_index(drop=True)
    
    # Usamos merge_asof para unir por el tiempo más cercano
    merged_df = pd.merge_asof(
        df_coords_sorted,
        df_telemetry_sorted,
        on=time_col,
        direction='nearest',
        tolerance=None  # puedes limitar si quieres máxima diferencia
    )
    return merged_df

def calcular_aceleracion_lateral(df, speed_col='Speed', radio_col='radio_curvatura'):
    """
    Calcula la aceleración lateral y la añade al DataFrame.

    Parámetros:
        df: DataFrame con columnas de velocidad (km/h) y radio de curvatura (m).
        speed_col: Nombre de la columna de velocidad.
        radio_col: Nombre de la columna de radio de curvatura.

    Retorna:
        El DataFrame con una nueva columna 'aceleracion_lateral' en m/s².
    """
    # Convertir velocidad a m/s
    velocidad_ms = df[speed_col] * (1000 / 3600)
    
    # Evitar división por cero o radios nulos
    with np.errstate(divide='ignore', invalid='ignore'):
        aceleracion_lateral = (velocidad_ms ** 2) / df[radio_col]
        aceleracion_lateral = np.where(df[radio_col] == 0, np.nan, aceleracion_lateral)

    df['aceleracion_lateral'] = aceleracion_lateral/9.81  # Convertir a g (gravedad)
    return df






def detect_filter_lateral_g_outliers(df, lateral_g_col='Lateral_G', window_size=5, z_score_threshold=3.0, 
                                     rate_threshold=2.0, distance_col='Distance', min_valid_g=0.0, max_valid_g=6.0):
    """
    Detecta y filtra outliers en fuerzas G laterales usando múltiples métodos:
    1. Valores absolutos fuera de un rango físicamente plausible
    2. Valores que se desvían significativamente de la tendencia local (z-score)
    3. Cambios bruscos/tasas de cambio improbables entre puntos consecutivos
    
    Args:
        df: DataFrame con datos de telemetría
        lateral_g_col: Nombre de la columna con fuerzas G laterales
        window_size: Tamaño de la ventana deslizante para el análisis local
        z_score_threshold: Umbral para considerar un punto como outlier basado en z-score
        rate_threshold: Umbral para la tasa de cambio máxima permitida entre puntos consecutivos
        distance_col: Nombre de la columna de distancia para normalizar cambios
        min_valid_g: Fuerza G lateral mínima válida (valores absolutos)
        max_valid_g: Fuerza G lateral máxima válida (valores absolutos)
    
    Returns:
        DataFrame: DataFrame original con columnas adicionales:
          - lateral_g_outlier: Boolean, True si es outlier
          - lateral_g_filtered: Fuerzas G laterales filtradas/corregidas
          - outlier_reason: Razón por la que un punto fue marcado como outlier
    """
    # Trabajar con una copia para no modificar el original
    result_df = df.copy()
    
    # 1. Detección por valores absolutos fuera de rango
    absolute_outliers = (abs(result_df[lateral_g_col]) > max_valid_g) | (abs(result_df[lateral_g_col]) < min_valid_g)
    
    # 2. Detección por z-score (desviación respecto a la media local)
    result_df['rolling_mean'] = result_df[lateral_g_col].rolling(window=window_size, center=True).mean()
    result_df['rolling_std'] = result_df[lateral_g_col].rolling(window=window_size, center=True).std()
    # Manejar STD=0 para evitar divisiones por cero
    result_df['rolling_std'] = result_df['rolling_std'].replace(0, result_df['rolling_std'].mean())
    result_df['z_score'] = abs(result_df[lateral_g_col] - result_df['rolling_mean']) / result_df['rolling_std']
    z_score_outliers = result_df['z_score'] > z_score_threshold
    
    # 3. Detección por tasa de cambio improbable
    # Calcular diferencia en G lateral y en distancia
    result_df['g_diff'] = result_df[lateral_g_col].diff().abs()
    result_df['distance_diff'] = result_df[distance_col].diff().abs()
    
    # Evitar divisiones por cero
    result_df['distance_diff'] = result_df['distance_diff'].replace(0, 0.1)
    
    # Calcular tasa de cambio (cambio de G por metro)
    result_df['change_rate'] = result_df['g_diff'] / result_df['distance_diff']
    rate_outliers = result_df['change_rate'] > rate_threshold
    
    # 4. Combinación de detecciones
    result_df['lateral_g_outlier'] = absolute_outliers | z_score_outliers | rate_outliers
    
    # Determinar la razón de clasificación como outlier
    result_df['outlier_reason'] = ""
    result_df.loc[absolute_outliers, 'outlier_reason'] += "Valor absoluto fuera de rango;"
    result_df.loc[z_score_outliers, 'outlier_reason'] += "Desviación estadística significativa;"
    result_df.loc[rate_outliers, 'outlier_reason'] += "Cambio brusco improbable;"
    
    # 5. Corrección de outliers: reemplazar con interpolación
    # Primero creamos una copia de los valores originales
    result_df['lateral_g_filtered'] = result_df[lateral_g_col].copy()
    
    # Índices de outliers
    outlier_indices = result_df[result_df['lateral_g_outlier']].index
    
    if len(outlier_indices) > 0:
        # Valores buenos para la interpolación
        good_indices = result_df[~result_df['lateral_g_outlier']].index
        good_values = result_df.loc[good_indices, lateral_g_col]
        
        # Solo interpolar si hay suficientes valores buenos
        if len(good_indices) > 2:
            # Interpolar
            interpolated = np.interp(outlier_indices, good_indices, good_values)
            result_df.loc[outlier_indices, 'lateral_g_filtered'] = interpolated
    
    # 6. Suavizado adicional opcional para los valores filtrados
    result_df['lateral_g_filtered'] = result_df['lateral_g_filtered'].rolling(
        window=3, center=True, min_periods=1).mean()
    
    # Estadísticas sobre outliers
    outlier_percent = result_df['lateral_g_outlier'].mean() * 100
    print(f"Se detectaron {len(outlier_indices)} outliers ({outlier_percent:.2f}% de los datos)")
    
    # Eliminar columnas temporales de cálculo
    cols_to_drop = ['rolling_mean', 'rolling_std', 'z_score', 'g_diff', 'distance_diff', 'change_rate']
    result_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    return result_df

def fit_circle(x, y):
    """Ajusta un círculo a un conjunto de puntos (x, y) y devuelve el centro y radio."""
    # Función de error: distancia al círculo
    def circle_residuals(params, x, y):
        h, k, R = params
        return (x - h)**2 + (y - k)**2 - R**2
    
    # Estimación inicial: centro en el promedio de los puntos, radio promedio
    h0 = np.mean(x)
    k0 = np.mean(y)
    R0 = np.mean(np.sqrt((x - h0)**2 + (y - k0)**2))
    initial_guess = [h0, k0, R0]
    
    # Ajuste por mínimos cuadrados
    result = least_squares(circle_residuals, initial_guess, args=(x, y))
    h, k, R = result.x
    
    return h, k, abs(R)  # Aseguramos que el radio sea positivo

def calculate_radius_window(x_coords, y_coords):
    """
    Calcula el radio de curvatura y la dirección del giro para una ventana de puntos.
    
    Returns:
        tuple: (radius, direction) donde:
               - radius: radio de curvatura en metros (positivo)
               - direction: 1 para giro a la derecha, -1 para giro a la izquierda, 0 para línea recta
    """
    if len(x_coords) < 3:  # Necesitamos al menos 3 puntos
        return np.inf, 0
    
    # Ajustar un círculo a los puntos
    h, k, R = fit_circle(x_coords, y_coords)
    
    # Determinar dirección del giro basado en la posición del centro del círculo
    # en relación a la trayectoria
    
    # Tomar el punto medio de la trayectoria como referencia
    mid_idx = len(x_coords) // 2
    x_mid = x_coords[mid_idx]
    y_mid = y_coords[mid_idx]
    
    # Calcular vectores para determinar dirección
    # 1. Vector desde el punto inicial al punto medio
    if mid_idx > 0:
        vec_path = [x_mid - x_coords[0], y_mid - y_coords[0]]
    else:
        vec_path = [x_coords[-1] - x_coords[0], y_coords[-1] - y_coords[0]]
    
    # 2. Vector desde el punto medio al centro del círculo
    vec_center = [h - x_mid, k - y_mid]
    
    # Calcular el producto vectorial para determinar si el centro está 
    # a la izquierda o derecha de la trayectoria
    cross_product = vec_path[0] * vec_center[1] - vec_path[1] * vec_center[0]
    
    # Determinar dirección basada en el producto vectorial
    if abs(cross_product) < 1e-10 or R > 5000:  # Umbral para considerar línea recta
        direction = 0  # Recta
    elif cross_product > 0:
        direction = -1  # Giro a la izquierda
    else:
        direction = 1  # Giro a la derecha
        
    return abs(R), direction

def calculate_lateral_g(df, window_size=10):
    """Calcula las fuerzas g laterales y la dirección de giro usando una ventana de puntos."""
    g = 9.81  # Aceleración gravitacional en m/s²
    radii = []
    directions = []
    g_forces = []
    half_window = window_size // 2
    
    # Iterar sobre el DataFrame
    for i in range(len(df)):
        # Definir la ventana: tomar puntos antes y después del índice actual
        start_idx = max(0, i - half_window)
        end_idx = min(len(df), i + half_window + 1)
        
        # Extraer coordenadas de la ventana
        x_window = df['X'].iloc[start_idx:end_idx].values
        y_window = df['Y'].iloc[start_idx:end_idx].values
        
        # Calcular el radio de curvatura y la dirección del giro
        R, direction = calculate_radius_window(x_window, y_window)
        radii.append(R)
        directions.append(direction)
        
        # Convertir velocidad de km/h a m/s
        v = df['Speed'].iloc[i] / 3.6
        
        # Calcular aceleración lateral en g
        if R == np.inf or R == 0:  # Evitar división por cero o infinito
            g_force = 0
        else:
            g_force = (v**2 / R) / g
            
        # Aplicar signo según la dirección del giro
        g_force = g_force * direction
        g_forces.append(g_force)
    
    return radii, directions, g_forces

def interpolate_xy(df, interpolation_factor=5, smoothing=1):
    """
    Interpola las coordenadas X e Y para crear curvas más suaves en la representación del mapa,
    aumentando significativamente la densidad de puntos.
    
    Args:
        df: DataFrame con columnas 'X', 'Y', 'Distance' y otras columnas para interpolar
        interpolation_factor: Factor de multiplicación para la cantidad de puntos (3 significa 3 veces más puntos)
        smoothing: Parámetro de suavizado para la interpolación spline (0-1, donde mayor valor = más suavizado)
    
    Returns:
        DataFrame interpolado con curvas más suaves
    """
    print(f"Interpolando datos con factor {interpolation_factor}x...")
    
    # Verificar que el DataFrame contiene las columnas necesarias
    required_columns = ['X', 'Y', 'Distance']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"El DataFrame debe contener la columna '{col}'")
    
    # Obtener el rango de distancia
    min_distance = df['Distance'].min()
    max_distance = df['Distance'].max()
    print(f"Rango de distancia original: {min_distance:.2f} a {max_distance:.2f} metros")
    
    # Número de puntos en el dataset original
    num_points_original = len(df)
    
    # Preparar arrays de coordenadas X e Y ordenados por distancia
    df_sorted = df.sort_values('Distance').reset_index(drop=True)
    x = df_sorted['X'].values
    y = df_sorted['Y'].values
    distances = df_sorted['Distance'].values
    
    # Método 1: Utilizar splines paramétricos para coordenadas X,Y (mejor para circuitos)
    # Este método crea una curva paramétrica más suave
    try:
        # Crear una parametrización basada en la longitud acumulada a lo largo de la curva
        tck, u = splprep([x, y], s=smoothing, k=3, per=False)
        
        # Crear un nuevo conjunto de puntos evaluando el spline en una malla más densa
        num_points_new = num_points_original * interpolation_factor
        u_fine = np.linspace(0, 1, num_points_new)
        x_fine, y_fine = splev(u_fine, tck)
        
        # Crear una nueva matriz de distancia interpolada
        # La distancia sigue siendo proporcional pero ahora con más puntos
        distances_fine = np.linspace(min_distance, max_distance, num_points_new)
        
        # Crear un nuevo DataFrame con los puntos interpolados
        df_interpolated = pd.DataFrame({
            'X': x_fine,
            'Y': y_fine,
            'Distance': distances_fine
        })
        
        print(f"Interpolación paramétrica completada: {len(df_interpolated)} puntos generados")
    except Exception as e:
        print(f"Error en la interpolación paramétrica: {e}")
        print("Aplicando método alternativo de interpolación...")
        
        # Método 2 (alternativo): Interpolar X e Y con respecto a la distancia
        # Crear una nueva grilla de distancia con más puntos
        distances_fine = np.linspace(min_distance, max_distance, num_points_original * interpolation_factor)
        
        # Interpolación para X e Y con respecto a la distancia
        interp_x = interp1d(distances, x, kind='cubic', bounds_error=False, fill_value="extrapolate")
        interp_y = interp1d(distances, y, kind='cubic', bounds_error=False, fill_value="extrapolate")
        
        # Evaluar los interpoladores en la nueva grilla de distancia
        x_fine = interp_x(distances_fine)
        y_fine = interp_y(distances_fine)
        
        # Crear el DataFrame interpolado
        df_interpolated = pd.DataFrame({
            'X': x_fine,
            'Y': y_fine,
            'Distance': distances_fine
        })
        
        print(f"Interpolación alternativa completada: {len(df_interpolated)} puntos generados")
    
    # Para el resto de las columnas que sean numéricas, interpolarlas respecto a la distancia
    numeric_columns = [col for col in df.columns if col not in ['X', 'Y', 'Distance', 'Source'] 
                       and pd.api.types.is_numeric_dtype(df[col])]
    
    for column in numeric_columns:
        try:
            values = df_sorted[column].values
            interp_func = interp1d(distances, values, kind='linear', bounds_error=False, fill_value="extrapolate")
            df_interpolated[column] = interp_func(distances_fine)
        except Exception as e:
            print(f"No se pudo interpolar la columna {column}: {e}")
    
    # Para columnas no numéricas (excepto 'Source'), usar el método del vecino más cercano
    non_numeric_columns = [col for col in df.columns if col not in ['X', 'Y', 'Distance', 'Source'] + numeric_columns]
    
    for column in non_numeric_columns:
        try:
            # Encontrar el índice del punto más cercano en el dataset original para cada punto nuevo
            nearest_indices = [np.abs(distances - d).argmin() for d in distances_fine]
            df_interpolated[column] = df_sorted[column].iloc[nearest_indices].values
        except Exception as e:
            print(f"No se pudo asignar la columna {column}: {e}")
    
    # Para la columna 'Source', establecer NaN para puntos interpolados
    if 'Source' in df.columns:
        # Identificar qué distancias eran del dataset original
        original_distances = set(distances)
        # Crear una máscara que identifica puntos originales vs interpolados
        is_original_point = np.array([d in original_distances for d in distances_fine])
        
        # Asignar valores de Source: NaN para puntos interpolados, valor original para puntos originales
        df_interpolated['Source'] = np.nan  # Inicialmente todos NaN
        
        # Para los puntos originales, encontrar y asignar el valor correspondiente
        for i, is_original in enumerate(is_original_point):
            print()
            if is_original:
                # Encontrar el índice en el dataset original
                orig_idx = np.where(distances == distances_fine[i])[0][0]
                df_interpolated.loc[i, 'Source'] = 'pos'
    
    # Aplicar suavizado adicional a las coordenadas para eliminar cualquier pico restante
    window_size = max(3, len(df_interpolated) // 100)  # Ventana adaptativa
    df_interpolated['X'] = df_interpolated['X'].rolling(window=window_size, center=True, min_periods=1).mean()
    df_interpolated['Y'] = df_interpolated['Y'].rolling(window=window_size, center=True, min_periods=1).mean()
    
    # Verificar y eliminar cualquier valor NaN resultante
    df_interpolated = df_interpolated.dropna(subset=['X', 'Y']).reset_index(drop=True)
    
    print(f"Datos originales: {num_points_original} puntos")
    print(f"Datos interpolados: {len(df_interpolated)} puntos")
    print(f"Factor de aumento: {len(df_interpolated)/num_points_original:.2f}x")
    
    return df_interpolated

def calculate_longitudinal_g(df):
    # Convertir velocidad de km/h a m/s
    df['speed_ms'] = df['Speed'] * (5 / 18)

    # Calcular diferencias (Delta v y Delta t)
    df['delta_v'] = df['speed_ms'].diff()  # Diferencia de velocidad
    df['delta_t'] = df['Time'].diff()      # Diferencia de tiempo

    # Calcular aceleración en m/s²
    df['acceleration_ms2'] = df['delta_v'] / df['delta_t']

    # Convertir aceleración a fuerzas g (1 g = 9.81 m/s²)
    df['longitudinal_aceleration'] = df['acceleration_ms2'] / 9.81

    return df

def visualize_track_telemetry(df_resultado):
    """
    Crea dos figuras interactivas para visualizar el mapa de la pista con telemetría.
    
    Args:
        df_resultado: DataFrame con columnas X_scaled, Y_scaled, Speed, radio_final y aceleracion_lateral
        
    Returns:
        tuple: (fig_track_speed, fig_track_g) - Dos figuras de Plotly que muestran diferentes aspectos de telemetría
    """
    # Verificar que el DataFrame contiene las columnas necesarias
    required_columns = ['X_scaled', 'Y_scaled', 'Speed', 'radio_final', 'aceleracion_lateral', 'Distance']
    for col in required_columns:
        if col not in df_resultado.columns:
            raise ValueError(f"El DataFrame debe contener la columna '{col}'")
    
    # Figura 1: Mapa de pista con velocidad y radio de curva
    fig_track_speed = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scatter"}]]
    )
    
    # Añadir el trazado de la pista, coloreado por velocidad
    fig_track_speed.add_trace(
    go.Scatter(
        x=df_resultado['X_scaled'],
        y=df_resultado['Y_scaled'],
        mode='lines',
        line=dict(
            width=5,
            # Remove the color parameter from here
        ),
        marker=dict(
            color=df_resultado['Speed'],  # Move the color mapping to marker
            colorscale='Viridis',
            colorbar=dict(
                title="Velocidad (km/h)",
                thickness=15,
                len=0.9,
                y=0.5,
                yanchor='middle'
            ),
            line=dict(width=0)
        ),
        hovertemplate=(
            '<b>Distancia:</b> %{customdata[0]:.0f}m<br>' +
            '<b>Velocidad:</b> %{customdata[1]:.1f} km/h<br>' +
            '<b>Radio de curva:</b> %{customdata[2]:.1f}m<br>' +
            '<extra></extra>'
        ),
        name='Velocidad',
        customdata=np.stack((
            df_resultado['Distance'],
            df_resultado['Speed'],
            df_resultado['radio_final']
        ), axis=1)
    )
)
    
    # Añadir puntos para indicar radio de curva (más pequeño = curva más cerrada)
    # Normalizar el radio para el tamaño de los puntos (radio inverso, curvas cerradas = puntos grandes)
    max_radius = df_resultado['radio_final'].max()
    min_radius = df_resultado['radio_final'].min()
    normalized_size = 15 - 10 * ((df_resultado['radio_final'] - min_radius) / (max_radius - min_radius))
    
    # Solo mostrar puntos donde hay curvas significativas (radio pequeño)
    threshold = np.percentile(df_resultado['radio_final'], 30)  # Mostrar puntos para el 30% de curvas más cerradas
    curve_idx = df_resultado['radio_final'] <= threshold
    
    if curve_idx.any():
        fig_track_speed.add_trace(
            go.Scatter(
                x=df_resultado.loc[curve_idx, 'X_scaled'],
                y=df_resultado.loc[curve_idx, 'Y_scaled'],
                mode='markers',
                marker=dict(
                    size=normalized_size[curve_idx],
                    color='red',
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                hoverinfo='skip',
                showlegend=False
            )
        )
    
    # Configurar el diseño para la primera figura
    fig_track_speed.update_layout(
        title='Mapa de Pista: Velocidad y Radio de Curva',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=''
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title='',
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgb(240, 240, 240)',
        height=700,
        width=900,
        margin=dict(l=50, r=50, b=50, t=50),
    )
    
    # Añadir anotación para la leyenda de radio de curva
    fig_track_speed.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="Radio de curva: 🔴 = curva cerrada",
        showarrow=False,
        font=dict(
            size=12,
            color="black"
        ),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Añadir flecha para indicar dirección
    # Encontrar un punto aproximadamente en el primer tercio de la pista
    idx = int(len(df_resultado) * 0.3)
    if idx < len(df_resultado) - 1:
        x0, y0 = df_resultado['X_scaled'].iloc[idx], df_resultado['Y_scaled'].iloc[idx]
        x1, y1 = df_resultado['X_scaled'].iloc[idx+1], df_resultado['Y_scaled'].iloc[idx+1]
        
        # Añadir flecha para indicar dirección
        fig_track_speed.add_annotation(
            x=x0,
            y=y0,
            ax=x1,
            ay=y1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor="black"
        )
    
    # Figura 2: Mapa de pista con fuerzas G laterales
    fig_track_g = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scatter"}]]
    )
    
    # Definir la escala de colores para las fuerzas G
    # Rojo para giros a la derecha (valores positivos), azul para giros a la izquierda (valores negativos)
    max_g_abs = max(abs(df_resultado['aceleracion_lateral'].max()), abs(df_resultado['aceleracion_lateral'].min()))
    
    # Crear una escala de colores personalizada
    colorscale = [
        [0, 'blue'],
        [0.45, 'lightblue'],
        [0.5, 'white'],
        [0.55, 'pink'],
        [1, 'red']
    ]
    
    # Añadir el trazado de la pista, coloreado por fuerza G lateral
    fig_track_g.add_trace(
    go.Scatter(
        x=df_resultado['X_scaled'],
        y=df_resultado['Y_scaled'],
        mode='lines',
        line=dict(
            width=5,
            # Remove color from here
        ),
        marker=dict(
            color=df_resultado['aceleracion_lateral'],  # Move to marker
            colorscale=colorscale,
            cmin=-max_g_abs,
            cmax=max_g_abs,
            colorbar=dict(
                title="Fuerza G Lateral",
                thickness=15,
                len=0.9,
                y=0.5,
                yanchor='middle',
                tickvals=[-max_g_abs, 0, max_g_abs],
                ticktext=["Izquierda (-)", "0", "Derecha (+)"]
            ),
            line=dict(width=0)
        ),
        hovertemplate=(
            '<b>Distancia:</b> %{customdata[0]:.0f}m<br>' +
            '<b>Fuerza G Lateral:</b> %{customdata[1]:.2f}g<br>' +
            '<b>Velocidad:</b> %{customdata[2]:.1f} km/h<br>' +
            '<extra></extra>'
        ),
        name='Fuerza G Lateral',
        customdata=np.stack((
            df_resultado['Distance'],
            df_resultado['aceleracion_lateral'],
            df_resultado['Speed']
        ), axis=1)
    )
)
    
    # Marcar puntos donde hay grandes fuerzas G (como zonas destacadas de frenada o giro)
    g_threshold = np.percentile(abs(df_resultado['aceleracion_lateral']), 90)  # Top 10% de fuerzas G
    high_g_idx = abs(df_resultado['aceleracion_lateral']) >= g_threshold
    
    if high_g_idx.any():
        # Determinar colores basados en el signo de aceleración lateral
        high_g_colors = np.where(
            df_resultado.loc[high_g_idx, 'aceleracion_lateral'] > 0,
            'red',   # Positivo: giro a la derecha
            'blue'   # Negativo: giro a la izquierda
        )
        
        fig_track_g.add_trace(
            go.Scatter(
                x=df_resultado.loc[high_g_idx, 'X_scaled'],
                y=df_resultado.loc[high_g_idx, 'Y_scaled'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=high_g_colors,
                    symbol='circle',
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hoverinfo='skip',
                showlegend=False
            )
        )
    
    # Configurar el diseño para la segunda figura
    fig_track_g.update_layout(
        title='Mapa de Pista: Fuerzas G Laterales',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=''
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title='',
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgb(240, 240, 240)',
        height=700,
        width=900,
        margin=dict(l=50, r=50, b=50, t=50),
    )
    
    # Añadir leyenda para interpretación de colores
    fig_track_g.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="🔴 = Giro a la derecha, 🔵 = Giro a la izquierda",
        showarrow=False,
        font=dict(
            size=12,
            color="black"
        ),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Flecha para indicar dirección (misma que en la primera figura)
    if idx < len(df_resultado) - 1:
        fig_track_g.add_annotation(
            x=x0,
            y=y0,
            ax=x1,
            ay=y1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor="black"
        )
    
    return fig_track_speed, fig_track_g