import numpy as np
import matplotlib.pyplot as plt 


# -------------------------------------------
# FUNCIONES TRANSFORMADA DE HOUGH PARA RECTAS
# - generar_acumulador_T_H_Rectas
# - obtener_rectas_mas_votadas
# -------------------------------------------

def generar_acumulador_T_H_Rectas(bw, rho_resolution=1, theta_resolution_grados=1):
    """
    Implementación de la Transformada de Hough estándar.

    Parámetros:
    ------------
    bw : ndarray (2D)
        Imagen binaria (True/False o 0/1).
    rho_resolution : float, opcional
        Espaciado entre bins de la distancia rho (en píxeles).
    theta_resolution_grados : float, opcional
        Espaciado angular en grados entre bins de theta.

    Retorna:
    ---------
    H : ndarray
        Matriz acumuladora de Hough (NRHO × NTHETA).
    theta : ndarray
        Valores de ángulos (en radianes, entre 0 y π).
    rho : ndarray
        Valores de distancias (en píxeles, entre -ρ_max y +ρ_max).
    """

    # Asegurar la matriz de entrada es booleana
    if bw.dtype != bool:
        bw = bw != 0
    if bw.ndim != 2:
        raise ValueError("La matriz binaria de entrada debe ser 2D")

    M, N = bw.shape

    # Definir el rango de theta [0, π)
    theta_resolution_radianes = np.deg2rad(theta_resolution_grados)
    theta = np.arange(0, np.pi, theta_resolution_radianes)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Calcular valores posibles de rho (centrado estilo MATLAB)
    D = np.sqrt((M - 1)**2 + (N - 1)**2)
    q = np.ceil(D / rho_resolution)
    rho_max = q * rho_resolution
    rho = np.linspace(-rho_max, rho_max, int(2 * q + 1))
    nrho = len(rho)

    # Crear acumulador
    H = np.zeros((nrho, len(theta)), dtype=np.uint64)

    # Obtener píxeles de borde (no nulos)
    y_idxs, x_idxs = np.nonzero(bw)

    # Acumular votos
    for (x, y) in zip(x_idxs, y_idxs):
        for t_idx in range(len(theta)):
            rho_val = x * cos_t[t_idx] + y * sin_t[t_idx]
            rho_idx = int(np.round((rho_val + rho_max) / rho_resolution))
            if 0 <= rho_idx < nrho:
                H[rho_idx, t_idx] += 1

    return H, np.rad2deg(theta), rho


def obtener_rectas_mas_votadas(H, num_peaks=5, threshold=None, nhood_size=9):
    """
    Implementación de houghpeaks 

    Parámetros
    ----------
    H : ndarray
        Matriz acumuladora de Hough (rho × theta)
    num_peaks : int
        Número máximo de picos a detectar.
    threshold : float
        Valor mínimo de votos para considerar un pico.
    nhood_size : int
        Tamaño de la vecindad (en píxeles) para suprimir alrededor de un máximo (debe ser impar).

    Retorna
    -------
    peaks : ndarray (N×2)
        Coordenadas (fila, columna) de los picos detectados.
    """

    if threshold is None:
        threshold = 0.5 * np.max(H)

    # Copiamos H para no modificar el original
    H_copy = H.copy()
    peaks = []

    # Definir tamaño de vecindad
    nhood_half = nhood_size // 2

    for _ in range(num_peaks):
        # Buscar máximo actual
        max_val = np.max(H_copy)
        if max_val < threshold:
            break

        # Índices del máximo
        # Buscar el valor máximo en la matriz
        indice_max_lineal = np.argmax(H_copy)
        # Convertir el índice lineal en coordenadas (fila, columna)
        fila, columna = np.unravel_index(indice_max_lineal, H_copy.shape)
        # Equivalente, pero más eficiente que: fila, columna = np.argwhere(H == H.max())[0]

        peaks.append((fila, columna))

        # Suprimir vecindad alrededor del máximo
        fila_min = max(0, fila - nhood_half)
        fila_max = min(H.shape[0], fila + nhood_half + 1)
        col_min = max(0, columna - nhood_half)
        col_max = min(H.shape[1], columna + nhood_half + 1)
        H_copy[fila_min:fila_max, col_min:col_max] = 0

    return np.array(peaks)

# ----------------------------------------
# FUNCIONES TRANSFORMADA CIRCULAR DE HOUGH
# - circlepoints
# - generar_acumulador_T_H_Circulos
# - obtener_circulos_mas_votados
# - mostrar_circulos_Hough
# - crear_mascara_circulo 
# ----------------------------------------

def circlepoints(r):
    """
    Devuelve coordenadas (x, y) de puntos enteros cercanos a una circunferencia de radio r.

    """
    l = round(r / np.sqrt(2))
    if round(np.sqrt(r ** 2 - l ** 2)) < l:  # cruza la diagonal
        l -= 1

    x0 = np.arange(0, l + 1)
    y0 = np.round(np.sqrt(r ** 2 - x0 ** 2)).astype(int)

    if y0[-1] == l:
        l2 = l
    else:
        l2 = l + 1

    # primer cuadrante
    x = np.concatenate([x0, y0[l2 - 1:1:-1]])
    y = np.concatenate([y0, x0[l2 - 1:1:-1]])

    # segundo cuadrante
    x0 = np.concatenate([x, y])
    y0 = np.concatenate([y, -x])

    # círculo completo
    x = np.concatenate([x0, -x0])
    y = np.concatenate([y0, -y0])

    return x.astype(int), y.astype(int)



def generar_acumulador_T_H_Circulos(b, rrange, normalise=False):

    """
    Implementación de la Transformada Circular de Hough en Python

    Parámetros:
    -----------
    b : np.ndarray
        Imagen binaria 2D (0 y 1).
    rrange : iterable
        Lista o array de radios a evaluar.
    normalise : bool
        Si True, normaliza cada plano del acumulador dividiendo por el radio.

    Retorna:
    --------
    h : np.ndarray
        Acumulador 3D (filas, columnas, radios).
        Cada capa h[:,:,k] tiene el mismo tamaño que la imagen original.
    """

    # Coordenadas de los píxeles no nulos (características)
    featR, featC = np.nonzero(b)
    nr, nc = b.shape
    nradii = len(rrange)
    margin = int(np.ceil(max(rrange)))

    # Tamaño extendido del acumulador (con márgenes)
    nrh = nr + 2 * margin
    nch = nc + 2 * margin
    h = np.zeros((nrh, nch, nradii), dtype=np.uint32)

    # Precalcular los offsets de los círculos
    tempR, tempC, tempRad = [], [], []
    for i, r in enumerate(rrange):
        tR, tC = circlepoints(r)
        tempR.extend(tR)
        tempC.extend(tC)
        tempRad.extend([i] * len(tR))

    tempR = np.array(tempR, dtype=np.int32)
    tempC = np.array(tempC, dtype=np.int32)
    tempRad = np.array(tempRad, dtype=np.int32)

    # Sumar votos en el acumulador
    for fr, fc in zip(featR, featC):
        rr = tempR + fr + margin
        cc = tempC + fc + margin
        valid = (rr >= 0) & (rr < nrh) & (cc >= 0) & (cc < nch)
        rr, cc, rad = rr[valid], cc[valid], tempRad[valid]
        h[rr, cc, rad] += 1

    # Recortar para mantener solo los centros dentro de la imagen
    h = h[margin:-margin, margin:-margin, :]

    # Normalización opcional
    if normalise:
        for k, r in enumerate(rrange):
            h[:, :, k] = h[:, :, k] / r

    return h.astype(float)


def obtener_circulos_mas_votados(h, rrange, num_peaks=1, threshold=None, nhood_size=15):
    """
    Implementacion de circle_houghpeaks: detecta los picos (centros probables de círculos) en el acumulador de la Transformada de Hough.

    Parámetros:
    -----------
    h : np.ndarray
        Acumulador 3D devuelto por circle_hough.
    rrange : iterable
        Radios usados en la transformada.
    num_peaks : int
        Número máximo de picos a detectar.
    threshold : float o None
        Valor mínimo para considerar un pico. Si None, se usa el 50% del máximo global.
    nhood_size : int
        Tamaño del vecindario (en píxeles) para suprimir picos cercanos.

    Retorna:
    --------
    peaks : list of (row, col, radius)
        Lista de coordenadas de los centros detectados y sus radios.
    """

    h_copy = h.copy()
    peaks = []

    if threshold is None:
        threshold = 0.5 * np.max(h_copy)

    for _ in range(num_peaks):
        # Buscar el máximo global
        idx = np.argmax(h_copy)
        val = h_copy.flat[idx]
        if val < threshold:
            break

        # Convertir índice lineal a coordenadas
        r, c, k = np.unravel_index(idx, h_copy.shape)
        peaks.append((r, c, rrange[k]))

        # Suprimir vecindario alrededor del pico detectado
        r_min = max(r - nhood_size, 0)
        r_max = min(r + nhood_size, h_copy.shape[0])
        c_min = max(c - nhood_size, 0)
        c_max = min(c + nhood_size, h_copy.shape[1])
        k_min = max(k - 1, 0)
        k_max = min(k + 1, h_copy.shape[2])

        h_copy[r_min:r_max, c_min:c_max, k_min:k_max] = 0

    return peaks


def mostrar_circulos_Hough(image, peaks, color='r'):
    """
    Dibuja los círculos detectados sobre image.
    
    Parámetros:
    -----------
    image : np.ndarray
        Imagen original (2D).
    peaks : list of (row, col, radius)
        Lista de círculos detectados (salida de obtener_circulos_mas_votados).
    color : str
        Color de los círculos.
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for (r, c, rad) in peaks:
        circ = plt.Circle((c, r), rad, color=color, fill=False, linewidth=2)
        ax.add_patch(circ)
    ax.set_title(f'{len(peaks)} círculos detectados')
    plt.show()


def crear_mascara_circulo(shape, centroX, centroY, radio):
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    # el +0.5 ayuda a incluir píxeles limítrofes, similar a redondeo de circunferencia
    return (xx - centroX)**2 + (yy - centroY)**2 <= (radio + 0.5)**2