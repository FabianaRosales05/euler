# Método de Euler para integración numérica
# Resuelve la EDO: dy/dx = f(x) con y(x0) = y0 para aproximar
# la integral definida de f(x) = ax³ + bx² + cx + d en [x0, xn].

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configuración global de estilos, colores y fuentes

# Paleta de colores
ROSE = '#e75480'
ROSE_HOVER = '#d6336c'
ROSE_LIGHT = '#ffb3c6'
TEXT_DARK = '#1a1a2e'
BG_SOFT = '#f5f0f2'
WHITE = '#ffffff'
BORDER = '#e8e0e4'
GRAY_TEXT = '#8e8e93'
SUCCESS = '#27ae60'
ERROR = '#e74c3c'
CHART_LINE = '#2c3e50'

# Fonts
FONT_TITLE = ('Segoe UI', 15, 'bold')
FONT_SUBTITLE = ('Segoe UI', 11, 'bold')
FONT_NORMAL = ('Segoe UI', 10)
FONT_SMALL = ('Segoe UI', 9)
FONT_MONO = ('Consolas', 11, 'bold')

# Window dimensions
WIN_WIDTH = 1100
WIN_HEIGHT = 750
WIN_MIN_WIDTH = 1000
WIN_MIN_HEIGHT = 700

# Funciones matemáticas
def construir_polinomio(a, b, c, d):
    # Construye la expresión simbólica f(x) = ax³ + bx² + cx + d.
    x = sp.Symbol('x')
    polinomio = a * x**3 + b * x**2 + c * x + d
    return polinomio, x


def metodo_euler(f, x0, xn, h, y0):
    # Aproxima la integral definida resolviendo dy/dx = f(x).
    # Fórmula: y_{i+1} = y_i + h · f(x_i)
    
    n = int(round((xn - x0) / h))
    x_vals, y_vals = [], []
    x_actual, y_actual = x0, y0
    x_vals.append(x_actual)
    y_vals.append(y_actual)

    for _ in range(n):
        pendiente = f(x_actual)
        y_actual += h * pendiente
        x_actual = round(x_actual + h, 12)
        x_vals.append(x_actual)
        y_vals.append(y_actual)

    return x_vals, y_vals, y_vals[-1]


def integral_exacta(polinomio, x, x0, xn):
    # Calcula la integral analítica exacta con sympy.
    antiderivada = sp.integrate(polinomio, x)
    exacta = float(antiderivada.subs(x, xn) - antiderivada.subs(x, x0))
    return exacta, antiderivada


def calcular_error(aproximada, exacta):
    # Error porcentual absoluto: |(aprox - exacta) / exacta| × 100
    if abs(exacta) < 1e-15:
        return 0.0 if abs(aproximada) < 1e-15 else float('inf')
    return abs((aproximada - exacta) / exacta) * 100


def generar_tabla(x_vals, y_vals, f):
    # Prepara los datos iterativos para mostrar en la tabla.
    tabla = []
    for i in range(len(x_vals)):
        fxi = f(x_vals[i]) if i < len(x_vals) - 1 else None
        if i < len(x_vals) - 1:
            yi1 = y_vals[i] + f(x_vals[i]) * (x_vals[i + 1] - x_vals[i])
        else:
            yi1 = None
        tabla.append({'i': i, 'x_i': x_vals[i], 'f(x_i)': fxi,
                      'y_i': y_vals[i], 'y_{i+1}': yi1})
    return tabla

# Funciones de interfaz y gráficos

def _limpiar_frame(frame):
    # Elimina todos los widgets dentro de un frame.
    for widget in frame.winfo_children():
        widget.destroy()


def _incrustar_figura(fig, frame):
    # Incrusta una figura de matplotlib en un frame de tkinter.
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Funciones para graficar la solución analítica y la aproximación numérica

def graficar_analitica(frame, polinomio_str, x0, xn, a, b, c, d, resultado_exacto):
    # Grafica la curva del polinomio con el área bajo la curva sombreada.
    _limpiar_frame(frame)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG_SOFT)
    ax.set_facecolor(WHITE)

    margen = max(0.5, (xn - x0) * 0.15)
    xs = np.linspace(x0 - margen, xn + margen, 2000)
    f = lambda x: a * x**3 + b * x**2 + c * x + d
    ys = f(xs)

    ax.plot(xs, ys, color=CHART_LINE, linewidth=2.5, label=f'f(x) = {polinomio_str}')

    x_fill = np.linspace(x0, xn, 500)
    y_fill = f(x_fill)
    ax.fill_between(x_fill, y_fill, alpha=0.35, color=ROSE_LIGHT,
                    label=f'Área = {resultado_exacto:.6f}')

    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.6)
    ax.axvline(x=x0, color=SUCCESS, linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'x₀ = {x0}')
    ax.axvline(x=xn, color=ERROR, linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'xₙ = {xn}')

    ax.set_title('Integración Analítica — Curva y Área', fontsize=13,
                 fontweight='bold', pad=12, color=CHART_LINE)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('f(x)', fontsize=11)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, edgecolor=BORDER)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _incrustar_figura(fig, frame)


def graficar_numerica(frame, x_vals, y_vals, x0, xn, a, b, c, d,
                      polinomio_str, resultado_exacto):
    # Grafica la comparación entre la solución exacta y Euler.
    _limpiar_frame(frame)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(BG_SOFT)
    ax.set_facecolor(WHITE)

    margen = max(0.3, (xn - x0) * 0.1)
    xs = np.linspace(x0 - margen, xn + margen, 2000)

    def F_exacta(x):
        return a * x**4 / 4 + b * x**3 / 3 + c * x**2 / 2 + d * x

    F_x0 = F_exacta(x0)
    ys_exact = F_exacta(xs) - F_x0 + y_vals[0]

    ax.plot(xs, ys_exact, color=ROSE, linewidth=2.2, alpha=0.8,
            label='Solución exacta')
    ax.plot(x_vals, y_vals, color=CHART_LINE, linewidth=1.8, alpha=0.9,
            label='Euler', zorder=3)
    ax.scatter(x_vals, y_vals, color=CHART_LINE, s=40, zorder=4,
               edgecolors=WHITE, linewidth=1.2, label='Puntos Euler')
    ax.scatter([x_vals[-1]], [y_vals[-1]], color=ROSE, s=100, zorder=5,
               edgecolors=ROSE_HOVER, linewidth=2,
               label=f'Resultado ≈ {y_vals[-1]:.6f}')

    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.5)
    ax.axvline(x=x0, color=SUCCESS, linestyle='--', linewidth=1.2, alpha=0.5)
    ax.axvline(x=xn, color=ERROR, linestyle='--', linewidth=1.2, alpha=0.5)

    ax.set_title('Método de Euler — Comparación Exacta vs Aproximada', fontsize=13,
                 fontweight='bold', pad=12, color=CHART_LINE)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('F(x) = ∫f(x) dx', fontsize=11)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, edgecolor=BORDER)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _incrustar_figura(fig, frame)

# Aplicación principal con interfaz gráfica usando tkinter

class EulerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integración Numérica — Método de Euler")
        self.root.configure(bg=BG_SOFT)

        # Centrar ventana en pantalla
        px = (self.root.winfo_screenwidth() - WIN_WIDTH) // 2
        py = (self.root.winfo_screenheight() - WIN_HEIGHT) // 2
        self.root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{px}+{py}")
        self.root.minsize(WIN_MIN_WIDTH, WIN_MIN_HEIGHT)
        self.root.protocol("WM_DELETE_WINDOW", self._salir)

        # Estado del cálculo
        self.resultado_aproximado = None
        self.resultado_exacto = None
        self.error_porcentual = None
        self.x_vals = None
        self.y_vals = None
        self.funcion_f = None
        self.polinomio_str = ""

        self._crear_estilos()
        self._crear_widgets()

    # Estilos

    def _crear_estilos(self):
        # Configura estilos globales de ttk con tema clam.
        estilo = ttk.Style()
        estilo.theme_use('clam')

        estilo.configure('TLabel', background=WHITE, foreground=TEXT_DARK, font=FONT_NORMAL)
        estilo.configure('TFrame', background=WHITE)
        estilo.configure('TButton', font=FONT_NORMAL, padding=(12, 6))
        estilo.configure('TEntry', font=FONT_NORMAL, padding=(6, 4))
        estilo.configure('TLabelframe', background=WHITE, foreground=ROSE, font=FONT_SUBTITLE)
        estilo.configure('TLabelframe.Label', background=WHITE, foreground=ROSE, font=FONT_SUBTITLE)
        estilo.configure('Panel.TFrame', background=BG_SOFT)

        # Botón Calcular
        estilo.configure('Calcular.TButton', background=ROSE, foreground=WHITE,
                         font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=(20, 8))
        estilo.map('Calcular.TButton', background=[('active', ROSE_HOVER)])

        # Botón Limpiar
        estilo.configure('Limpiar.TButton', background=GRAY_TEXT, foreground=WHITE,
                         font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=(20, 8))
        estilo.map('Limpiar.TButton', background=[('active', '#7a7a7f')])

        # Botón Salir
        estilo.configure('Salir.TButton', background=ERROR, foreground=WHITE,
                         font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=(20, 8))
        estilo.map('Salir.TButton', background=[('active', '#c0392b')])

        # Pestañas
        estilo.configure('TNotebook', background=WHITE, borderwidth=0)
        estilo.configure('TNotebook.Tab', font=FONT_NORMAL, padding=(12, 5))
        estilo.map('TNotebook.Tab', background=[('selected', WHITE)])

        # Tabla
        estilo.configure('Treeview', font=FONT_SMALL, rowheight=26)
        estilo.configure('Treeview.Heading', font=FONT_NORMAL, padding=(6, 4))
        estilo.map('Treeview', background=[('selected', ROSE)],
                   foreground=[('selected', WHITE)])

    # Widgets

    def _crear_widgets(self):
        # Construye la jerarquía completa de la interfaz.
        principal = ttk.Frame(self.root, style='Panel.TFrame')
        principal.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self._crear_cabecera(principal)

        cuerpo = ttk.Frame(principal, style='Panel.TFrame')
        cuerpo.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self._crear_panel_izquierdo(cuerpo)
        self._crear_panel_derecho(cuerpo)

        # Barra de estado
        self.barra = tk.Label(principal, text="Listo. Ingrese los coeficientes y presione Calcular.",
                              font=FONT_SMALL, bg='#e8e0e4', fg=GRAY_TEXT,
                              anchor=tk.W, padx=12, pady=6, relief='sunken', bd=1)
        self.barra.pack(fill=tk.X, pady=(8, 0))

    def _crear_cabecera(self, padre):
        # Crea la barra de título superior.
        tk.Label(padre, text="Integración Numérica — Método de Euler",
                 font=('Segoe UI', 16, 'bold'), bg=BG_SOFT, fg=TEXT_DARK).pack(anchor=tk.W)
        tk.Label(padre, text="f(x) = ax³ + bx² + cx + d",
                 font=('Segoe UI', 11, 'italic'), bg=BG_SOFT, fg=GRAY_TEXT).pack(anchor=tk.W)

    def _crear_panel_izquierdo(self, padre):
        # Panel izquierdo: coeficientes, parámetros y botones.
        frame = tk.Frame(padre, bg=WHITE, highlightbackground=BORDER,
                         highlightthickness=1, relief='solid')
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), ipadx=5, ipady=8)

        tk.Label(frame, text="Parámetros de entrada", font=FONT_SUBTITLE,
                 bg=WHITE, fg=ROSE).pack(anchor=tk.W, padx=15, pady=(12, 5))
        tk.Frame(frame, bg=ROSE_LIGHT, height=2).pack(fill=tk.X, padx=15, pady=(0, 8))

        # Coeficientes del polinomio
        tk.Label(frame, text="Coeficientes del polinomio", font=('Segoe UI', 9, 'bold'),
                 bg=WHITE, fg=GRAY_TEXT).pack(anchor=tk.W, padx=15, pady=(5, 3))
        self.entry_a = self._campo(frame, "a (x³):")
        self.entry_b = self._campo(frame, "b (x²):")
        self.entry_c = self._campo(frame, "c (x):")
        self.entry_d = self._campo(frame, "d (cte):")

        # Parámetros de integración
        tk.Label(frame, text="Parámetros de integración", font=('Segoe UI', 9, 'bold'),
                 bg=WHITE, fg=GRAY_TEXT).pack(anchor=tk.W, padx=15, pady=(12, 3))
        self.entry_x0 = self._campo(frame, "x₀ (límite inf.):")
        self.entry_xn = self._campo(frame, "xₙ (límite sup.):")
        self.entry_h = self._campo(frame, "h (tamaño paso):")
        self.entry_y0 = self._campo(frame, "y₀ (cond. inicial):")

        # Botones
        tk.Frame(frame, bg=WHITE, height=8).pack(fill=tk.X)
        bf = tk.Frame(frame, bg=WHITE)
        bf.pack(fill=tk.X, padx=15, pady=(8, 10))

        ttk.Button(bf, text="Calcular", style='Calcular.TButton',
                   command=self._calcular).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(bf, text="Limpiar", style='Limpiar.TButton',
                   command=self._limpiar).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(bf, text="Salir", style='Salir.TButton',
                   command=self._salir).pack(fill=tk.X)

    def _campo(self, padre, texto):
        # Crea una fila etiqueta + campo de entrada.
        fila = tk.Frame(padre, bg=WHITE)
        fila.pack(fill=tk.X, padx=15, pady=3)
        tk.Label(fila, text=texto, font=FONT_NORMAL, bg=WHITE,
                 fg=TEXT_DARK, width=18, anchor=tk.W).pack(side=tk.LEFT)
        e = tk.Entry(fila, font=FONT_NORMAL, width=12, bd=1, relief='solid',
                     highlightthickness=1, highlightcolor=ROSE,
                     highlightbackground=BORDER)
        e.pack(side=tk.RIGHT)
        return e

    def _crear_panel_derecho(self, padre):
        # Panel derecho: resumen de resultados y notebook con pestañas.
        frame = tk.Frame(padre, bg=WHITE, highlightbackground=BORDER,
                         highlightthickness=1, relief='solid')
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Resumen de resultados
        rf = tk.Frame(frame, bg=WHITE)
        rf.pack(fill=tk.X, padx=12, pady=(10, 5))
        tk.Label(rf, text="Resultados", font=FONT_SUBTITLE,
                 bg=WHITE, fg=TEXT_DARK).pack(anchor=tk.W)
        tk.Frame(rf, bg=ROSE_LIGHT, height=2).pack(fill=tk.X, pady=(5, 8))

        # Tarjetas de resultados (3 columnas)
        ind = tk.Frame(rf, bg=WHITE)
        ind.pack(fill=tk.X)
        for i, (tit, color, attr) in enumerate([
            ("Aproximado (Euler)", ROSE, 'lbl_aprox'),
            ("Exacto (Analítico)", SUCCESS, 'lbl_exact'),
            ("Error porcentual", '#e67e22', 'lbl_err')
        ]):
            card = tk.Frame(ind, bg=BG_SOFT, highlightbackground=BORDER,
                            highlightthickness=1, relief='solid')
            card.grid(row=0, column=i, padx=4, pady=3, sticky='nsew')
            ind.grid_columnconfigure(i, weight=1)
            tk.Label(card, text=tit, font=('Segoe UI', 9),
                     bg=BG_SOFT, fg=GRAY_TEXT).pack(padx=10, pady=(8, 2))
            lbl = tk.Label(card, text="—", font=FONT_MONO, bg=BG_SOFT, fg=color)
            lbl.pack(padx=10, pady=(0, 8))
            setattr(self, attr, lbl)

        # Notebook con pestañas
        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(5, 8))

        # Pestaña 1: Tabla iterativa
        self.tab_tabla = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tabla, text="Tabla iterativa")
        self._crear_tabla()

        # Pestaña 2: Gráfica analítica
        self.tab_an = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_an, text="Gráfica analítica")
        self.frame_an = tk.Frame(self.tab_an, bg=WHITE)
        self.frame_an.pack(fill=tk.BOTH, expand=True)
        self._placeholder(self.frame_an, "Gráfica analítica")

        # Pestaña 3: Gráfica numérica
        self.tab_num = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_num, text="Gráfica numérica")
        self.frame_num = tk.Frame(self.tab_num, bg=WHITE)
        self.frame_num.pack(fill=tk.BOTH, expand=True)
        self._placeholder(self.frame_num, "Gráfica numérica")

    def _crear_tabla(self):
        # Crea el Treeview para la tabla iterativa.
        ft = tk.Frame(self.tab_tabla, bg=WHITE)
        ft.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        sy = ttk.Scrollbar(ft, orient=tk.VERTICAL)
        sx = ttk.Scrollbar(ft, orient=tk.HORIZONTAL)

        cols = ('i', 'x_i', 'f(x_i)', 'y_i', 'y_{i+1}')
        self.tabla = ttk.Treeview(ft, columns=cols, show='headings',
                                  yscrollcommand=sy.set, xscrollcommand=sx.set, height=15)
        sy.config(command=self.tabla.yview)
        sx.config(command=self.tabla.xview)

        for c in cols:
            self.tabla.heading(c, text=c)
        self.tabla.column('i', width=50, anchor='center')
        self.tabla.column('x_i', width=120, anchor='center')
        self.tabla.column('f(x_i)', width=140, anchor='center')
        self.tabla.column('y_i', width=160, anchor='center')
        self.tabla.column('y_{i+1}', width=160, anchor='center')

        self.tabla.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        sx.pack(side=tk.BOTTOM, fill=tk.X)

        self.tabla_ph = tk.Label(self.tab_tabla, text="Realice un cálculo para ver la tabla",
                                 font=FONT_NORMAL, bg=WHITE, fg='#bdc3c7')
        self.tabla_ph.pack(pady=20)

    def _placeholder(self, frame, texto):
        # Muestra un mensaje placeholder en un frame vacío.
        _limpiar_frame(frame)
        tk.Label(frame, text=f"Realice un cálculo para ver la {texto.lower()}",
                 font=FONT_NORMAL, bg=WHITE, fg='#bdc3c7').pack(expand=True)

    # Logica de cálculo

    def _calcular(self):
        # Ejecuta el cálculo completo: lee, valida, procesa y actualiza.
        try:
            # Leer coeficientes
            a = self._leer_float('entry_a', 'coeficiente a')
            b = self._leer_float('entry_b', 'coeficiente b')
            c = self._leer_float('entry_c', 'coeficiente c')
            d = self._leer_float('entry_d', 'coeficiente d')

            # Leer parámetros
            x0 = self._leer_float('entry_x0', 'límite inferior x₀')
            xn = self._leer_float('entry_xn', 'límite superior xₙ')
            h = self._leer_positivo('entry_h', 'tamaño de paso h')
            y0 = self._leer_float('entry_y0', 'condición inicial y₀')

            # Validar intervalo
            if x0 >= xn:
                raise ValueError("x₀ debe ser menor que xₙ")
            if h > (xn - x0):
                raise ValueError(f"El paso h = {h} es mayor que el intervalo [{x0}, {xn}]")

            # Construir polinomio
            polinomio, x_sym = construir_polinomio(a, b, c, d)
            self.polinomio_str = str(polinomio)
            self.funcion_f = sp.lambdify(x_sym, polinomio, 'numpy')

            # Método de Euler
            self.x_vals, self.y_vals, self.resultado_aproximado = metodo_euler(
                self.funcion_f, x0, xn, h, y0
            )

            # Integral exacta y error
            self.resultado_exacto, _ = integral_exacta(polinomio, x_sym, x0, xn)
            self.error_porcentual = calcular_error(self.resultado_aproximado, self.resultado_exacto)

            # Actualizar interfaz
            self._actualizar_resultados()
            self._actualizar_tabla()
            self._actualizar_graficas(x0, xn, a, b, c, d)

            self.barra.config(text=f"Cálculo completado. Integral ≈ {self.resultado_aproximado:.6f}", fg=SUCCESS)

        except ValueError as e:
            messagebox.showerror("Error de entrada", str(e))
            self.barra.config(text=f"Error: {str(e)}", fg=ERROR)
        except Exception as e:
            messagebox.showerror("Error inesperado", str(e))
            self.barra.config(text="Error inesperado.", fg=ERROR)

    def _leer_float(self, attr, nombre):
        # Lee un valor float de un Entry. Lanza ValueError si es inválido.
        texto = getattr(self, attr).get().strip()
        if not texto:
            raise ValueError(f"'{nombre}' está vacío.")
        return float(texto)

    def _leer_positivo(self, attr, nombre):
        # Lee un valor float que debe ser positivo (> 0).
        v = self._leer_float(attr, nombre)
        if v <= 0:
            raise ValueError(f"'{nombre}' debe ser positivo.")
        return v

    def _actualizar_resultados(self):
        # Actualiza las tres tarjetas de resultados numéricos.
        self.lbl_aprox.config(text=f"{self.resultado_aproximado:.8f}")
        self.lbl_exact.config(text=f"{self.resultado_exacto:.8f}")
        if self.error_porcentual == float('inf'):
            self.lbl_err.config(text="∞ (división por cero)")
        else:
            self.lbl_err.config(text=f"{self.error_porcentual:.6f} %")

    def _actualizar_tabla(self):
        # Llena la tabla iterativa con los datos de cada paso.
        self.tabla_ph.pack_forget()
        for item in self.tabla.get_children():
            self.tabla.delete(item)

        datos = generar_tabla(self.x_vals, self.y_vals, self.funcion_f)
        for f in datos:
            fxi = f"{f['f(x_i)']:.6f}" if f['f(x_i)'] is not None else "—"
            yi1 = f"{f['y_{i+1}']:.8f}" if f['y_{i+1}'] is not None else "—"
            self.tabla.insert('', tk.END, values=(
                f['i'], f"{f['x_i']:.6f}", fxi,
                f"{f['y_i']:.8f}", yi1
            ))

    def _actualizar_graficas(self, x0, xn, a, b, c, d):
        # Genera y muestra ambas gráficas.
        graficar_analitica(self.frame_an, self.polinomio_str, x0, xn, a, b, c, d,
                           self.resultado_exacto)
        graficar_numerica(self.frame_num, self.x_vals, self.y_vals, x0, xn, a, b, c, d,
                          self.polinomio_str, self.resultado_exacto)

    # Acciones de botones

    def _limpiar(self):
        # Restablece todos los campos y resultados al estado inicial.
        for attr in ['entry_a', 'entry_b', 'entry_c', 'entry_d',
                     'entry_x0', 'entry_xn', 'entry_h', 'entry_y0']:
            getattr(self, attr).delete(0, tk.END)

        # Valores por defecto
        self.entry_x0.insert(0, "0")
        self.entry_xn.insert(0, "1")
        self.entry_h.insert(0, "0.1")
        self.entry_y0.insert(0, "0")

        # Restablecer etiquetas
        for lbl in [self.lbl_aprox, self.lbl_exact, self.lbl_err]:
            lbl.config(text="—")

        # Limpiar tabla
        for item in self.tabla.get_children():
            self.tabla.delete(item)
        self.tabla_ph.pack(pady=20)

        # Restablecer gráficas
        self._placeholder(self.frame_an, "Gráfica analítica")
        self._placeholder(self.frame_num, "Gráfica numérica")

        # Restablecer estado
        self.resultado_aproximado = None
        self.resultado_exacto = None
        self.error_porcentual = None
        self.x_vals = None
        self.y_vals = None
        self.funcion_f = None
        self.polinomio_str = ""

        self.barra.config(text="Campos limpiados.", fg=GRAY_TEXT)

    def _salir(self):
        self.root.destroy()

# Entrada al programa
if __name__ == "__main__":
    root = tk.Tk()
    app = EulerApp(root)
    root.mainloop()