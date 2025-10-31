import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта sklearn с обработкой ошибок
try:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    st.error("❌ Библиотека scikit-learn не установлена. Установите её: pip install scikit-learn")
    SKLEARN_AVAILABLE = False
    # Создаем заглушки для функций
    def r2_score(*args, **kwargs):
        return 0
    def mean_squared_error(*args, **kwargs):
        return 0
    def mean_absolute_error(*args, **kwargs):
        return 0

# Универсальная газовая постоянная
R = 8.314  # Дж/(моль·К)

st.title("🔬 Комплексный анализ кинетики σ-фазы с универсальной моделью и калькулятором")
st.markdown("""
**Улучшенный анализ с универсальными моделями:**
- Модели, работающие для всех температур одновременно
- Учет минимальной температуры начала превращения (550°C)
- Учет температуры растворения σ-фазы (900°C)
- Интерактивный калькулятор для прогнозирования
- **НОВОЕ: Модель температуры T = k·(c/t^0.5)^n**
- Аррениусовская зависимость констант скорости
- Автоматическое масштабирование графиков под данные
- **Поддержка анализа для разных зерен**
""")

# Информация об установке зависимостей
with st.expander("🔧 Установка зависимостей (если возникают ошибки)"):
    st.markdown("""
    **Если возникают ошибки, установите зависимости:**
    ```bash
    pip install streamlit pandas numpy matplotlib scipy seaborn scikit-learn openpyxl
    ```
    
    **Или создайте файл requirements.txt:**
    ```
    streamlit
    pandas
    numpy
    matplotlib
    scipy
    seaborn
    scikit-learn
    openpyxl
    ```
    """)

# Загрузка данных
st.header("1. Загрузка данных")

uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", 
                               type=['csv', 'xlsx', 'xls'])

# Параметры анализа
st.subheader("Параметры анализа:")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    # Получаем список доступных зерен из данных, если файл загружен
    available_grains = ["8", "9", "10"]  # значения по умолчанию как строки
    
    if uploaded_file is not None:
        try:
            # Предварительная загрузка для получения списка зерен
            if uploaded_file.name.endswith('.csv'):
                preview_df = pd.read_csv(uploaded_file)
            else:
                preview_df = pd.read_excel(uploaded_file)
            
            if 'G' in preview_df.columns:
                # Преобразуем все значения в строки для единообразия
                available_grains = sorted([str(g) for g in preview_df['G'].unique() if pd.notna(g)])
        except:
            pass
    
    # Выбор способа ввода номера зерна
    grain_input_method = st.radio(
        "Выбор зерна",
        ["Из списка", "Вручную"],
        horizontal=True,
        help="Выберите способ указания номера зерна"
    )
    
    if grain_input_method == "Из списка":
        target_grain = st.selectbox(
            "Обозначение зерна из списка", 
            options=available_grains,
            index=min(2, len(available_grains)-1) if available_grains else 0,
            help="Выберите обозначение зерна из доступных в данных"
        )
    else:
        target_grain = st.text_input(
            "Обозначение зерна (вручную)",
            value="10",
            help="Введите обозначение зерна (может быть числом или текстом, например: 8, 9, 10, РД1, РД2)"
        )

with col2:
    initial_diameter = st.number_input("Начальный диаметр d₀ (мкм)", 
                                     value=0.1, min_value=0.0, step=0.1,
                                     help="Рекомендуется использовать небольшое положительное значение (0.1-0.5 мкм)")
with col3:
    enable_phase_analysis = st.checkbox("Анализ содержания фазы", 
                                      value=True, 
                                      help="Анализ кинетики фазового превращения по содержанию σ-фазы")
with col4:
    min_temperature = st.number_input("Мин. температура (°C)", 
                                    value=550.0, min_value=0.0, step=10.0,
                                    help="Температура, ниже которой превращение не происходит")
with col5:
    dissolution_temperature = st.number_input("Темп. растворения (°C)", 
                                           value=900.0, min_value=0.0, step=10.0,
                                           help="Температура, выше которой σ-фаза растворяется")

# НОВЫЙ ПАРАМЕТР: Включение модели температуры
enable_temperature_model = st.checkbox(
    "Включить модель температуры T = k·(c/t^0.5)^n", 
    value=True,
    help="Анализ зависимости температуры от содержания фазы и времени"
)

# Инициализация session_state для калькулятора
if 'calc_type' not in st.session_state:
    st.session_state.calc_type = "Прогноз диаметра/содержания"
if 'target_time' not in st.session_state:
    st.session_state.target_time = 100.0
if 'target_temp' not in st.session_state:
    st.session_state.target_temp = 800.0
if 'calc_mode' not in st.session_state:
    st.session_state.calc_mode = "Диаметр"
if 'target_time_temp' not in st.session_state:
    st.session_state.target_time_temp = 100.0
if 'target_value' not in st.session_state:
    st.session_state.target_value = 2.0
if 'temp_mode' not in st.session_state:
    st.session_state.temp_mode = "Диаметр (мкм)"
if 'interactive_temp' not in st.session_state:
    st.session_state.interactive_temp = 800.0
if 'interactive_mode' not in st.session_state:
    st.session_state.interactive_mode = "Диаметр"
if 'max_time_interactive' not in st.session_state:
    st.session_state.max_time_interactive = 400000.0

# Функции для расчета метрик качества
def calculate_comprehensive_metrics(y_true, y_pred):
    """Расчет комплексных метрик качества модели"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'R²': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0}
    
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    try:
        metrics = {
            'R²': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-10))) * 100
        }
        return metrics
    except:
        return {'R²': 0, 'RMSE': 0, 'MAE': 0, 'MAPE': 0}

def safe_plot_with_diagnostics(ax, t_exp, y_exp, y_pred, t_range=None, y_range=None, 
                              title="", xlabel="Время (часы)", ylabel="", 
                              model_name="Модель"):
    """Безопасная визуализация с диагностикой"""
    try:
        ax.clear()
        
        if len(t_exp) == 0 or len(y_exp) == 0:
            ax.text(0.5, 0.5, 'Нет данных', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            return
        
        valid_mask = ~np.isnan(t_exp) & ~np.isnan(y_exp) & ~np.isnan(y_pred)
        t_exp = t_exp[valid_mask]
        y_exp = y_exp[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(t_exp) == 0:
            ax.text(0.5, 0.5, 'Нет валидных данных', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            return
        
        ax.scatter(t_exp, y_exp, alpha=0.8, s=60, label='Эксперимент', color='blue')
        
        if t_range is not None and y_range is not None and len(t_range) > 0 and len(y_range) > 0:
            if ylabel == 'Диаметр (мкм)':
                valid_range_mask = y_range > 0
                if np.any(valid_range_mask):
                    ax.plot(t_range[valid_range_mask], y_range[valid_range_mask], 'r--', 
                           linewidth=2, label=model_name)
            else:
                ax.plot(t_range, y_range, 'r--', linewidth=2, label=model_name)
        
        sorted_idx = np.argsort(t_exp)
        ax.plot(t_exp.iloc[sorted_idx] if hasattr(t_exp, 'iloc') else t_exp[sorted_idx], 
               y_exp.iloc[sorted_idx] if hasattr(y_exp, 'iloc') else y_exp[sorted_idx], 
               'b:', alpha=0.5, label='Тренд эксперимента')
        
        for i in range(min(len(t_exp), len(y_exp), len(y_pred))):
            t_val = t_exp.iloc[i] if hasattr(t_exp, 'iloc') else t_exp[i]
            y_true = y_exp.iloc[i] if hasattr(y_exp, 'iloc') else y_exp[i]
            y_pred_val = y_pred.iloc[i] if hasattr(y_pred, 'iloc') else y_pred[i]
            ax.plot([t_val, t_val], [y_true, y_pred_val], 'gray', alpha=0.3, linewidth=1)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        metrics = calculate_comprehensive_metrics(y_exp, y_pred)
        ax.text(0.02, 0.98, f"R² = {metrics['R²']:.3f}\nRMSE = {metrics['RMSE']:.2f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
    except Exception as e:
        ax.text(0.5, 0.5, f'Ошибка построения:\n{str(e)[:50]}...', 
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.set_title(title)

# Функции для JMAK-анализа
def jmak_model(t, k, n):
    """JMAK модель: X(t) = 1 - exp(-(k*t)^n)"""
    return 1 - np.exp(-(k * t) ** n)

def fit_jmak_model(time, f_phase, initial_n=1.0):
    """Подбор параметров JMAK модели"""
    f_normalized = np.array(f_phase) / 100.0
    
    valid_mask = ~np.isnan(time) & ~np.isnan(f_normalized) & (f_normalized >= 0) & (f_normalized <= 1)
    time_valid = time[valid_mask]
    f_valid = f_normalized[valid_mask]
    
    if len(time_valid) < 2:
        return None, None, None
    
    try:
        k_guess = 1.0 / np.mean(time_valid) if np.mean(time_valid) > 0 else 0.1
        
        popt, pcov = curve_fit(jmak_model, time_valid, f_valid, 
                              p0=[k_guess, initial_n],
                              bounds=([1e-6, 0.1], [10, 4]),
                              maxfev=5000)
        
        k_fit, n_fit = popt
        return k_fit, n_fit, pcov
    
    except Exception as e:
        return None, None, None

def calculate_jmak_predictions(time, k, n):
    """Расчет предсказаний JMAK модели"""
    return jmak_model(time, k, n) * 100

# ФУНКЦИИ ДЛЯ УНИВЕРСАЛЬНОЙ МОДЕЛИ
def arrhenius_model(T, A, Ea):
    """Аррениусовская модель: k = A * exp(-Ea/(R*T))"""
    return A * np.exp(-Ea / (R * T))

def effective_rate_constant_single(T, A, Ea, T_min, T_diss):
    """Эффективная константа скорости для одного значения температуры"""
    T_kelvin = T + 273.15
    T_min_kelvin = T_min + 273.15
    T_diss_kelvin = T_diss + 273.15
    
    if T_kelvin <= T_min_kelvin:
        return 0.0
    elif T_kelvin >= T_diss_kelvin:
        return 0.0
    else:
        return arrhenius_model(T_kelvin, A, Ea)

def effective_rate_constant_array(T_array, A, Ea, T_min, T_diss):
    """Эффективная константа скорости для массива температур"""
    T_kelvin = T_array + 273.15
    T_min_kelvin = T_min + 273.15
    T_diss_kelvin = T_diss + 273.15
    
    k_eff = np.zeros_like(T_kelvin)
    valid_mask = (T_kelvin > T_min_kelvin) & (T_kelvin < T_diss_kelvin)
    k_eff[valid_mask] = arrhenius_model(T_kelvin[valid_mask], A, Ea)
    
    return k_eff

def universal_diameter_model_single(t, T, A, Ea, n, d0, T_min, T_diss):
    """Универсальная модель для диаметра для одного значения"""
    k_eff = effective_rate_constant_single(T, A, Ea, T_min, T_diss)
    return (k_eff * t + d0**n)**(1/n)

def universal_diameter_model_array(t_array, T_array, A, Ea, n, d0, T_min, T_diss):
    """Универсальная модель для диаметра для массивов"""
    k_eff = effective_rate_constant_array(T_array, A, Ea, T_min, T_diss)
    return (k_eff * t_array + d0**n)**(1/n)

def universal_phase_model_single(t, T, A, Ea, n_jmak, T_min, T_diss):
    """Универсальная модель для содержания фазы для одного значения"""
    k_eff = effective_rate_constant_single(T, A, Ea, T_min, T_diss)
    return jmak_model(t, k_eff, n_jmak) * 100

def universal_phase_model_array(t_array, T_array, A, Ea, n_jmak, T_min, T_diss):
    """Универсальная модель для содержания фазы для массивов"""
    k_eff = effective_rate_constant_array(T_array, A, Ea, T_min, T_diss)
    f_normalized = jmak_model(t_array, k_eff, n_jmak)
    return f_normalized * 100

# НОВАЯ ФУНКЦИЯ: Модель температуры T = k·(c/t^0.5)^n
def temperature_model(params, c, t):
    """Модель температуры: T = k·(c/t^0.5)^n"""
    k, n = params
    # Защита от деления на ноль и отрицательных значений
    t_safe = np.maximum(t, 1e-10)
    c_safe = np.maximum(c, 1e-10)
    return k * (c_safe / np.sqrt(t_safe)) ** n

def fit_temperature_model(df, temp_min=590, temp_max=660, time_min=20000, time_max=400000):
    """Подбор параметров модели температуры T = k·(c/t^0.5)^n с учетом ограничений"""
    try:
        # Фильтруем данные по указанным ограничениям
        df_filtered = df[
            (df['T'] >= temp_min) & 
            (df['T'] <= temp_max) & 
            (df['t'] >= time_min) & 
            (df['t'] <= time_max) &
            (df['f'].notna())
        ].copy()
        
        st.info(f"🔍 Для подбора модели используется {len(df_filtered)} точек в диапазоне: "
               f"T={temp_min}-{temp_max}°C, t={time_min:,}-{time_max:,} часов")
        
        if len(df_filtered) < 3:
            st.warning(f"❌ Недостаточно данных в указанном диапазоне ({len(df_filtered)} точек)")
            return None, None
        
        # Преобразуем температуру в Кельвины
        T_kelvin = df_filtered['T'].values + 273.15
        c_values = df_filtered['f'].values
        t_values = df_filtered['t'].values
        
        # Улучшенные начальные приближения
        k_guess = 900
        n_guess = 1.2
        
        def model_to_fit(x, k, n):
            c, t = x
            return temperature_model([k, n], c, t)
        
        # Пробуем разные начальные значения
        initial_guesses = [
            [800, 1.0],
            [900, 1.2],  
            [1000, 1.5],
            [700, 0.8],
            [1100, 1.8]
        ]
        
        best_params = None
        best_pcov = None
        best_r2 = -float('inf')
        
        for i, (k_guess, n_guess) in enumerate(initial_guesses):
            try:
                popt, pcov = curve_fit(
                    model_to_fit,
                    [c_values, t_values],
                    T_kelvin,
                    p0=[k_guess, n_guess],
                    bounds=([500, 0.5], [1500, 3.0]),
                    method='trf',
                    maxfev=10000
                )
                
                # Проверяем качество подбора
                predictions = temperature_model(popt, c_values, t_values)
                r2 = r2_score(T_kelvin, predictions)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = popt
                    best_pcov = pcov
                    
            except Exception as e:
                continue
        
        if best_params is not None:
            return best_params, best_pcov
        else:
            # Аналитический метод если автоматический не сработал
            return analytical_parameter_estimation(df_filtered)
            
    except Exception as e:
        st.error(f"Ошибка подбора модели температуры: {str(e)}")
        return None, None

def analytical_parameter_estimation(df):
    """Аналитическая оценка параметров на основе данных"""
    try:
        # Преобразуем температуру в Кельвины
        T_kelvin = df['T'].values + 273.15
        c_values = df['f'].values
        t_values = df['t'].values
        
        # Вычисляем c/√t для каждой точки
        c_over_sqrt_t = c_values / np.sqrt(t_values)
        
        # Линеаризуем модель: log(T) = log(k) + n * log(c/√t)
        log_T = np.log(T_kelvin)
        log_ratio = np.log(c_over_sqrt_t)
        
        # Исключаем бесконечные значения
        valid_mask = np.isfinite(log_T) & np.isfinite(log_ratio)
        log_T_valid = log_T[valid_mask]
        log_ratio_valid = log_ratio[valid_mask]
        
        if len(log_T_valid) < 2:
            return None, None
        
        # Линейная регрессия для нахождения n и log(k)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_ratio_valid, log_T_valid
        )
        
        n_est = slope
        k_est = np.exp(intercept)
        
        return [k_est, n_est], None
        
    except Exception as e:
        st.error(f"Ошибка аналитического метода: {str(e)}")
        return None, None

def diagnose_temperature_data(df, temp_min=590, temp_max=660, time_min=20000, time_max=400000):
    """Диагностика данных для модели температуры"""
    st.subheader("🔍 Диагностика данных для модели температуры")
    
    # Фильтруем данные по указанным ограничениям
    df_filtered = df[
        (df['T'] >= temp_min) & 
        (df['T'] <= temp_max) & 
        (df['t'] >= time_min) & 
        (df['t'] <= time_max) &
        (df['f'].notna())
    ].copy()
    
    st.info(f"**Статистика данных в диапазоне T={temp_min}-{temp_max}°C, t={time_min:,}-{time_max:,} часов:**")
    st.write(f"- Всего точек: {len(df_filtered)}")
    if len(df_filtered) > 0:
        st.write(f"- Температуры: от {df_filtered['T'].min():.1f} до {df_filtered['T'].max():.1f}°C")
        st.write(f"- Время: от {df_filtered['t'].min():,} до {df_filtered['t'].max():,} часов")
        st.write(f"- Содержание фазы: от {df_filtered['f'].min():.2f} до {df_filtered['f'].max():.2f}%")
    
    if len(df_filtered) > 0:
        # Анализ распределения c/√t
        df_filtered['c_over_sqrt_t'] = df_filtered['f'] / np.sqrt(df_filtered['t'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # График 1: Распределение c/√t
        axes[0].scatter(df_filtered['c_over_sqrt_t'], df_filtered['T'], alpha=0.7)
        axes[0].set_xlabel('c/√t (%/√час)')
        axes[0].set_ylabel('Температура (°C)')
        axes[0].set_title('Распределение данных')
        axes[0].grid(True, alpha=0.3)
        
        # График 2: Зависимость температуры от времени
        scatter = axes[1].scatter(df_filtered['t'], df_filtered['T'], 
                                 c=df_filtered['f'], cmap='viridis', alpha=0.7)
        axes[1].set_xlabel('Время (часы)')
        axes[1].set_ylabel('Температура (°C)')
        axes[1].set_title('Температура vs Время (цвет - содержание фазы)')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Содержание фазы (%)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Анализ корреляции
        if len(df_filtered) >= 3:
            correlation = df_filtered['c_over_sqrt_t'].corr(df_filtered['T'])
            st.write(f"- Корреляция между c/√t и T: {correlation:.3f}")
            
            if abs(correlation) < 0.3:
                st.warning("⚠️ Слабая корреляция между переменными. Модель может плохо подходить.")
            elif abs(correlation) < 0.6:
                st.info("📊 Умеренная корреляция между переменными.")
            else:
                st.success("✅ Хорошая корреляция между переменными.")
    
    return df_filtered

def fit_universal_diameter_model(df, best_n, d0, T_min, T_diss):
    """Подбор параметров универсальной модели для диаметра"""
    try:
        df_filtered = df[(df['T'] >= T_min) & (df['T'] <= T_diss)].copy()
        
        if len(df_filtered) < 3:
            st.warning(f"❌ Недостаточно данных в рабочем диапазоне ({len(df_filtered)} точек)")
            return None, None
        
        t_all = df_filtered['t'].values
        T_all = df_filtered['T'].values
        d_all = df_filtered['d'].values
        
        A_guess = 0.1
        Ea_guess = 150000
        
        def model_function(params, t_data, T_data):
            A, Ea = params
            return universal_diameter_model_array(t_data, T_data, A, Ea, best_n, d0, T_min, T_diss)
        
        popt, pcov = curve_fit(
            lambda x, A, Ea: model_function([A, Ea], x[0], x[1]),
            [t_all, T_all], d_all,
            p0=[A_guess, Ea_guess],
            bounds=([1e-10, 50000], [1e5, 500000]),
            maxfev=10000
        )
        
        return popt, pcov
        
    except Exception as e:
        st.error(f"Ошибка подбора универсальной модели диаметра: {str(e)}")
        return None, None

def fit_universal_phase_model(df, T_min, T_diss):
    """Подбор параметров универсальной модели для содержания фазы"""
    try:
        df_filtered = df[(df['T'] >= T_min) & (df['T'] <= T_diss)].copy()
        
        if len(df_filtered) < 3:
            st.warning(f"❌ Недостаточно данных в рабочем диапазоне для JMAK ({len(df_filtered)} точек)")
            return None, None
        
        t_all = df_filtered['t'].values
        T_all = df_filtered['T'].values
        f_all = df_filtered['f'].values
        
        A_guess = 0.1
        Ea_guess = 150000
        n_guess = 1.5
        
        def model_function(params, t_data, T_data):
            A, Ea, n = params
            return universal_phase_model_array(t_data, T_data, A, Ea, n, T_min, T_diss)
        
        popt, pcov = curve_fit(
            lambda x, A, Ea, n: model_function([A, Ea, n], x[0], x[1]),
            [t_all, T_all], f_all,
            p0=[A_guess, Ea_guess, n_guess],
            bounds=([1e-10, 50000, 0.5], [1e5, 500000, 4.0]),
            maxfev=10000
        )
        
        return popt, pcov
        
    except Exception as e:
        st.error(f"Ошибка подбора универсальной модели фазы: {str(e)}")
        return None, None

# Функция для безопасной загрузки данных
def safe_load_data(uploaded_file):
    """Безопасная загрузка данных с обработкой ошибок"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV файл успешно загружен")
            return df
        else:
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                st.success("✅ Excel файл загружен с использованием openpyxl")
                return df
            except Exception as e:
                st.error(f"❌ Не удалось загрузить Excel файл: {e}")
                return None
    except Exception as e:
        st.error(f"❌ Ошибка загрузки файла: {e}")
        return None

# Основной код приложения
if uploaded_file is not None:
    df = safe_load_data(uploaded_file)
    
    if df is not None:
        required_cols = ['G', 'T', 't']
        optional_cols = ['d', 'f']
        
        # Проверяем обязательные колонки
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            st.error(f"❌ Отсутствуют обязательные колонки: {missing_required}")
            st.info("""
            **Обязательные колонки:**
            - `G` - номер/обозначение зерна
            - `T` - температура (°C)
            - `t` - время (часы)
            """)
            st.stop()
        
        # Проверяем наличие опциональных колонок
        has_diameter_data = 'd' in df.columns
        has_phase_data = 'f' in df.columns
        
        if not has_diameter_data and not has_phase_data:
            st.error("❌ В данных отсутствуют и диаметры (d), и содержание фазы (f). Нечего анализировать.")
            st.stop()
        
        # Преобразуем все значения в столбце G в строки для единообразия
        df['G'] = df['G'].astype(str)
        
        # Показываем информацию о доступных зернах
        all_grains = sorted(df['G'].unique())
        
        st.info(f"📊 В данных найдены зерна: {', '.join(map(str, all_grains))}")
        
        # Проверяем, существует ли выбранное зерно в данных
        if target_grain not in all_grains:
            st.warning(f"⚠️ Зерно '{target_grain}' не найдено в данных. Доступные зерна: {', '.join(map(str, all_grains))}")
            st.stop()
        
        # Фильтруем данные по выбранному зерну
        df_selected_grain = df[df['G'] == target_grain].copy()
        
        # Преобразуем числовые колонки
        if 'f' in df_selected_grain.columns:
            df_selected_grain['f'] = df_selected_grain['f'].astype(str).str.replace(',', '.').astype(float)
        
        if 'd' in df_selected_grain.columns:
            df_selected_grain['d'] = df_selected_grain['d'].astype(str).str.replace(',', '.').astype(float)
        
        if len(df_selected_grain) > 0:
            st.session_state['grain_data'] = df_selected_grain
            st.session_state['current_grain'] = target_grain
            st.session_state['has_diameter_data'] = has_diameter_data
            st.session_state['has_phase_data'] = has_phase_data
            
            st.success(f"✅ Данные для зерна '{target_grain}' успешно загружены! Найдено {len(df_selected_grain)} записей")
            
            # Статистика данных
            st.subheader(f"📊 Статистика данных для зерна '{target_grain}':")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                unique_temps = df_selected_grain['T'].unique()
                st.metric("Температуры", f"{len(unique_temps)} уровней")
            with col2:
                st.metric("Всего точек", f"{len(df_selected_grain)}")
            with col3:
                if has_diameter_data:
                    diameter_points = len(df_selected_grain[df_selected_grain['d'].notna()])
                    st.metric("Точек с диаметром", f"{diameter_points}")
            with col4:
                if has_phase_data:
                    phase_points = len(df_selected_grain[df_selected_grain['f'].notna()])
                    st.metric("Точек с фазой", f"{phase_points}")
            
        else:
            st.error(f"❌ В данных нет записей для зерна '{target_grain}'")
            st.stop()
    else:
        st.error("❌ Не удалось загрузить данные")
        st.stop()

# Основной расчет
if 'grain_data' in st.session_state:
    df_grain = st.session_state['grain_data']
    current_grain = st.session_state.get('current_grain', target_grain)
    has_diameter_data = st.session_state.get('has_diameter_data', False)
    has_phase_data = st.session_state.get('has_phase_data', False)
    
    # Упрощенная фильтрация данных
    df_grain_clean = df_grain.copy()
    
    # Преобразуем данные в числовой формат
    if has_phase_data and df_grain_clean['f'].dtype == 'object':
        try:
            df_grain_clean['f'] = df_grain_clean['f'].astype(str).str.replace(',', '.').astype(float)
        except:
            pass
    
    if has_diameter_data and df_grain_clean['d'].dtype == 'object':
        try:
            df_grain_clean['d'] = df_grain_clean['d'].astype(str).str.replace(',', '.').astype(float)
        except:
            pass
    
    # Фильтрация данных
    mask = (df_grain_clean['T'].notna()) & (df_grain_clean['t'].notna()) & (df_grain_clean['t'] >= 0)
    df_grain = df_grain_clean[mask]
    
    if len(df_grain) == 0:
        st.error("❌ Нет данных с корректным временем и температурой")
        st.stop()
    
    st.success(f"✅ Для анализа доступно {len(df_grain)} точек")
    
    # Анализ диаметров
    if has_diameter_data:
        st.header(f"2. 📏 Анализ диаметров σ-фазы для зерна '{current_grain}'")
        
        # Подбор показателя степени n
        st.subheader("Поиск оптимального показателя степени n")
        
        n_min, n_max, n_step = 1.0, 6.0, 0.1
        n_candidates = np.arange(n_min, n_max + n_step, n_step)
        
        n_results = {}
        
        for n in n_candidates:
            k_values = []
            
            for temp in df_grain['T'].unique():
                if temp < min_temperature or temp > dissolution_temperature:
                    continue
                    
                temp_data = df_grain[df_grain['T'] == temp]
                
                if len(temp_data) >= 2:
                    d_transformed = temp_data['d']**n - initial_diameter**n
                    
                    if (d_transformed < 0).any():
                        continue
                    
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            temp_data['t'], d_transformed
                        )
                        
                        if slope > 0:
                            d_pred_transformed = slope * temp_data['t'] + intercept
                            d_pred = (d_pred_transformed + initial_diameter**n)**(1/n)
                            
                            if (d_pred > 0).all():
                                r2_original = r2_score(temp_data['d'].values, d_pred)
                                
                                if r2_original > -100:
                                    k_values.append({
                                        'T': temp, 'K': slope, 'R2_original': r2_original
                                    })
                    except:
                        continue
            
            if k_values:
                k_df = pd.DataFrame(k_values)
                overall_r2 = k_df['R2_original'].mean()
                
                n_results[n] = {
                    'k_df': k_df, 
                    'mean_R2': overall_r2,
                    'n_temperatures': len(k_df)
                }
        
        # Визуализация подбора n
        if n_results:
            comparison_data = []
            for n, results in n_results.items():
                comparison_data.append({
                    'n': n, 
                    'Средний R²': results['mean_R2'],
                    'Количество температур': results['n_temperatures']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            best_n_row = comparison_df.loc[comparison_df['Средний R²'].idxmax()]
            best_n = best_n_row['n']
            
            st.success(f"🎯 Оптимальный показатель: n = {best_n:.1f}")
            
            # Сохраняем best_n
            grain_key = f"grain_{current_grain}"
            st.session_state[f'best_n_{grain_key}'] = best_n
            st.session_state['current_best_n'] = best_n
            
    # УНИВЕРСАЛЬНАЯ МОДЕЛЬ
    st.header("3. 🔬 Универсальная модель для всех температур")
    
    # Универсальная модель для диаметра
    if has_diameter_data and 'current_best_n' in st.session_state and st.session_state.current_best_n is not None:
        best_n = st.session_state.current_best_n
        
        st.subheader("Универсальная модель роста диаметра")
        
        universal_diameter_params, universal_diameter_cov = fit_universal_diameter_model(
            df_grain, best_n, initial_diameter, min_temperature, dissolution_temperature
        )
        
        if universal_diameter_params is not None:
            A_diam, Ea_diam = universal_diameter_params
            
            grain_key = f"grain_{current_grain}"
            st.session_state[f'universal_diameter_params_{grain_key}'] = universal_diameter_params
            st.session_state['current_universal_diameter_params'] = universal_diameter_params
            
            st.success(f"✅ Универсальная модель диаметра для зерна №{current_grain} успешно подобрана!")
            st.info(f"""
            **Параметры универсальной модели диаметра:**
            - Предэкспоненциальный множитель A = {A_diam:.4e}
            - Энергия активации Ea = {Ea_diam:.0f} Дж/моль ({Ea_diam/1000:.1f} кДж/моль)
            - Показатель степени n = {best_n:.1f}
            """)
    
    # Универсальная модель для содержания фазы
    if has_phase_data and enable_phase_analysis:
        st.subheader("Универсальная модель содержания фазы (JMAK)")
        
        universal_phase_params, universal_phase_cov = fit_universal_phase_model(
            df_grain, min_temperature, dissolution_temperature
        )
        
        if universal_phase_params is not None:
            A_phase, Ea_phase, n_phase = universal_phase_params
            
            grain_key = f"grain_{current_grain}"
            st.session_state[f'universal_phase_params_{grain_key}'] = universal_phase_params
            st.session_state['current_universal_phase_params'] = universal_phase_params
            
            st.success(f"✅ Универсальная модель содержания фазы для зерна №{current_grain} успешно подобрана!")
            st.info(f"""
            **Параметры универсальной модели фазы:**
            - Предэкспоненциальный множитель A = {A_phase:.4e}
            - Энергия активации Ea = {Ea_phase:.0f} Дж/моль ({Ea_phase/1000:.1f} кДж/моль)
            - Показатель Аврами n = {n_phase:.2f}
            """)

    # НОВЫЙ РАЗДЕЛ: МОДЕЛЬ ТЕМПЕРАТУРЫ
    if has_phase_data and enable_temperature_model:
        st.header("4. 🌡️ Модель температуры T = k·(c/t^0.5)^n")
        
        # ДИАГНОСТИКА ДАННЫХ
        df_diagnosed = diagnose_temperature_data(df_grain, 590, 660, 20000, 400000)
        
        if len(df_diagnosed) >= 3:
            # НАСТРОЙКИ ПОДБОРА ПАРАМЕТРОВ
            st.subheader("Настройки подбора параметров")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                custom_temp_min = st.number_input("Мин. температура (°C)", 
                                                value=590.0, min_value=0.0, step=10.0)
            with col2:
                custom_temp_max = st.number_input("Макс. температура (°C)", 
                                                value=660.0, min_value=0.0, step=10.0)
            with col3:
                custom_time_min = st.number_input("Мин. время (часы)", 
                                               value=20000.0, min_value=0.0, step=1000.0)
            with col4:
                custom_time_max = st.number_input("Макс. время (часы)", 
                                               value=400000.0, min_value=0.0, step=10000.0)
            
            if st.button("Подобрать параметры модели", key='fit_temp_model'):
                with st.spinner("Подбираем параметры модели..."):
                    # Подбор параметров модели температуры
                    temperature_model_params, temperature_model_cov = fit_temperature_model(
                        df_grain, custom_temp_min, custom_temp_max, custom_time_min, custom_time_max
                    )
                
                if temperature_model_params is not None:
                    k_temp, n_temp = temperature_model_params
                    
                    # Сохраняем параметры в session_state
                    grain_key = f"grain_{current_grain}"
                    st.session_state[f'temperature_model_params_{grain_key}'] = temperature_model_params
                    st.session_state['current_temperature_model_params'] = temperature_model_params
                    
                    # РАСЧЕТ КАЧЕСТВА МОДЕЛИ
                    T_kelvin_actual = df_diagnosed['T'].values + 273.15
                    predictions = temperature_model([k_temp, n_temp], 
                                                  df_diagnosed['f'].values, 
                                                  df_diagnosed['t'].values)
                    r2 = r2_score(T_kelvin_actual, predictions)
                    rmse = np.sqrt(mean_squared_error(T_kelvin_actual, predictions))
                    
                    st.success(f"✅ Модель температуры для зерна №{current_grain} успешно подобрана!")
                    st.info(f"""
                    **Параметры модели температуры:**
                    - Коэффициент k = {k_temp:.2f} K
                    - Показатель степени n = {n_temp:.3f}
                    - **Качество модели: R² = {r2:.3f}**
                    - Формула: T(K) = {k_temp:.2f}·(c/√t)^{n_temp:.3f}
                    - Формула в °C: T(°C) = {k_temp:.2f}·(c/√t)^{n_temp:.3f} - 273.15
                    """)
                    
                    # ВИЗУАЛИЗАЦИЯ МОДЕЛИ ТЕМПЕРАТУРЫ
                    st.subheader("Визуализация модели температуры")
                    
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # График 1: Предсказания vs экспериментальные данные
                    axes[0].scatter(T_kelvin_actual - 273.15, predictions - 273.15, 
                                   alpha=0.6, color='purple')
                    
                    min_temp = min(T_kelvin_actual - 273.15)
                    max_temp = max(T_kelvin_actual - 273.15)
                    margin = (max_temp - min_temp) * 0.1
                    
                    axes[0].plot([min_temp - margin, max_temp + margin], 
                               [min_temp - margin, max_temp + margin], 
                               'r--', linewidth=2, label='Идеальное согласие')
                    axes[0].set_xlabel('Фактическая температура (°C)')
                    axes[0].set_ylabel('Предсказанная температура (°C)')
                    axes[0].set_title(f'Качество модели температуры (R² = {r2:.3f})')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Метрики качества
                    axes[0].text(0.05, 0.95, 
                                f"R² = {r2:.3f}\nRMSE = {rmse:.2f}°C", 
                                transform=axes[0].transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    # График 2: Зависимость температуры от c/√t
                    c_over_sqrt_t = df_diagnosed['f'].values / np.sqrt(df_diagnosed['t'].values)
                    
                    # Сортируем для красивого графика
                    sorted_idx = np.argsort(c_over_sqrt_t)
                    c_over_sqrt_t_sorted = c_over_sqrt_t[sorted_idx]
                    T_kelvin_sorted = T_kelvin_actual[sorted_idx]
                    
                    # Предсказания модели
                    T_pred_sorted = temperature_model([k_temp, n_temp], 
                                                    c_over_sqrt_t_sorted * np.sqrt([1]*len(c_over_sqrt_t_sorted)), 
                                                    [1]*len(c_over_sqrt_t_sorted))
                    
                    axes[1].scatter(c_over_sqrt_t_sorted, T_kelvin_sorted - 273.15, alpha=0.6, 
                                   color='green', label='Эксперимент')
                    axes[1].plot(c_over_sqrt_t_sorted, T_pred_sorted - 273.15, 'r-', 
                                linewidth=2, label='Модель')
                    axes[1].set_xlabel('c/√t (%/√час)')
                    axes[1].set_ylabel('Температура (°C)')
                    axes[1].set_title(f'Зависимость температуры от c/√t')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # КАЛЬКУЛЯТОР ДЛЯ МОДЕЛИ ТЕМПЕРАТУРЫ
                    st.subheader("🧮 Калькулятор модели температуры")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        calc_c = st.number_input("Содержание фазы c (%)", 
                                               min_value=0.0, max_value=100.0, value=5.0, step=0.1)
                    with col2:
                        calc_t = st.number_input("Время t (часы)", 
                                               min_value=0.1, value=100000.0, step=1000.0)
                    
                    if st.button("Рассчитать температуру", key='calc_temp'):
                        T_pred_k = temperature_model([k_temp, n_temp], calc_c, calc_t)
                        T_pred_c = T_pred_k - 273.15
                        
                        st.success(f"**Прогнозируемая температура для зерна №{current_grain}:**")
                        st.info(f"""
                        - При содержании фазы {calc_c}% за время {calc_t:,.0f} часов:
                        - Температура: {T_pred_c:.1f}°C ({T_pred_k:.1f} K)
                        - По формуле: T = {k_temp:.2f}·({calc_c}/√{calc_t:,.0f})^{n_temp:.3f}
                        """)
        else:
            st.error("❌ Недостаточно данных в указанном диапазоне для построения модели")

# КАЛЬКУЛЯТОР ПРОГНОЗИРОВАНИЯ
st.header("5. 🧮 Калькулятор прогнозирования")

calc_type = st.radio("Тип расчета:", 
                    ["Прогноз диаметра/содержания", "Определение температуры"],
                    key='calc_type_radio')

if calc_type == "Прогноз диаметра/содержания":
    st.subheader("Прогноз параметров по заданным времени и температуре")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        target_time = st.number_input("Время (часы)", 
                                    value=100.0, min_value=0.0, step=10.0)
    with col2:
        target_temp = st.number_input("Температура (°C)", 
                                     value=800.0, min_value=0.0, step=10.0)
    with col3:
        calc_mode = st.selectbox("Рассчитать:", 
                               ["Диаметр", "Содержание фазы", "Оба параметра"])
    
    if st.button("Рассчитать прогноз"):
        current_grain = st.session_state.get('current_grain', target_grain)
        
        if target_temp < min_temperature:
            st.warning(f"⚠️ Температура {target_temp}°C ниже минимальной {min_temperature}°C")
        elif target_temp > dissolution_temperature:
            st.warning(f"⚠️ Температура {target_temp}°C выше температуры растворения {dissolution_temperature}°C")
        else:
            st.success(f"✅ Температура {target_temp}°C в рабочем диапазоне")
            
            universal_diameter_params = st.session_state.get('current_universal_diameter_params')
            universal_phase_params = st.session_state.get('current_universal_phase_params')
            best_n = st.session_state.get('current_best_n')
            
            if calc_mode in ["Диаметр", "Оба параметра"] and universal_diameter_params is not None and best_n is not None:
                A_diam, Ea_diam = universal_diameter_params
                predicted_diameter = universal_diameter_model_single(
                    target_time, target_temp, A_diam, Ea_diam, best_n, initial_diameter, 
                    min_temperature, dissolution_temperature
                )
                st.success(f"**Прогнозируемый диаметр для зерна №{current_grain}:** {predicted_diameter:.2f} мкм")
            
            if calc_mode in ["Содержание фазы", "Оба параметра"] and universal_phase_params is not None:
                A_phase, Ea_phase, n_phase = universal_phase_params
                predicted_phase = universal_phase_model_single(
                    target_time, target_temp, A_phase, Ea_phase, n_phase, 
                    min_temperature, dissolution_temperature
                )
                st.success(f"**Прогнозируемое содержание фазы для зерна №{current_grain}:** {predicted_phase:.1f}%")

st.header("🎯 Рекомендации по использованию моделей")

st.markdown(f"""
**Универсальные модели с учетом температурных ограничений позволяют:**

1. **Точно прогнозировать** поведение системы в реальных условиях
2. **Учитывать физические ограничения** процесса:
   - Ниже {min_temperature}°C: превращение не начинается
   - Выше {dissolution_temperature}°C: σ-фаза растворяется
   - {min_temperature}°C - {dissolution_temperature}°C: нормальный рост

**Новая модель температуры T = k·(c/t^0.5)^n:**
- Описывает зависимость температуры от содержания фазы и времени
- Позволяет прогнозировать температурные режимы для заданного содержания фазы
- Полезна для оптимизации технологических процессов

**Поддержка произвольных зерен:**
- Автоматическое определение доступных зерен из данных
- Возможность выбора зерна из списка или ввода вручную
- Параметры моделей сохраняются отдельно для каждого зерна
""")
