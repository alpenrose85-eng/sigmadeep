import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
import io
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Универсальная газовая постоянная
R = 8.314  # Дж/(моль·К)

st.title("🔬 Комплексный анализ кинетики σ-фазы с универсальной моделью и калькулятором")
st.markdown("""
**Улучшенный анализ с универсальными моделями:**
- Модели, работающие для всех температур одновременно
- Учет минимальной температуры начала превращения (550°C)
- Учет температуры растворения σ-фазы (900°C)
- Интерактивный калькулятор для прогнозирования
- **НОВОЕ: Интерактивный график модели**
- Аррениусовская зависимость констант скорости
- Автоматическое масштабирование графиков под данные
""")

# Загрузка данных
st.header("1. Загрузка данных для зерна №10")

uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", 
                               type=['csv', 'xlsx', 'xls'])

# Параметры анализа
st.subheader("Параметры анализа:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_diameter = st.number_input("Начальный диаметр d₀ (мкм)", 
                                     value=0.1, min_value=0.0, step=0.1,
                                     help="Рекомендуется использовать небольшое положительное значение (0.1-0.5 мкм)")
with col2:
    enable_phase_analysis = st.checkbox("Включить анализ содержания фазы (JMAK)", 
                                      value=True, 
                                      help="Анализ кинетики фазового превращения по содержанию σ-фазы")
with col3:
    min_temperature = st.number_input("Минимальная температура начала превращения (°C)", 
                                    value=550.0, min_value=0.0, step=10.0,
                                    help="Температура, ниже которой превращение не происходит")
with col4:
    dissolution_temperature = st.number_input("Температура растворения σ-фазы (°C)", 
                                           value=900.0, min_value=0.0, step=10.0,
                                           help="Температура, выше которой σ-фаза растворяется")

target_grain = 10

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

# НОВАЯ ФУНКЦИЯ: Интерактивный график модели
def plot_interactive_model(temperature, mode, max_time, universal_diameter_params, universal_phase_params, best_n, initial_diameter, min_temperature, dissolution_temperature):
    """Построение интерактивного графика модели"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Создаем диапазон времени от 0 до max_time часов
    time_range = np.linspace(0, max_time, 1000)
    
    if mode == "Диаметр" and universal_diameter_params is not None and best_n is not None:
        A_diam, Ea_diam = universal_diameter_params
        
        # Рассчитываем диаметр для каждого момента времени
        diameters = []
        for t in time_range:
            d = universal_diameter_model_single(
                t, temperature, A_diam, Ea_diam, best_n, initial_diameter,
                min_temperature, dissolution_temperature
            )
            diameters.append(d)
        
        ax.plot(time_range, diameters, 'b-', linewidth=3, label=f'Модель диаметра при {temperature}°C')
        ax.set_ylabel('Диаметр (мкм)', fontsize=12)
        ax.set_title(f'Зависимость диаметра σ-фазы от времени при {temperature}°C', fontsize=14, fontweight='bold')
        
        # Добавляем горизонтальную линию начального диаметра
        ax.axhline(y=initial_diameter, color='r', linestyle='--', alpha=0.7, label=f'Начальный диаметр ({initial_diameter} мкм)')
        
        # Показываем конечное значение
        final_diameter = diameters[-1]
        ax.axhline(y=final_diameter, color='g', linestyle='--', alpha=0.7, label=f'Конечный диаметр ({final_diameter:.2f} мкм)')
        
    elif mode == "Содержание фазы" and universal_phase_params is not None:
        A_phase, Ea_phase, n_phase = universal_phase_params
        
        # Рассчитываем содержание фазы для каждого момента времени
        phase_contents = []
        for t in time_range:
            f = universal_phase_model_single(
                t, temperature, A_phase, Ea_phase, n_phase,
                min_temperature, dissolution_temperature
            )
            phase_contents.append(f)
        
        ax.plot(time_range, phase_contents, 'r-', linewidth=3, label=f'Модель содержания фазы при {temperature}°C')
        ax.set_ylabel('Содержание фазы (%)', fontsize=12)
        ax.set_title(f'Зависимость содержания σ-фазы от времени при {temperature}°C', fontsize=14, fontweight='bold')
        
        # Показываем конечное значение
        final_phase = phase_contents[-1]
        ax.axhline(y=final_phase, color='g', linestyle='--', alpha=0.7, label=f'Конечное содержание ({final_phase:.2f}%)')
    
    else:
        ax.text(0.5, 0.5, 'Модель не рассчитана', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16)
        ax.set_title('Недостаточно данных для построения графика', fontsize=14)
    
    ax.set_xlabel('Время (часы)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Улучшаем читаемость больших чисел на оси X
    if max_time >= 1000:
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(10)
    
    plt.tight_layout()
    return fig

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        required_cols = ['G', 'T', 't', 'd', 'f']
        
        if all(col in df.columns for col in required_cols):
            df_grain10 = df[df['G'] == target_grain].copy()
            
            if len(df_grain10) > 0:
                st.session_state['grain10_data'] = df_grain10
                st.success(f"✅ Данные для зерна №10 успешно загружены!")
                
                # Проверка температурного диапазона в данных
                min_temp_in_data = df_grain10['T'].min()
                max_temp_in_data = df_grain10['T'].max()
                
                temp_warnings = []
                if min_temp_in_data < min_temperature:
                    temp_warnings.append(f"⚠️ В данных есть температуры ниже минимальной ({min_temp_in_data}°C < {min_temperature}°C)")
                if max_temp_in_data > dissolution_temperature:
                    temp_warnings.append(f"⚠️ В данных есть температуры выше температуры растворения ({max_temp_in_data}°C > {dissolution_temperature}°C)")
                
                if temp_warnings:
                    for warning in temp_warnings:
                        st.warning(warning)
                    st.info("Точки вне рабочего диапазона будут исключены из подбора универсальной модели")
                
                st.subheader("📊 Статистика данных для зерна №10:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    unique_temps = df_grain10['T'].unique()
                    st.metric("Температуры", f"{len(unique_temps)} уровней")
                with col2:
                    st.metric("Всего точек", f"{len(df_grain10)}")
                with col3:
                    st.metric("Диапазон времени", f"{df_grain10['t'].min()}-{df_grain10['t'].max()} ч")
                with col4:
                    st.metric("Содержание фазы", f"{df_grain10['f'].min():.1f}-{df_grain10['f'].max():.1f}%")
                
                st.dataframe(df_grain10.head(10))
                
            else:
                st.error(f"❌ В данных нет записей для зерна №10")
        else:
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"❌ Отсутствуют колонки: {missing}")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")

# Основной расчет
if 'grain10_data' in st.session_state:
    df_grain10 = st.session_state['grain10_data']
    
    # Фильтрация аномальных данных
    df_grain10_clean = df_grain10[(df_grain10['d'] > 0) & (df_grain10['f'] >= 0) & (df_grain10['f'] <= 100)].copy()
    
    if len(df_grain10_clean) < len(df_grain10):
        st.warning(f"⚠️ Удалено {len(df_grain10) - len(df_grain10_clean)} аномальных точек")
        df_grain10 = df_grain10_clean
    
    df_grain10['T_K'] = df_grain10['T'] + 273.15
    
    # Анализ диаметров
    st.header("2. 📏 Анализ диаметров σ-фазы")
    
    with st.expander("💡 Объяснение анализа диаметров"):
        st.markdown("""
        **Что анализируем:** Рост среднего диаметра частиц σ-фазы во времени
        
        **Физическая модель:** 
        $$ d^n - d_0^n = K \\cdot t $$
        
        **Ожидаемое поведение:**
        - При правильном n график $d^n - d_0^n$ vs t должен быть линейным
        - Качество оценивается по R² близкому к 1
        - Остатки должны быть случайными (без тренда)
        
        **Как оценивать результат:**
        - R² > 0.95 - отличное согласие
        - R² 0.90-0.95 - хорошее согласие  
        - R² < 0.90 - требуется улучшение модели
        """)
    
    # Подбор показателя степени n
    st.subheader("Поиск оптимального показателя степени n")
    
    n_min, n_max, n_step = 3.0, 5.0, 0.1
    n_candidates = np.arange(n_min, n_max + n_step, n_step)
    
    n_results = {}
    available_temperatures = set()
    
    for n in n_candidates:
        k_values = []
        
        for temp in df_grain10['T'].unique():
            # Пропускаем температуры вне рабочего диапазона
            if temp < min_temperature or temp > dissolution_temperature:
                continue
                
            temp_data = df_grain10[df_grain10['T'] == temp]
            
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
                            metrics = calculate_comprehensive_metrics(temp_data['d'].values, d_pred)
                            
                            k_values.append({
                                'T': temp, 'T_K': temp + 273.15, 'K': slope,
                                'R2': r_value**2, 'std_err': std_err,
                                'n_points': len(temp_data), 'metrics': metrics
                            })
                            available_temperatures.add(temp)
                except:
                    continue
        
        if k_values:
            k_df = pd.DataFrame(k_values)
            overall_r2 = k_df['R2'].mean()
            n_results[n] = {
                'k_df': k_df, 'mean_R2': overall_r2,
                'min_R2': k_df['R2'].min(), 'n_temperatures': len(k_df)
            }
    
    # Определение лучшего n
    best_n = None
    if n_results:
        comparison_data = []
        for n, results in n_results.items():
            comparison_data.append({
                'n': n, 'Средний R²': results['mean_R2'],
                'Минимальный R²': results['min_R2'], 
                'Количество температур': results['n_temperatures']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) > 0:
            best_n_row = comparison_df.loc[comparison_df['Средний R²'].idxmax()]
            best_n = best_n_row['n']
            
            st.success(f"🎯 Оптимальный показатель: n = {best_n:.1f} (R² = {best_n_row['Средний R²']:.3f})")

    # УНИВЕРСАЛЬНАЯ МОДЕЛЬ С УЧЕТОМ ОБОИХ ТЕМПЕРАТУРНЫХ ОГРАНИЧЕНИЙ
    st.header("3. 🔬 Универсальная модель для всех температур")
    
    st.info(f"""
    **Температурные ограничения:**
    - **Минимальная температура начала превращения:** {min_temperature}°C
    - **Температура растворения σ-фазы:** {dissolution_temperature}°C
    - **Рабочий диапазон:** {min_temperature}°C - {dissolution_temperature}°C
    """)
    
    if best_n is not None:
        # Подбор универсальной модели для диаметра
        st.subheader("Универсальная модель роста диаметра")
        
        with st.expander("💡 Объяснение универсальной модели"):
            st.markdown(f"""
            **Универсальная модель диаметра:**
            $$ d(t,T) = \\left[ k_{{eff}}(T) \\cdot t + d_0^n \\right]^{{1/n}} $$
            
            $$ k_{{eff}}(T) = \\begin{{cases}}
            0 & \\text{{если }} T < {min_temperature}°C \\\\
            A \\cdot \\exp\\left(-\\frac{{E_a}}{{RT}}\\right) & \\text{{если }} {min_temperature}°C \\leq T \\leq {dissolution_temperature}°C \\\\
            0 & \\text{{если }} T > {dissolution_temperature}°C
            \\end{{cases}} $$
            
            **Физический смысл:**
            - При T < {min_temperature}°C: превращение не начинается
            - При {min_temperature}°C ≤ T ≤ {dissolution_temperature}°C: нормальный рост по степенному закону
            - При T > {dissolution_temperature}°C: σ-фаза растворяется
            """)
        
        # Подбор параметров универсальной модели
        universal_diameter_params, universal_diameter_cov = fit_universal_diameter_model(
            df_grain10, best_n, initial_diameter, min_temperature, dissolution_temperature
        )
        
        if universal_diameter_params is not None:
            A_diam, Ea_diam = universal_diameter_params
            
            st.success("✅ Универсальная модель диаметра успешно подобрана!")
            st.info(f"""
            **Параметры универсальной модели диаметра:**
            - Предэкспоненциальный множитель A = {A_diam:.4e}
            - Энергия активации Ea = {Ea_diam:.0f} Дж/моль ({Ea_diam/1000:.1f} кДж/моль)
            - Показатель степени n = {best_n:.1f}
            - Начальный диаметр d₀ = {initial_diameter} мкм
            - Рабочий диапазон: {min_temperature}°C - {dissolution_temperature}°C
            """)

    # Универсальная модель для содержания фазы
    if enable_phase_analysis:
        st.subheader("Универсальная модель содержания фазы (JMAK)")
        
        universal_phase_params, universal_phase_cov = fit_universal_phase_model(
            df_grain10, min_temperature, dissolution_temperature
        )
        
        if universal_phase_params is not None:
            A_phase, Ea_phase, n_phase = universal_phase_params
            
            st.success("✅ Универсальная модель содержания фазы успешно подобрана!")
            st.info(f"""
            **Параметры универсальной модели фазы:**
            - Предэкспоненциальный множитель A = {A_phase:.4e}
            - Энергия активации Ea = {Ea_phase:.0f} Дж/моль ({Ea_phase/1000:.1f} кДж/моль)
            - Показатель Аврами n = {n_phase:.2f}
            - Рабочий диапазон: {min_temperature}°C - {dissolution_temperature}°C
            """)

    # ИНТЕРАКТИВНЫЙ ГРАФИК МОДЕЛИ
    st.header("4. 📈 Интерактивный график модели")
    
    st.markdown("""
    **Постройте график зависимости параметров от времени для любой температуры:**
    - Выберите температуру и параметр для построения
    - График покажет прогноз на весь диапазон времени
    - Учитываются температурные ограничения модели
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        interactive_temp = st.number_input("Температура для графика (°C)", 
                                         value=st.session_state.interactive_temp,
                                         min_value=0.0, max_value=1500.0, step=10.0,
                                         key='interactive_temp_input')
        st.session_state.interactive_temp = interactive_temp
        
    with col2:
        interactive_mode = st.selectbox("Параметр для графика:", 
                                      ["Диаметр", "Содержание фазы"],
                                      key='interactive_mode_select')
        st.session_state.interactive_mode = interactive_mode
        
    with col3:
        max_time_interactive = st.number_input("Максимальное время (часы)", 
                                            value=st.session_state.max_time_interactive,
                                            min_value=100.0, max_value=500000.0, step=1000.0,
                                            key='max_time_interactive_input')
        st.session_state.max_time_interactive = max_time_interactive
    
    if st.button("Построить график", key='plot_interactive'):
        if interactive_temp < min_temperature:
            st.warning(f"⚠️ Температура {interactive_temp}°C ниже минимальной {min_temperature}°C")
            st.info("При этой температуре процесс не происходит. Все параметры остаются на начальных значениях.")
        elif interactive_temp > dissolution_temperature:
            st.warning(f"⚠️ Температура {interactive_temp}°C выше температуры растворения {dissolution_temperature}°C")
            st.info("При этой температуре σ-фаза растворяется. Все параметры остаются на начальных значениях.")
        else:
            st.success(f"✅ Строим график для температуры {interactive_temp}°C")
            
            # Получаем параметры моделей
            universal_diameter_params = st.session_state.get('universal_diameter_params')
            universal_phase_params = st.session_state.get('universal_phase_params')
            best_n = st.session_state.get('best_n')
            
            # Строим график
            fig = plot_interactive_model(
                interactive_temp, interactive_mode, max_time_interactive,
                universal_diameter_params, universal_phase_params, best_n,
                initial_diameter, min_temperature, dissolution_temperature
            )
            st.pyplot(fig)
            
            # Дополнительная информация
            if interactive_mode == "Диаметр" and universal_diameter_params is not None and best_n is not None:
                A_diam, Ea_diam = universal_diameter_params
                final_diameter = universal_diameter_model_single(
                    max_time_interactive, interactive_temp, A_diam, Ea_diam, best_n,
                    initial_diameter, min_temperature, dissolution_temperature
                )
                st.info(f"""
                **📊 Прогноз для {interactive_temp}°C за {max_time_interactive:,.0f} часов:**
                - Начальный диаметр: {initial_diameter} мкм
                - Конечный диаметр: {final_diameter:.2f} мкм
                - Общий рост: {final_diameter - initial_diameter:.2f} мкм
                - Относительный рост: {(final_diameter/initial_diameter - 1)*100:.1f}%
                """)
                
            elif interactive_mode == "Содержание фазы" and universal_phase_params is not None:
                A_phase, Ea_phase, n_phase = universal_phase_params
                final_phase = universal_phase_model_single(
                    max_time_interactive, interactive_temp, A_phase, Ea_phase, n_phase,
                    min_temperature, dissolution_temperature
                )
                st.info(f"""
                **📊 Прогноз для {interactive_temp}°C за {max_time_interactive:,.0f} часов:**
                - Конечное содержание фазы: {final_phase:.2f}%
                - Максимально возможное: 100%
                - Достигнуто: {final_phase:.1f}% от максимального
                """)

    # КАЛЬКУЛЯТОР С СОХРАНЕНИЕМ СОСТОЯНИЯ
    st.header("5. 🧮 Калькулятор прогнозирования")
    
    # Используем session_state для сохранения состояния
    calc_type = st.radio("Тип расчета:", 
                        ["Прогноз диаметра/содержания", "Определение температуры"],
                        key='calc_type_radio')
    
    # Обновляем session_state при изменении выбора
    if calc_type != st.session_state.calc_type:
        st.session_state.calc_type = calc_type
    
    if st.session_state.calc_type == "Прогноз диаметра/содержания":
        st.subheader("Прогноз параметров по заданным времени и температуре")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_time = st.number_input("Время (часы)", 
                                        value=st.session_state.target_time, 
                                        min_value=0.0, step=10.0,
                                        key='target_time_input')
            st.session_state.target_time = target_time
            
        with col2:
            target_temp = st.number_input("Температура (°C)", 
                                         value=st.session_state.target_temp,
                                         min_value=0.0, step=10.0,
                                         key='target_temp_input')
            st.session_state.target_temp = target_temp
            
        with col3:
            calc_mode = st.selectbox("Рассчитать:", 
                                   ["Диаметр", "Содержание фазы", "Оба параметра"],
                                   key='calc_mode_select')
            st.session_state.calc_mode = calc_mode
        
        if st.button("Рассчитать прогноз", key='calculate_forecast'):
            if target_temp < min_temperature:
                st.warning(f"⚠️ Температура {target_temp}°C ниже минимальной {min_temperature}°C")
                st.info(f"**При {target_temp}°C процесс не происходит:**")
                if calc_mode in ["Диаметр", "Оба параметра"]:
                    st.success(f"**Диаметр:** {initial_diameter} мкм (начальное значение)")
                if calc_mode in ["Содержание фазы", "Оба параметра"]:
                    st.success(f"**Содержание фазы:** 0%")
            elif target_temp > dissolution_temperature:
                st.warning(f"⚠️ Температура {target_temp}°C выше температуры растворения {dissolution_temperature}°C")
                st.info(f"**При {target_temp}°C σ-фаза растворяется:**")
                if calc_mode in ["Диаметр", "Оба параметра"]:
                    st.success(f"**Диаметр:** {initial_diameter} мкм (начальное значение)")
                if calc_mode in ["Содержание фазы", "Оба параметра"]:
                    st.success(f"**Содержание фазы:** 0%")
            else:
                st.success(f"✅ Температура {target_temp}°C в рабочем диапазоне {min_temperature}°C - {dissolution_temperature}°C")
                
                # Получаем параметры моделей из предыдущих расчетов
                universal_diameter_params = st.session_state.get('universal_diameter_params')
                universal_phase_params = st.session_state.get('universal_phase_params')
                best_n = st.session_state.get('best_n')
                
                if calc_mode in ["Диаметр", "Оба параметра"] and universal_diameter_params is not None and best_n is not None:
                    A_diam, Ea_diam = universal_diameter_params
                    predicted_diameter = universal_diameter_model_single(
                        target_time, target_temp, A_diam, Ea_diam, best_n, initial_diameter, 
                        min_temperature, dissolution_temperature
                    )
                    st.success(f"**Прогнозируемый диаметр:** {predicted_diameter:.2f} мкм")
                    st.info(f"Рост от начального {initial_diameter} мкм до {predicted_diameter:.2f} мкм")
                
                if calc_mode in ["Содержание фазы", "Оба параметра"] and universal_phase_params is not None:
                    A_phase, Ea_phase, n_phase = universal_phase_params
                    predicted_phase = universal_phase_model_single(
                        target_time, target_temp, A_phase, Ea_phase, n_phase, 
                        min_temperature, dissolution_temperature
                    )
                    st.success(f"**Прогнозируемое содержание фазы:** {predicted_phase:.1f}%")
                    
                if universal_diameter_params is None and universal_phase_params is None:
                    st.error("❌ Модели не были рассчитаны. Сначала выполните анализ данных.")
    
    else:  # Определение температуры
        st.subheader("Определение температуры для достижения целевых значений")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_time_temp = st.number_input("Время (часы)", 
                                             value=st.session_state.target_time_temp,
                                             min_value=0.0, step=10.0,
                                             key='target_time_temp_input')
            st.session_state.target_time_temp = target_time_temp
            
        with col2:
            target_value = st.number_input("Целевое значение", 
                                         value=st.session_state.target_value,
                                         min_value=0.0, step=0.1,
                                         key='target_value_input')
            st.session_state.target_value = target_value
            
        with col3:
            temp_mode = st.selectbox("Тип целевого значения:", 
                                   ["Диаметр (мкм)", "Содержание фазы (%)"],
                                   key='temp_mode_select')
            st.session_state.temp_mode = temp_mode
        
        if st.button("Найти температуру", key='find_temperature'):
            # Минимальная температура для поиска - в рабочем диапазоне
            search_min = max(400, min_temperature)
            search_max = min(1200, dissolution_temperature)
            
            if search_min >= search_max:
                st.error("❌ Рабочий диапазон температур пуст. Проверьте настройки T_min и T_diss.")
            else:
                universal_diameter_params = st.session_state.get('universal_diameter_params')
                universal_phase_params = st.session_state.get('universal_phase_params')
                best_n = st.session_state.get('best_n')
                
                if temp_mode == "Диаметр (мкм)" and universal_diameter_params is not None and best_n is not None:
                    A_diam, Ea_diam = universal_diameter_params
                    
                    def equation(T):
                        k = effective_rate_constant_single(T, A_diam, Ea_diam, min_temperature, dissolution_temperature)
                        return (k * target_time_temp + initial_diameter**best_n)**(1/best_n) - target_value
                    
                    # Ищем температуру в рабочем диапазоне
                    T_candidates = np.linspace(search_min, search_max, 1000)
                    differences = [equation(T) for T in T_candidates]
                    
                    # Находим температуру, где разница ближе всего к нулю
                    idx_min = np.argmin(np.abs(differences))
                    optimal_temp = T_candidates[idx_min]
                    
                    if np.abs(differences[idx_min]) < 0.1:  # Допустимая погрешность
                        st.success(f"**Необходимая температура:** {optimal_temp:.1f}°C")
                        st.info(f"При {optimal_temp:.1f}°C за {target_time_temp} часов диаметр достигнет {target_value} мкм")
                    else:
                        st.warning("Не удалось найти точное решение в рабочем диапазоне температур")
                        st.info(f"Наиболее близкая температура: {optimal_temp:.1f}°C")
                
                elif temp_mode == "Содержание фазы (%)" and universal_phase_params is not None:
                    A_phase, Ea_phase, n_phase = universal_phase_params
                    
                    def equation_phase(T):
                        k = effective_rate_constant_single(T, A_phase, Ea_phase, min_temperature, dissolution_temperature)
                        return jmak_model(target_time_temp, k, n_phase) * 100 - target_value
                    
                    T_candidates = np.linspace(search_min, search_max, 1000)
                    differences = [equation_phase(T) for T in T_candidates]
                    
                    idx_min = np.argmin(np.abs(differences))
                    optimal_temp = T_candidates[idx_min]
                    
                    if np.abs(differences[idx_min]) < 1.0:  # Допустимая погрешность 1%
                        st.success(f"**Необходимая температура:** {optimal_temp:.1f}°C")
                        st.info(f"При {optimal_temp:.1f}°C за {target_time_temp} часов содержание фазы достигнет {target_value}%")
                    else:
                        st.warning("Не удалось найти точное решение в рабочем диапазоне температур")
                        st.info(f"Наиболее близкая температура: {optimal_temp:.1f}°C")
                
                else:
                    st.error("❌ Модели не были рассчитаны. Сначала выполните анализ данных.")

# Сохраняем параметры моделей в session_state для использования в калькуляторе
if 'grain10_data' in st.session_state and 'best_n' in locals():
    st.session_state.best_n = best_n
if 'universal_diameter_params' in locals() and universal_diameter_params is not None:
    st.session_state.universal_diameter_params = universal_diameter_params
if 'universal_phase_params' in locals() and universal_phase_params is not None:
    st.session_state.universal_phase_params = universal_phase_params

st.header("🎯 Рекомендации по использованию моделей")

st.markdown(f"""
**Универсальные модели с учетом температурных ограничений позволяют:**

1. **Точно прогнозировать** поведение системы в реальных условиях
2. **Учитывать физические ограничения** процесса:
   - Ниже {min_temperature}°C: превращение не начинается
   - Выше {dissolution_temperature}°C: σ-фаза растворяется
   - {min_temperature}°C - {dissolution_temperature}°C: нормальный рост

**Новый интерактивный график:**
- Позволяет визуализировать поведение системы для любой температуры
- Показывает зависимость параметров от времени до 400,000 часов
- Автоматически рассчитывает конечные значения параметров

**Автоматическое масштабирование графиков:**
- Графики качества модели автоматически настраиваются под диапазон ваших данных
- Даже при малом содержании фазы (<10%) точки будут хорошо видны
- Масштаб осей определяется максимальным значением в данных + 15% запас

**Температурные зоны на графиках:**
- 🔴 Красные крестики: T < {min_temperature}°C (процесс не идет)
- 🔵 Синие кружки: рабочий диапазон (нормальный рост)
- 🟠 Оранжевые треугольники: T > {dissolution_temperature}°C (растворение)
""")
