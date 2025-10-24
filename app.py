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
- Аррениусовская зависимость констант скорости
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

# НОВЫЕ ФУНКЦИИ ДЛЯ УНИВЕРСАЛЬНОЙ МОДЕЛИ С УЧЕТОМ ОБОИХ ТЕМПЕРАТУРНЫХ ОГРАНИЧЕНИЙ
def arrhenius_model(T, A, Ea):
    """Аррениусовская модель: k = A * exp(-Ea/(R*T))"""
    return A * np.exp(-Ea / (R * T))

def effective_rate_constant(T, A, Ea, T_min, T_diss):
    """Эффективная константа скорости с учетом минимальной температуры и температуры растворения"""
    T_kelvin = T + 273.15  # Перевод в Кельвины
    T_min_kelvin = T_min + 273.15
    T_diss_kelvin = T_diss + 273.15
    
    if T_kelvin <= T_min_kelvin:
        return 0.0  # Ниже минимальной температуры процесс не идет
    elif T_kelvin >= T_diss_kelvin:
        return 0.0  # Выше температуры растворения фаза растворяется
    else:
        return arrhenius_model(T_kelvin, A, Ea)

def universal_diameter_model(t, T, A, Ea, n, d0, T_min, T_diss):
    """Универсальная модель для диаметра с учетом температурных ограничений"""
    k_eff = effective_rate_constant(T, A, Ea, T_min, T_diss)
    return (k_eff * t + d0**n)**(1/n)

def universal_phase_model(t, T, A, Ea, n_jmak, T_min, T_diss):
    """Универсальная модель для содержания фазы с учетом температурных ограничений"""
    k_eff = effective_rate_constant(T, A, Ea, T_min, T_diss)
    return jmak_model(t, k_eff, n_jmak) * 100

def fit_universal_diameter_model(df, best_n, d0, T_min, T_diss):
    """Подбор параметров универсальной модели для диаметра с учетом температурных ограничений"""
    try:
        # Фильтруем данные в рабочем диапазоне температур
        df_filtered = df[(df['T'] >= T_min) & (df['T'] <= T_diss)].copy()
        
        if len(df_filtered) == 0:
            st.warning("❌ Нет данных в рабочем диапазоне температур для подбора модели")
            return None, None
        
        # Собираем все данные
        t_all = []
        T_all = []
        d_all = []
        
        for temp in df_filtered['T'].unique():
            temp_data = df_filtered[df_filtered['T'] == temp]
            t_all.extend(temp_data['t'].values)
            T_all.extend([temp] * len(temp_data))
            d_all.extend(temp_data['d'].values)
        
        t_all = np.array(t_all)
        T_all = np.array(T_all)
        d_all = np.array(d_all)
        
        # Начальные приближения
        A_guess = 1.0
        Ea_guess = 100000  # 100 кДж/моль
        
        # Подгонка
        popt, pcov = curve_fit(
            lambda x, A, Ea: universal_diameter_model(x[0], x[1], A, Ea, best_n, d0, T_min, T_diss),
            [t_all, T_all], d_all,
            p0=[A_guess, Ea_guess],
            bounds=([1e-10, 10000], [1e10, 1000000]),
            maxfev=10000
        )
        
        return popt, pcov
    except Exception as e:
        st.error(f"Ошибка подбора универсальной модели диаметра: {e}")
        return None, None

def fit_universal_phase_model(df, T_min, T_diss):
    """Подбор параметров универсальной модели для содержания фазы с учетом температурных ограничений"""
    try:
        # Фильтруем данные в рабочем диапазоне температур
        df_filtered = df[(df['T'] >= T_min) & (df['T'] <= T_diss)].copy()
        
        if len(df_filtered) == 0:
            st.warning("❌ Нет данных в рабочем диапазоне температур для подбора JMAK-модели")
            return None, None
        
        t_all = []
        T_all = []
        f_all = []
        
        for temp in df_filtered['T'].unique():
            temp_data = df_filtered[df_filtered['T'] == temp]
            t_all.extend(temp_data['t'].values)
            T_all.extend([temp] * len(temp_data))
            f_all.extend(temp_data['f'].values)
        
        t_all = np.array(t_all)
        T_all = np.array(T_all)
        f_all = np.array(f_all)
        
        # Начальные приближения
        A_guess = 1.0
        Ea_guess = 100000
        n_guess = 1.0
        
        # Подгонка
        popt, pcov = curve_fit(
            lambda x, A, Ea, n: universal_phase_model(x[0], x[1], A, Ea, n, T_min, T_diss),
            [t_all, T_all], f_all,
            p0=[A_guess, Ea_guess, n_guess],
            bounds=([1e-10, 10000, 0.1], [1e10, 1000000, 4]),
            maxfev=10000
        )
        
        return popt, pcov
    except Exception as e:
        st.error(f"Ошибка подбора универсальной модели фазы: {e}")
        return None, None

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
    
    # Подбор показателя степени n
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
            
            # Визуализация универсальной модели
            st.subheader("Визуализация универсальной модели диаметра")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # График 1: Предсказания vs экспериментальные данные
            all_predictions_diam = []
            all_actual_diam = []
            
            for temp in df_grain10['T'].unique():
                temp_data = df_grain10[df_grain10['T'] == temp]
                t_temp = temp_data['t'].values
                T_temp = np.array([temp] * len(t_temp))
                
                # Предсказания универсальной модели
                d_pred_universal = universal_diameter_model(t_temp, T_temp, A_diam, Ea_diam, best_n, initial_diameter, min_temperature, dissolution_temperature)
                
                # Определяем цвет и маркер в зависимости от температурной зоны
                if temp < min_temperature:
                    color, marker, label_suffix = 'red', 'x', ' (ниже T_min)'
                elif temp > dissolution_temperature:
                    color, marker, label_suffix = 'orange', '^', ' (выше T_diss)'
                else:
                    color, marker, label_suffix = 'blue', 'o', ''
                
                axes[0].scatter(temp_data['t'], temp_data['d'], alpha=0.7, 
                               color=color, marker=marker, s=50,
                               label=f'{temp}°C{label_suffix}')
                
                # Строим линии только для рабочего диапазона
                if min_temperature <= temp <= dissolution_temperature:
                    axes[0].plot(temp_data['t'], d_pred_universal, '--', 
                                color=color, linewidth=2)
                
                all_predictions_diam.extend(d_pred_universal)
                all_actual_diam.extend(temp_data['d'].values)
            
            # Добавляем линии температурных границ
            axes[0].axhline(initial_diameter, color='gray', linestyle=':', alpha=0.7, label=f'Начальный диаметр {initial_diameter} мкм')
            axes[0].set_xlabel('Время (часы)')
            axes[0].set_ylabel('Диаметр (мкм)')
            axes[0].set_title(f'Универсальная модель диаметра\nT_min = {min_temperature}°C, T_diss = {dissolution_temperature}°C')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # График 2: Качество предсказаний (только для рабочего диапазона)
            valid_mask = np.array([min_temperature <= T <= dissolution_temperature for T in df_grain10['T'].values])
            valid_actual = np.array(all_actual_diam)[valid_mask]
            valid_predictions = np.array(all_predictions_diam)[valid_mask]
            
            if len(valid_actual) > 0:
                axes[1].scatter(valid_actual, valid_predictions, alpha=0.6, color='blue')
                min_val = min(min(valid_actual), min(valid_predictions))
                max_val = max(max(valid_actual), max(valid_predictions))
                axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                axes[1].set_xlabel('Фактические диаметры (мкм)')
                axes[1].set_ylabel('Предсказанные диаметры (мкм)')
                axes[1].set_title('Качество универсальной модели диаметра\n(только рабочий диапазон)')
                axes[1].grid(True, alpha=0.3)
                
                # Метрики качества (только для валидных температур)
                metrics_universal_diam = calculate_comprehensive_metrics(valid_actual, valid_predictions)
                axes[1].text(0.05, 0.95, f"R² = {metrics_universal_diam['R²']:.3f}\nRMSE = {metrics_universal_diam['RMSE']:.2f}", 
                            transform=axes[1].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)
    
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
            
            # Визуализация универсальной модели фазы
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            all_predictions_phase = []
            all_actual_phase = []
            
            for temp in df_grain10['T'].unique():
                temp_data = df_grain10[df_grain10['T'] == temp]
                if len(temp_data) >= 2:
                    t_temp = temp_data['t'].values
                    T_temp = np.array([temp] * len(t_temp))
                    
                    f_pred_universal = universal_phase_model(t_temp, T_temp, A_phase, Ea_phase, n_phase, min_temperature, dissolution_temperature)
                    
                    # Определяем цвет и маркер в зависимости от температурной зоны
                    if temp < min_temperature:
                        color, marker, label_suffix = 'red', 'x', ' (ниже T_min)'
                    elif temp > dissolution_temperature:
                        color, marker, label_suffix = 'orange', '^', ' (выше T_diss)'
                    else:
                        color, marker, label_suffix = 'blue', 'o', ''
                    
                    axes[0].scatter(temp_data['t'], temp_data['f'], alpha=0.7, 
                                   color=color, marker=marker, s=50,
                                   label=f'{temp}°C{label_suffix}')
                    
                    # Строим линии только для рабочего диапазона
                    if min_temperature <= temp <= dissolution_temperature:
                        axes[0].plot(temp_data['t'], f_pred_universal, '--', 
                                    color=color, linewidth=2)
                    
                    all_predictions_phase.extend(f_pred_universal)
                    all_actual_phase.extend(temp_data['f'].values)
            
            axes[0].set_xlabel('Время (часы)')
            axes[0].set_ylabel('Содержание фазы (%)')
            axes[0].set_title(f'Универсальная модель содержания фазы\nT_min = {min_temperature}°C, T_diss = {dissolution_temperature}°C')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # График качества (только для рабочего диапазона)
            valid_mask_phase = np.array([min_temperature <= T <= dissolution_temperature for T in df_grain10['T'].values])
            valid_actual_phase = np.array(all_actual_phase)[valid_mask_phase]
            valid_predictions_phase = np.array(all_predictions_phase)[valid_mask_phase]
            
            if len(valid_actual_phase) > 0:
                axes[1].scatter(valid_actual_phase, valid_predictions_phase, alpha=0.6, color='blue')
                axes[1].plot([0, 100], [0, 100], 'r--', linewidth=2)
                axes[1].set_xlabel('Фактическое содержание фазы (%)')
                axes[1].set_ylabel('Предсказанное содержание фазы (%)')
                axes[1].set_title('Качество универсальной модели фазы\n(только рабочий диапазон)')
                axes[1].grid(True, alpha=0.3)
                
                metrics_universal_phase = calculate_comprehensive_metrics(
                    valid_actual_phase, valid_predictions_phase
                )
                axes[1].text(0.05, 0.95, f"R² = {metrics_universal_phase['R²']:.3f}\nRMSE = {metrics_universal_phase['RMSE']:.2f}", 
                            transform=axes[1].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)

    # КАЛЬКУЛЯТОР С УЧЕТОМ ОБОИХ ТЕМПЕРАТУРНЫХ ОГРАНИЧЕНИЙ
    st.header("4. 🧮 Калькулятор прогнозирования")
    
    st.markdown(f"""
    **Используйте рассчитанные универсальные модели для прогнозирования:**
    - **Ниже {min_temperature}°C:** процесс не идет, диаметр = d₀, содержание фазы = 0%
    - **Выше {dissolution_temperature}°C:** σ-фаза растворяется, диаметр = d₀, содержание фазы = 0%
    - **{min_temperature}°C - {dissolution_temperature}°C:** нормальный рост по моделям
    """)
    
    calc_type = st.radio("Тип расчета:", 
                        ["Прогноз диаметра/содержания", "Определение температуры"])
    
    if calc_type == "Прогноз диаметра/содержания":
        col1, col2, col3 = st.columns(3)
        with col1:
            target_time = st.number_input("Время (часы)", value=100.0, min_value=0.0, step=10.0)
        with col2:
            target_temp = st.number_input("Температура (°C)", value=800.0, min_value=0.0, step=10.0)
        with col3:
            calc_mode = st.selectbox("Рассчитать:", ["Диаметр", "Содержание фазы", "Оба параметра"])
        
        if st.button("Рассчитать"):
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
                
                if calc_mode in ["Диаметр", "Оба параметра"] and universal_diameter_params is not None:
                    A_diam, Ea_diam = universal_diameter_params
                    predicted_diameter = universal_diameter_model(
                        target_time, target_temp, A_diam, Ea_diam, best_n, initial_diameter, min_temperature, dissolution_temperature
                    )
                    st.success(f"**Прогнозируемый диаметр:** {predicted_diameter:.2f} мкм")
                
                if calc_mode in ["Содержание фазы", "Оба параметра"] and universal_phase_params is not None:
                    A_phase, Ea_phase, n_phase = universal_phase_params
                    predicted_phase = universal_phase_model(
                        target_time, target_temp, A_phase, Ea_phase, n_phase, min_temperature, dissolution_temperature
                    )
                    st.success(f"**Прогнозируемое содержание фазы:** {predicted_phase:.1f}%")
    
    else:  # Определение температуры
        col1, col2, col3 = st.columns(3)
        with col1:
            target_time_temp = st.number_input("Время (часы) ", value=100.0, min_value=0.0, step=10.0)
        with col2:
            target_value = st.number_input("Целевое значение", value=2.0, min_value=0.0, step=0.1)
        with col3:
            temp_mode = st.selectbox("Тип целевого значения:", ["Диаметр (мкм)", "Содержание фазы (%)"])
        
        if st.button("Найти температуру"):
            # Минимальная температура для поиска - в рабочем диапазоне
            search_min = max(400, min_temperature)
            search_max = min(1200, dissolution_temperature)
            
            if search_min >= search_max:
                st.error("❌ Рабочий диапазон температур пуст. Проверьте настройки T_min и T_diss.")
            else:
                if temp_mode == "Диаметр (мкм)" and universal_diameter_params is not None:
                    A_diam, Ea_diam = universal_diameter_params
                    
                    def equation(T):
                        k = effective_rate_constant(T, A_diam, Ea_diam, min_temperature, dissolution_temperature)
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
                        st.warning("Не удалось найти решение в рабочем диапазоне температур")
                
                elif temp_mode == "Содержание фазы (%)" and universal_phase_params is not None:
                    A_phase, Ea_phase, n_phase = universal_phase_params
                    
                    def equation_phase(T):
                        k = effective_rate_constant(T, A_phase, Ea_phase, min_temperature, dissolution_temperature)
                        return jmak_model(target_time_temp, k, n_phase) * 100 - target_value
                    
                    T_candidates = np.linspace(search_min, search_max, 1000)
                    differences = [equation_phase(T) for T in T_candidates]
                    
                    idx_min = np.argmin(np.abs(differences))
                    optimal_temp = T_candidates[idx_min]
                    
                    if np.abs(differences[idx_min]) < 1.0:  # Допустимая погрешность 1%
                        st.success(f"**Необходимая температура:** {optimal_temp:.1f}°C")
                        st.info(f"При {optimal_temp:.1f}°C за {target_time_temp} часов содержание фазы достигнет {target_value}%")
                    else:
                        st.warning("Не удалось найти решение в рабочем диапазоне температур")

    # Визуализация температурных зависимостей
    st.header("5. 📈 Температурные зависимости")
    
    if universal_diameter_params is not None or universal_phase_params is not None:
        st.subheader("Зависимость константы скорости от температуры")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График в координатах Аррениуса
        T_range_celsius = np.linspace(min_temperature - 50, dissolution_temperature + 50, 200)
        T_range_kelvin = T_range_celsius + 273
        
        if universal_diameter_params is not None:
            A_diam, Ea_diam = universal_diameter_params
            k_diam = [effective_rate_constant(T, A_diam, Ea_diam, min_temperature, dissolution_temperature) for T in T_range_celsius]
            ax1.semilogy(1000/T_range_kelvin, k_diam, 'b-', linewidth=2, label='Диаметр')
        
        if universal_phase_params is not None:
            A_phase, Ea_phase, n_phase = universal_phase_params
            k_phase = [effective_rate_constant(T, A_phase, Ea_phase, min_temperature, dissolution_temperature) for T in T_range_celsius]
            ax1.semilogy(1000/T_range_kelvin, k_phase, 'r-', linewidth=2, label='Содержание фазы')
        
        # Вертикальные линии границ
        ax1.axvline(1000/(min_temperature + 273), color='gray', linestyle='--', alpha=0.7, label=f'T_min = {min_temperature}°C')
        ax1.axvline(1000/(dissolution_temperature + 273), color='gray', linestyle='--', alpha=0.7, label=f'T_diss = {dissolution_temperature}°C')
        ax1.set_xlabel('1000/T (K⁻¹)')
        ax1.set_ylabel('Константа скорости k')
        ax1.set_title('Координаты Аррениуса')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График в обычных координатах
        if universal_diameter_params is not None:
            ax2.plot(T_range_celsius, k_diam, 'b-', linewidth=2, label='Диаметр')
        if universal_phase_params is not None:
            ax2.plot(T_range_celsius, k_phase, 'r-', linewidth=2, label='Содержание фазы')
        
        # Вертикальные линии и заливка рабочей области
        ax2.axvline(min_temperature, color='gray', linestyle='--', alpha=0.7, label=f'T_min = {min_temperature}°C')
        ax2.axvline(dissolution_temperature, color='gray', linestyle='--', alpha=0.7, label=f'T_diss = {dissolution_temperature}°C')
        ax2.axvspan(min_temperature, dissolution_temperature, alpha=0.1, color='green', label='Рабочий диапазон')
        
        ax2.set_xlabel('Температура (°C)')
        ax2.set_ylabel('Константа скорости k')
        ax2.set_title('Температурная зависимость')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

st.header("🎯 Рекомендации по использованию моделей")

st.markdown(f"""
**Универсальные модели с учетом температурных ограничений позволяют:**

1. **Точно прогнозировать** поведение системы в реальных условиях
2. **Учитывать физические ограничения** процесса:
   - Ниже {min_temperature}°C: превращение не начинается
   - Выше {dissolution_temperature}°C: σ-фаза растворяется
   - {min_temperature}°C - {dissolution_temperature}°C: нормальный рост

**Для промышленного применения:**
- Используйте калькулятор для определения оптимальных режимов термообработки
- Избегайте температур выше {dissolution_temperature}°C для сохранения σ-фазы
- Учитывайте, что при T < {min_temperature}°C процесс не идет даже при длительных выдержках

**Температурные зоны на графиках:**
- 🔴 Красные крестики: T < {min_temperature}°C (процесс не идет)
- 🔵 Синие кружки: рабочий диапазон (нормальный рост)
- 🟠 Оранжевые треугольники: T > {dissolution_temperature}°C (растворение)
""")
