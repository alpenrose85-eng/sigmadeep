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
- Интерактивный калькулятор для прогнозирования
- Аррениусовская зависимость констант скорости
""")

# Загрузка данных
st.header("1. Загрузка данных для зерна №10")

uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", 
                               type=['csv', 'xlsx', 'xls'])

# Параметры анализа
st.subheader("Параметры анализа:")
col1, col2 = st.columns(2)
with col1:
    initial_diameter = st.number_input("Начальный диаметр d₀ (мкм)", 
                                     value=0.1, min_value=0.0, step=0.1,
                                     help="Рекомендуется использовать небольшое положительное значение (0.1-0.5 мкм)")
with col2:
    enable_phase_analysis = st.checkbox("Включить анализ содержания фазы (JMAK)", 
                                      value=True, 
                                      help="Анализ кинетики фазового превращения по содержанию σ-фазы")

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

# НОВЫЕ ФУНКЦИИ ДЛЯ УНИВЕРСАЛЬНОЙ МОДЕЛИ
def arrhenius_model(T, A, Ea):
    """Аррениусовская модель: k = A * exp(-Ea/(R*T))"""
    return A * np.exp(-Ea / (R * T))

def universal_diameter_model(t, T, A, Ea, n, d0):
    """Универсальная модель для диаметра с аррениусовской зависимостью"""
    k = arrhenius_model(T + 273.15, A, Ea)  # T в Кельвинах
    return (k * t + d0**n)**(1/n)

def universal_phase_model(t, T, A, Ea, n_jmak):
    """Универсальная модель для содержания фазы с аррениусовской зависимостью"""
    k = arrhenius_model(T + 273.15, A, Ea)
    return jmak_model(t, k, n_jmak) * 100

def fit_universal_diameter_model(df, best_n, d0):
    """Подбор параметров универсальной модели для диаметра"""
    try:
        # Собираем все данные
        t_all = []
        T_all = []
        d_all = []
        
        for temp in df['T'].unique():
            temp_data = df[df['T'] == temp]
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
            lambda x, A, Ea: universal_diameter_model(x[0], x[1], A, Ea, best_n, d0),
            [t_all, T_all], d_all,
            p0=[A_guess, Ea_guess],
            bounds=([1e-10, 10000], [1e10, 1000000]),
            maxfev=10000
        )
        
        return popt, pcov
    except Exception as e:
        st.error(f"Ошибка подбора универсальной модели диаметра: {e}")
        return None, None

def fit_universal_phase_model(df):
    """Подбор параметров универсальной модели для содержания фазы"""
    try:
        t_all = []
        T_all = []
        f_all = []
        
        for temp in df['T'].unique():
            temp_data = df[df['T'] == temp]
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
            lambda x, A, Ea, n: universal_phase_model(x[0], x[1], A, Ea, n),
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
    
    # Анализ диаметров (существующий код)
    st.header("2. 📏 Анализ диаметров σ-фазы")
    
    # ... [существующий код анализа диаметров] ...
    
    # Подбор показателя степени n (существующий код)
    n_min, n_max, n_step = 3.0, 5.0, 0.1
    n_candidates = np.arange(n_min, n_max + n_step, n_step)
    
    n_results = {}
    available_temperatures = set()
    
    for n in n_candidates:
        k_values = []
        
        for temp in df_grain10['T'].unique():
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

    # НОВЫЙ РАЗДЕЛ: УНИВЕРСАЛЬНАЯ МОДЕЛЬ И КАЛЬКУЛЯТОР
    st.header("3. 🔬 Универсальная модель для всех температур")
    
    if best_n is not None:
        # Подбор универсальной модели для диаметра
        st.subheader("Универсальная модель роста диаметра")
        
        with st.expander("💡 Объяснение универсальной модели"):
            st.markdown("""
            **Универсальная модель диаметра:**
            $$ d(t,T) = \\left[ A \\cdot \\exp\\left(-\\frac{E_a}{RT}\\right) \\cdot t + d_0^n \\right]^{1/n} $$
            
            **Параметры:**
            - **A** - предэкспоненциальный множитель
            - **Ea** - энергия активации (Дж/моль)
            - **n** - показатель степени (фиксированный)
            - **d₀** - начальный диаметр (фиксированный)
            
            **Преимущества:**
            - Работает для всех температур одновременно
            - Учитывает температурную зависимость
            - Позволяет прогнозировать для любых условий
            """)
        
        # Подбор параметров универсальной модели
        universal_diameter_params, universal_diameter_cov = fit_universal_diameter_model(
            df_grain10, best_n, initial_diameter
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
                d_pred_universal = universal_diameter_model(t_temp, T_temp, A_diam, Ea_diam, best_n, initial_diameter)
                
                axes[0].scatter(temp_data['t'], temp_data['d'], alpha=0.7, 
                               label=f'{temp}°C (эксп.)', s=50)
                axes[0].plot(temp_data['t'], d_pred_universal, '--', 
                            label=f'{temp}°C (мод.)', linewidth=2)
                
                all_predictions_diam.extend(d_pred_universal)
                all_actual_diam.extend(temp_data['d'].values)
            
            axes[0].set_xlabel('Время (часы)')
            axes[0].set_ylabel('Диаметр (мкм)')
            axes[0].set_title('Универсальная модель диаметра\nдля всех температур')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # График 2: Качество предсказаний
            axes[1].scatter(all_actual_diam, all_predictions_diam, alpha=0.6)
            min_val = min(min(all_actual_diam), min(all_predictions_diam))
            max_val = max(max(all_actual_diam), max(all_predictions_diam))
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[1].set_xlabel('Фактические диаметры (мкм)')
            axes[1].set_ylabel('Предсказанные диаметры (мкм)')
            axes[1].set_title('Качество универсальной модели диаметра')
            axes[1].grid(True, alpha=0.3)
            
            # Метрики качества
            metrics_universal_diam = calculate_comprehensive_metrics(
                np.array(all_actual_diam), np.array(all_predictions_diam)
            )
            axes[1].text(0.05, 0.95, f"R² = {metrics_universal_diam['R²']:.3f}\nRMSE = {metrics_universal_diam['RMSE']:.2f}", 
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Универсальная модель для содержания фазы
    if enable_phase_analysis:
        st.subheader("Универсальная модель содержания фазы (JMAK)")
        
        universal_phase_params, universal_phase_cov = fit_universal_phase_model(df_grain10)
        
        if universal_phase_params is not None:
            A_phase, Ea_phase, n_phase = universal_phase_params
            
            st.success("✅ Универсальная модель содержания фазы успешно подобрана!")
            st.info(f"""
            **Параметры универсальной модели фазы:**
            - Предэкспоненциальный множитель A = {A_phase:.4e}
            - Энергия активации Ea = {Ea_phase:.0f} Дж/моль ({Ea_phase/1000:.1f} кДж/моль)
            - Показатель Аврами n = {n_phase:.2f}
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
                    
                    f_pred_universal = universal_phase_model(t_temp, T_temp, A_phase, Ea_phase, n_phase)
                    
                    axes[0].scatter(temp_data['t'], temp_data['f'], alpha=0.7, 
                                   label=f'{temp}°C (эксп.)', s=50)
                    axes[0].plot(temp_data['t'], f_pred_universal, '--', 
                                label=f'{temp}°C (мод.)', linewidth=2)
                    
                    all_predictions_phase.extend(f_pred_universal)
                    all_actual_phase.extend(temp_data['f'].values)
            
            axes[0].set_xlabel('Время (часы)')
            axes[0].set_ylabel('Содержание фазы (%)')
            axes[0].set_title('Универсальная модель содержания фазы\nдля всех температур')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # График качества
            axes[1].scatter(all_actual_phase, all_predictions_phase, alpha=0.6)
            axes[1].plot([0, 100], [0, 100], 'r--', linewidth=2)
            axes[1].set_xlabel('Фактическое содержание фазы (%)')
            axes[1].set_ylabel('Предсказанное содержание фазы (%)')
            axes[1].set_title('Качество универсальной модели фазы')
            axes[1].grid(True, alpha=0.3)
            
            metrics_universal_phase = calculate_comprehensive_metrics(
                np.array(all_actual_phase), np.array(all_predictions_phase)
            )
            axes[1].text(0.05, 0.95, f"R² = {metrics_universal_phase['R²']:.3f}\nRMSE = {metrics_universal_phase['RMSE']:.2f}", 
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)

    # НОВЫЙ РАЗДЕЛ: КАЛЬКУЛЯТОР
    st.header("4. 🧮 Калькулятор прогнозирования")
    
    st.markdown("""
    **Используйте рассчитанные универсальные модели для прогнозирования:**
    - Рассчитайте диаметр или содержание фазы для заданных условий
    - Определите температуру для достижения целевых значений
    - Оптимизируйте технологические параметры
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
            if calc_mode in ["Диаметр", "Оба параметра"] and universal_diameter_params is not None:
                A_diam, Ea_diam = universal_diameter_params
                predicted_diameter = universal_diameter_model(
                    target_time, target_temp, A_diam, Ea_diam, best_n, initial_diameter
                )
                st.success(f"**Прогнозируемый диаметр:** {predicted_diameter:.2f} мкм")
            
            if calc_mode in ["Содержание фазы", "Оба параметра"] and universal_phase_params is not None:
                A_phase, Ea_phase, n_phase = universal_phase_params
                predicted_phase = universal_phase_model(
                    target_time, target_temp, A_phase, Ea_phase, n_phase
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
            if temp_mode == "Диаметр (мкм)" and universal_diameter_params is not None:
                # Решаем уравнение для температуры
                A_diam, Ea_diam = universal_diameter_params
                
                def equation(T):
                    k = arrhenius_model(T + 273.15, A_diam, Ea_diam)
                    return (k * target_time_temp + initial_diameter**best_n)**(1/best_n) - target_value
                
                # Ищем температуру в разумном диапазоне
                T_candidates = np.linspace(400, 1200, 1000)
                differences = [equation(T) for T in T_candidates]
                
                # Находим температуру, где разница ближе всего к нулю
                idx_min = np.argmin(np.abs(differences))
                optimal_temp = T_candidates[idx_min]
                
                if np.abs(differences[idx_min]) < 0.1:  # Допустимая погрешность
                    st.success(f"**Необходимая температура:** {optimal_temp:.1f}°C")
                    st.info(f"При {optimal_temp:.1f}°C за {target_time_temp} часов диаметр достигнет {target_value} мкм")
                else:
                    st.warning("Не удалось найти решение в допустимом диапазоне температур")
            
            elif temp_mode == "Содержание фазы (%)" and universal_phase_params is not None:
                A_phase, Ea_phase, n_phase = universal_phase_params
                
                def equation_phase(T):
                    k = arrhenius_model(T + 273.15, A_phase, Ea_phase)
                    return jmak_model(target_time_temp, k, n_phase) * 100 - target_value
                
                T_candidates = np.linspace(400, 1200, 1000)
                differences = [equation_phase(T) for T in T_candidates]
                
                idx_min = np.argmin(np.abs(differences))
                optimal_temp = T_candidates[idx_min]
                
                if np.abs(differences[idx_min]) < 1.0:  # Допустимая погрешность 1%
                    st.success(f"**Необходимая температура:** {optimal_temp:.1f}°C")
                    st.info(f"При {optimal_temp:.1f}°C за {target_time_temp} часов содержание фазы достигнет {target_value}%")
                else:
                    st.warning("Не удалось найти решение в допустимом диапазоне температур")

    # Визуализация температурных зависимостей
    st.header("5. 📈 Температурные зависимости")
    
    if universal_diameter_params is not None:
        st.subheader("Зависимость константы скорости от температуры")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График в координатах Аррениуса
        T_range_kelvin = np.linspace(273 + 400, 273 + 1200, 100)
        T_range_celsius = T_range_kelvin - 273
        
        if universal_diameter_params is not None:
            A_diam, Ea_diam = universal_diameter_params
            k_diam = arrhenius_model(T_range_kelvin, A_diam, Ea_diam)
            ax1.semilogy(1000/T_range_kelvin, k_diam, 'b-', linewidth=2, label='Диаметр')
        
        if universal_phase_params is not None:
            A_phase, Ea_phase, n_phase = universal_phase_params
            k_phase = arrhenius_model(T_range_kelvin, A_phase, Ea_phase)
            ax1.semilogy(1000/T_range_kelvin, k_phase, 'r-', linewidth=2, label='Содержание фазы')
        
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
        
        ax2.set_xlabel('Температура (°C)')
        ax2.set_ylabel('Константа скорости k')
        ax2.set_title('Температурная зависимость')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

st.header("🎯 Рекомендации по использованию моделей")

st.markdown("""
**Универсальные модели позволяют:**

1. **Прогнозировать** поведение системы для любых температур и времен
2. **Оптимизировать** технологические параметры
3. **Сравнивать** кинетику разных процессов

**Для точных прогнозов:**
- Убедитесь, что целевые параметры находятся в диапазоне экспериментальных данных
- Проверьте физическую осмысленность результатов
- Учитывайте ограничения моделей за пределами экспериментального диапазона

**Энергии активации:**
- Типичные значения для диффузионных процессов: 100-300 кДж/моль
- Сравнение Ea для диаметра и содержания фазы может выявить разные лимитирующие стадии
""")
