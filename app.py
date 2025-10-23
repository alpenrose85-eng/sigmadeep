import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
import io
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Универсальная газовая постоянная
R = 8.314  # Дж/(моль·К)

st.title("🔬 Комплексный анализ кинетики σ-фазы: диаметр + содержание")
st.markdown("""
**Двойной подход:**
1. **Анализ диаметра:** Степенной закон роста $d^n - d_0^n = K \\cdot t$
2. **Анализ содержания:** JMAK-модель $X(t) = 1 - \\exp(-[k(T)t]^m)$
""")

# Загрузка данных
st.header("1. Загрузка данных для зерна №10")

uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", 
                               type=['csv', 'xlsx', 'xls'])

# Параметры анализа
st.subheader("Параметры анализа:")
initial_diameter = st.number_input("Начальный диаметр d₀ (мкм)", 
                                 value=0.0, min_value=0.0, step=0.1)

target_grain = 10  # Фокусируемся на зерне №10

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        required_cols = ['G', 'T', 't', 'd', 'f']  # Добавили 'f' для содержания фазы
        
        if all(col in df.columns for col in required_cols):
            # Фильтруем данные для зерна №10
            df_grain10 = df[df['G'] == target_grain].copy()
            
            if len(df_grain10) > 0:
                st.session_state['grain10_data'] = df_grain10
                st.success(f"✅ Данные для зерна №10 успешно загружены!")
                
                # Статистика по зерну №10
                st.subheader("📊 Статистика данных для зерна №10:")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    unique_temps = df_grain10['T'].unique()
                    st.metric("Температуры", f"{len(unique_temps)} уровней")
                with col2:
                    time_range = f"{df_grain10['t'].min()} - {df_grain10['t'].max()}"
                    st.metric("Время выдержки", time_range + " ч")
                with col3:
                    diam_range = f"{df_grain10['d'].min():.1f} - {df_grain10['d'].max():.1f}"
                    st.metric("Диаметры", diam_range + " мкм")
                with col4:
                    content_range = f"{df_grain10['f'].min():.1f} - {df_grain10['f'].max():.1f}"
                    st.metric("Содержание σ-фазы", content_range + " %")
                
                # Визуализация всех данных
                st.subheader("📈 Визуализация данных зерна №10")
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # График 1: Диаметры по температурам
                temperatures = sorted(df_grain10['T'].unique())
                colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))
                
                for i, temp in enumerate(temperatures):
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    ax1.scatter(temp_data['t'], temp_data['d'], 
                               color=colors[i], label=f'{temp}°C', s=80, alpha=0.8)
                    
                    sorted_data = temp_data.sort_values('t')
                    ax1.plot(sorted_data['t'], sorted_data['d'], 
                            color=colors[i], linestyle='--', alpha=0.5)
                
                ax1.set_xlabel('Время (часы)')
                ax1.set_ylabel('Диаметр σ-фазы (мкм)')
                ax1.set_title('Кинетика роста диаметра')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # График 2: Содержание фазы по температурам
                for i, temp in enumerate(temperatures):
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    ax2.scatter(temp_data['t'], temp_data['f'], 
                               color=colors[i], label=f'{temp}°C', s=80, alpha=0.8)
                    
                    sorted_data = temp_data.sort_values('t')
                    ax2.plot(sorted_data['t'], sorted_data['f'], 
                            color=colors[i], linestyle='--', alpha=0.5)
                
                ax2.set_xlabel('Время (часы)')
                ax2.set_ylabel('Содержание σ-фазы (%)')
                ax2.set_title('Кинетика изменения содержания фазы')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # График 3: Связь диаметра и содержания
                for i, temp in enumerate(temperatures):
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    ax3.scatter(temp_data['d'], temp_data['f'], 
                               color=colors[i], label=f'{temp}°C', s=80, alpha=0.8)
                
                ax3.set_xlabel('Диаметр (мкм)')
                ax3.set_ylabel('Содержание σ-фазы (%)')
                ax3.set_title('Связь диаметра и содержания фазы')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # График 4: Распределение данных по времени
                ax4.hist(df_grain10['t'], bins=15, alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Время (часы)')
                ax4.set_ylabel('Количество точек')
                ax4.set_title('Распределение данных по времени выдержки')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.dataframe(df_grain10.head(10))
                
            else:
                st.error(f"❌ В данных нет записей для зерна №10")
        else:
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"❌ Отсутствуют колонки: {missing}")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")

# Функции для JMAK-модели
def jmak_model(t, k, m, X_inf=100):
    """
    JMAK-модель для кинетики фазового превращения
    X(t) = X_inf * [1 - exp(-(k*t)^m)]
    """
    return X_inf * (1 - np.exp(-(k * t) ** m))

def jmak_model_fixed_m(t, k, X_inf=100, m=1):
    """JMAK-модель с фиксированным показателем m"""
    return X_inf * (1 - np.exp(-(k * t) ** m))

def jmak_arrhenius(T, k0, Q, R=8.314):
    """Уравнение Аррениуса для JMAK-константы"""
    return k0 * np.exp(-Q / (R * T))

# Основной расчет
if 'grain10_data' in st.session_state:
    df_grain10 = st.session_state['grain10_data']
    df_grain10['T_K'] = df_grain10['T'] + 273.15
    
    # Выбор типа анализа
    st.header("2. Выбор типа анализа")
    analysis_type = st.radio("Выберите анализ:", 
                           ["Только диаметры", "Только содержание фазы", "Оба анализа параллельно"],
                           index=2)
    
    # АНАЛИЗ ДИАМЕТРОВ
    if analysis_type in ["Только диаметры", "Оба анализа параллельно"]:
        st.header("📏 Анализ диаметров σ-фазы")
        
        # Подбор показателя степени n
        st.subheader("Подбор оптимального показателя степени n")
        
        n_min = 3.0
        n_max = 5.0
        n_step = 0.1
        n_candidates = np.arange(n_min, n_max + n_step, n_step)
        
        n_results = {}
        
        for n in n_candidates:
            k_values = []
            
            for temp in df_grain10['T'].unique():
                temp_data = df_grain10[df_grain10['T'] == temp]
                
                if len(temp_data) >= 2:
                    d_transformed = temp_data['d']**n - initial_diameter**n
                    
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            temp_data['t'], d_transformed
                        )
                        
                        if slope > 0:
                            k_values.append({
                                'T': temp,
                                'T_K': temp + 273.15,
                                'K': slope,
                                'R2': r_value**2,
                                'std_err': std_err,
                                'n_points': len(temp_data)
                            })
                    except:
                        continue
            
            if k_values:
                k_df = pd.DataFrame(k_values)
                n_results[n] = {
                    'k_df': k_df,
                    'mean_R2': k_df['R2'].mean(),
                    'min_R2': k_df['R2'].min(),
                    'n_temperatures': len(k_df)
                }
        
        # Выбор лучшего n
        if n_results:
            comparison_data = []
            for n, results in n_results.items():
                comparison_data.append({
                    'n': n,
                    'Средний R²': results['mean_R2'],
                    'Минимальный R²': results['min_R2'],
                    'Количество температур': results['n_temperatures']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            best_n_row = comparison_df.loc[comparison_df['Средний R²'].idxmax()]
            best_n = best_n_row['n']
            
            st.success(f"🎯 Оптимальный показатель для диаметров: n = {best_n:.1f}")
            
            # Анализ Аррениуса для диаметров
            best_k_df = n_results[best_n]['k_df']
            
            if len(best_k_df) >= 2:
                x = 1 / best_k_df['T_K']
                y = np.log(best_k_df['K'])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                Q_diam = -slope * R
                K0_diam = np.exp(intercept)
                
                # Сохраняем параметры диаметров
                st.session_state['diam_params'] = {
                    'n': best_n,
                    'Q': Q_diam,
                    'K0': K0_diam,
                    'd0': initial_diameter,
                    'arrhenius_R2': r_value**2
                }
    
    # АНАЛИЗ СОДЕРЖАНИЯ ФАЗЫ
    if analysis_type in ["Только содержание фазы", "Оба анализа параллельно"]:
        st.header("📊 Анализ содержания σ-фазы (JMAK-модель)")
        
        st.markdown("""
        **JMAK-модель:**
        $$
        X(t) = X_\\infty \\cdot [1 - \\exp(-(k \\cdot t)^m)]
        $$
        """)
        
        # Подбор параметров JMAK для каждой температуры
        jmak_results = {}
        
        for temp in df_grain10['T'].unique():
            temp_data = df_grain10[df_grain10['T'] == temp]
            
            if len(temp_data) >= 3:  # Нужно минимум 3 точки для JMAK
                try:
                    # Подбор параметров JMAK
                    popt, pcov = curve_fit(jmak_model, 
                                         temp_data['t'], 
                                         temp_data['f'],
                                         p0=[0.001, 1.0, 100],  # k, m, X_inf
                                         bounds=([1e-6, 0.1, 50], 
                                                [1.0, 3.0, 150]))
                    
                    k_jmak, m_jmak, X_inf = popt
                    y_pred = jmak_model(temp_data['t'], k_jmak, m_jmak, X_inf)
                    r2 = r2_score(temp_data['f'], y_pred)
                    
                    jmak_results[temp] = {
                        'k': k_jmak,
                        'm': m_jmak,
                        'X_inf': X_inf,
                        'R2': r2,
                        'n_points': len(temp_data)
                    }
                    
                except Exception as e:
                    st.warning(f"Ошибка подбора JMAK для {temp}°C: {e}")
        
        if jmak_results:
            # Визуализация JMAK-подбора
            st.subheader("Результаты подбора JMAK-параметров")
            
            n_temps = len(jmak_results)
            n_cols = min(3, n_temps)
            n_rows = (n_temps + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            elif n_temps == 1:
                axes = np.array([[axes]])
            
            for idx, (temp, results) in enumerate(jmak_results.items()):
                if idx < n_rows * n_cols:
                    row = idx // n_cols
                    col = idx % n_cols
                    
                    if n_rows == 1:
                        ax = axes[col]
                    else:
                        ax = axes[row, col]
                    
                    temp_data = df_grain10[df_grain10['T'] == temp]
                    
                    # Экспериментальные точки
                    ax.scatter(temp_data['t'], temp_data['f'], 
                              alpha=0.7, s=60, label='Эксперимент')
                    
                    # JMAK-кривая
                    t_range = np.linspace(temp_data['t'].min(), temp_data['t'].max() * 1.2, 100)
                    y_pred_range = jmak_model(t_range, results['k'], results['m'], results['X_inf'])
                    ax.plot(t_range, y_pred_range, 'r--', 
                           label=f'JMAK: k={results["k"]:.4f}, m={results["m"]:.2f}')
                    
                    ax.set_xlabel('Время (часы)')
                    ax.set_ylabel('Содержание σ-фазы (%)')
                    ax.set_title(f'{temp}°C, R²={results["R2"]:.3f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            # Скрываем пустые subplots
            for idx in range(len(jmak_results), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                if n_rows == 1:
                    axes[col].set_visible(False)
                else:
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Анализ Аррениуса для JMAK-констант
            st.subheader("Аррениус-анализ для JMAK-констант")
            
            jmak_df = pd.DataFrame([
                {**{'T': temp, 'T_K': temp + 273.15}, **results} 
                for temp, results in jmak_results.items()
            ])
            
            if len(jmak_df) >= 2:
                x = 1 / jmak_df['T_K']
                y = np.log(jmak_df['k'])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                Q_jmak = -slope * R
                k0_jmak = np.exp(intercept)
                
                # Визуализация Аррениуса для JMAK
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # График 1: Аррениус
                ax1.scatter(x, y, s=100, color='green', alpha=0.7)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                ax1.plot(x_fit, y_fit, 'r--', linewidth=2,
                       label=f'Q = {Q_jmak:.0f} Дж/моль\nR² = {r_value**2:.4f}')
                ax1.set_xlabel('1/T (1/K)')
                ax1.set_ylabel('ln(k)')
                ax1.set_title('Аррениус для JMAK-констант')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # График 2: Показатель m по температурам
                ax2.scatter(jmak_df['T'], jmak_df['m'], s=100, alpha=0.7)
                ax2.set_xlabel('Температура (°C)')
                ax2.set_ylabel('Показатель m')
                ax2.set_title('Показатель JMAK по температурам')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Сохраняем параметры JMAK
                st.session_state['jmak_params'] = {
                    'Q': Q_jmak,
                    'k0': k0_jmak,
                    'mean_m': jmak_df['m'].mean(),
                    'arrhenius_R2': r_value**2
                }
                
                st.success(f"**JMAK-анализ:** Q = {Q_jmak:.0f} Дж/моль, k₀ = {k0_jmak:.6f}")
    
    # СРАВНИТЕЛЬНЫЙ АНАЛИЗ
    if analysis_type == "Оба анализа параллельно" and 'diam_params' in st.session_state and 'jmak_params' in st.session_state:
        st.header("📊 Сравнительный анализ методов")
        
        diam_params = st.session_state['diam_params']
        jmak_params = st.session_state['jmak_params']
        
        # Таблица сравнения
        st.subheader("Сравнение параметров")
        
        comparison_data = {
            'Параметр': ['Энергия активации Q, Дж/моль', 'Предэкспонента', 'R² Аррениуса'],
            'Анализ диаметров': [
                f"{diam_params['Q']:.0f}",
                f"K₀ = {diam_params['K0']:.6f}",
                f"{diam_params['arrhenius_R2']:.4f}"
            ],
            'JMAK-анализ': [
                f"{jmak_params['Q']:.0f}",
                f"k₀ = {jmak_params['k0']:.6f}", 
                f"{jmak_params['arrhenius_R2']:.4f}"
            ]
        }
        
        st.table(pd.DataFrame(comparison_data))
        
        # Визуализация сравнения
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: Сравнение предсказаний для одной температуры
        example_temp = sorted(df_grain10['T'].unique())[0]
        temp_data = df_grain10[df_grain10['T'] == example_temp]
        
        # Предсказания по диаметру
        k_diam = diam_params['K0'] * np.exp(-diam_params['Q'] / (R * (example_temp + 273.15)))
        t_range = np.linspace(temp_data['t'].min(), temp_data['t'].max() * 1.2, 100)
        d_pred = (k_diam * t_range + initial_diameter**diam_params['n'])**(1/diam_params['n'])
        
        axes[0].scatter(temp_data['t'], temp_data['d'], label='Эксперимент (диаметр)', alpha=0.7)
        axes[0].plot(t_range, d_pred, 'b--', label='Модель диаметра')
        axes[0].set_xlabel('Время (часы)')
        axes[0].set_ylabel('Диаметр (мкм)')
        axes[0].set_title(f'Сравнение моделей для {example_temp}°C')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Предсказания по JMAK
        k_jmak = jmak_params['k0'] * np.exp(-jmak_params['Q'] / (R * (example_temp + 273.15)))
        f_pred = jmak_model(t_range, k_jmak, jmak_params['mean_m'])
        
        axes[1].scatter(temp_data['t'], temp_data['f'], label='Эксперимент (содержание)', alpha=0.7)
        axes[1].plot(t_range, f_pred, 'g--', label='JMAK-модель')
        axes[1].set_xlabel('Время (часы)')
        axes[1].set_ylabel('Содержание (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # ОБРАТНЫЙ РАСЧЕТ ТЕМПЕРАТУРЫ
    st.header("🎯 Обратный расчет температуры эксплуатации")
    
    if analysis_type == "Оба анализа параллельно":
        method = st.radio("Выберите метод для обратного расчета:",
                         ["По диаметру", "По содержанию фазы", "Совместный анализ"])
    else:
        method = "По диаметру" if analysis_type == "Только диаметры" else "По содержанию фазы"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if method == "По диаметру":
            d_obs = st.number_input("Наблюдаемый диаметр d (мкм)", 
                                  value=5.0, min_value=0.1, step=0.1)
        else:
            f_obs = st.number_input("Наблюдаемое содержание f (%)", 
                                  value=10.0, min_value=0.1, step=0.1)
    with col2:
        t_obs = st.number_input("Время эксплуатации t (часы)", 
                              value=5000, min_value=1, step=100)
    
    if st.button("Рассчитать температуру"):
        if method == "По диаметру" and 'diam_params' in st.session_state:
            params = st.session_state['diam_params']
            k_obs = (d_obs**params['n'] - params['d0']**params['n']) / t_obs
            
            if k_obs > 0:
                denominator = R * (np.log(params['K0']) - np.log(k_obs))
                if denominator > 0:
                    T_K = params['Q'] / denominator
                    T_C = T_K - 273.15
                    st.success(f"**По диаметру: {T_C:.1f}°C**")
        
        elif method == "По содержанию фазы" and 'jmak_params' in st.session_state:
            params = st.session_state['jmak_params']
            # Решаем уравнение JMAK относительно k
            # f_obs = 100 * (1 - exp(-(k*t)^m))
            k_obs = (-np.log(1 - f_obs/100))**(1/params['mean_m']) / t_obs
            
            if k_obs > 0:
                denominator = R * (np.log(params['k0']) - np.log(k_obs))
                if denominator > 0:
                    T_K = params['Q'] / denominator
                    T_C = T_K - 273.15
                    st.success(f"**По содержанию: {T_C:.1f}°C**")
        
        elif method == "Совместный анализ" and 'diam_params' in st.session_state and 'jmak_params' in st.session_state:
            # Рассчитываем по обоим методам и усредняем
            temp_diam = temp_jmak = None
            
            # По диаметру
            params_diam = st.session_state['diam_params']
            k_obs_diam = (d_obs**params_diam['n'] - params_diam['d0']**params_diam['n']) / t_obs
            if k_obs_diam > 0:
                denominator_diam = R * (np.log(params_diam['K0']) - np.log(k_obs_diam))
                if denominator_diam > 0:
                    temp_diam = params_diam['Q'] / denominator_diam - 273.15
            
            # По содержанию
            params_jmak = st.session_state['jmak_params']
            k_obs_jmak = (-np.log(1 - f_obs/100))**(1/params_jmak['mean_m']) / t_obs
            if k_obs_jmak > 0:
                denominator_jmak = R * (np.log(params_jmak['k0']) - np.log(k_obs_jmak))
                if denominator_jmak > 0:
                    temp_jmak = params_jmak['Q'] / denominator_jmak - 273.15
            
            if temp_diam is not None and temp_jmak is not None:
                temp_avg = (temp_diam + temp_jmak) / 2
                st.success(f"**Совместный анализ: {temp_avg:.1f}°C**")
                st.write(f"- По диаметру: {temp_diam:.1f}°C")
                st.write(f"- По содержанию: {temp_jmak:.1f}°C")

# Теоретическая справка
with st.expander("📚 Теоретическая справка"):
    st.markdown("""
    **Двойной физический подход:**
    
    **1. Анализ диаметров (рост частиц):**
    $$
    d^n - d_0^n = K_0 \\cdot \\exp\\left(-\\frac{Q}{RT}\\right) \\cdot t
    $$
    - n ≈ 4.0: диффузия по границам зерен
    - n ≈ 3.0: объемная диффузия (LSW)
    
    **2. JMAK-анализ (фазовое превращение):**
    $$
    X(t) = X_\\infty \\cdot [1 - \\exp(-(k \\cdot t)^m)]
    $$
    $$
    k = k_0 \\cdot \\exp\\left(-\\frac{Q}{RT}\\right)
    $$
    - m: показатель, связанный с механизмом зарождения
    - X_∞: предельное содержание фазы
    
    **Преимущества двойного подхода:**
    - Взаимная проверка физической согласованности
    - Повышение надежности обратного расчета
    - Учет разных аспектов кинетики
    """)
