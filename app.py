import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
import io
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Данные из таблицы ГОСТ 5639-82
grain_data = {
    'grain_size': [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'grain_area': [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.00781, 0.00390, 
                   0.00195, 0.00098, 0.00049, 0.000244, 0.000122, 0.000061, 0.000030, 
                   0.000015, 0.000008],
    'grain_diameter': [0.875, 0.650, 0.444, 0.313, 0.222, 0.157, 0.111, 0.0783, 0.0553,
                       0.0391, 0.0267, 0.0196, 0.0138, 0.0099, 0.0069, 0.0049, 0.0032, 0.0027]
}

grain_df = pd.DataFrame(grain_data)

# Универсальная газовая постоянная
R = 8.314  # Дж/(моль·К)

# ФИЗИЧЕСКИЕ МОДЕЛИ С УЧЕТОМ ТЕМПЕРАТУРЫ
def arrhenius_model(T, k0, Q):
    """Уравнение Аррениуса: k = k0 * exp(-Q/RT)"""
    return k0 * np.exp(-Q / (R * T))

def growth_model_with_temperature(t, T, k0, Q, n, grain_area, alpha=0.1, d0=0):
    """
    Полная модель роста с учетом температуры через Аррениуса:
    d = d0 + k0 * exp(-Q/RT) * (1 + alpha/grain_area) * t^n
    """
    k_arrhenius = arrhenius_model(T, k0, Q)
    boundary_effect = 1 + alpha / grain_area
    return d0 + k_arrhenius * boundary_effect * (t ** n)

def universal_growth_model(X, k0, Q, n, alpha, d0=0):
    """
    Универсальная модель для всех данных:
    X = [t, T, grain_area]
    """
    t, T, grain_area = X[:, 0], X[:, 1], X[:, 2]
    k_arrhenius = arrhenius_model(T, k0, Q)
    boundary_effect = 1 + alpha / grain_area
    return d0 + k_arrhenius * boundary_effect * (t ** n)

def grain_specific_model(X, k0, Q, n, d0=0):
    """
    Модель для конкретного зерна (без учета grain_area в модели):
    X = [t, T]
    """
    t, T = X[:, 0], X[:, 1]
    k_arrhenius = arrhenius_model(T, k0, Q)
    return d0 + k_arrhenius * (t ** n)

# Функции для оценки качества модели
def calculate_metrics(y_true, y_pred):
    """Расчет метрик качества модели"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-10))) * 100
    
    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def plot_residuals(y_true, y_pred, title):
    """График остатков"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Гистограмма остатков
    ax1.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Остатки')
    ax1.set_ylabel('Частота')
    ax1.set_title(f'Распределение остатков\n{title}')
    ax1.grid(True, alpha=0.3)
    
    # Остатки vs предсказанные значения
    ax2.scatter(y_pred, residuals, alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Предсказанные значения')
    ax2.set_ylabel('Остатки')
    ax2.set_title('Остатки vs Предсказания')
    ax2.grid(True, alpha=0.3)
    
    return fig

def plot_arrhenius_analysis(temperatures, k_values, k0, Q, grain_size=None):
    """График анализа Аррениуса"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    T_kelvin = temperatures + 273.15
    inv_T = 1 / T_kelvin
    log_k = np.log(k_values)
    
    # Экспериментальные точки
    ax.scatter(inv_T, log_k, s=100, color='blue', alpha=0.7, label='Экспериментальные k')
    
    # Линия Аррениуса
    T_range = np.linspace(T_kelvin.min(), T_kelvin.max(), 100)
    inv_T_range = 1 / T_range
    k_range = arrhenius_model(T_range, k0, Q)
    log_k_range = np.log(k_range)
    
    ax.plot(inv_T_range, log_k_range, 'r-', linewidth=2, 
            label=f'Уравнение Аррениуса\nQ = {Q:.0f} Дж/моль')
    
    ax.set_xlabel('1/T (1/K)')
    ax.set_ylabel('ln(k)')
    title = 'Анализ Аррениуса'
    if grain_size:
        title += f' для зерна {grain_size}'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Добавляем вторую ось с температурами в °C
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    temp_ticks = np.array([500, 600, 700, 800, 900])
    inv_temp_ticks = 1 / (temp_ticks + 273.15)
    ax2.set_xticks(inv_temp_ticks)
    ax2.set_xticklabels(temp_ticks)
    ax2.set_xlabel('Температура (°C)')
    
    return fig

# Основная программа Streamlit
st.title("🌡️ Моделирование кинетики роста σ-фазы с учетом температуры")

# Загрузка данных
st.header("1. Загрузка данных")

def create_template():
    template_data = {
        'G': [7, 7, 7, 7, 5, 5, 5, 5, 3, 3, 3, 3],
        'T': [600, 600, 600, 600, 650, 650, 650, 650, 700, 700, 700, 700],
        't': [2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000],
        'd': [2.1, 3.0, 3.6, 4.1, 3.5, 4.8, 5.8, 6.5, 5.2, 7.1, 8.5, 9.6]
    }
    return pd.DataFrame(template_data)

template_df = create_template()
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    template_df.to_excel(writer, sheet_name='Шаблон_данных', index=False)
excel_buffer.seek(0)

st.download_button(
    label="📥 Скачать шаблон Excel",
    data=excel_buffer,
    file_name="шаблон_данных_сигма_фаза.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded_file = st.file_uploader("Загрузите файл с данными", type=['csv', 'xlsx', 'xls'])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if all(col in df.columns for col in ['G', 'T', 't', 'd']):
            st.session_state['experimental_data'] = df
            st.success("Данные успешно загружены!")
            
            # Показываем статистику
            st.subheader("Статистика данных:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Температуры", f"{df['T'].min()} - {df['T'].max()}°C")
            with col2:
                st.metric("Время", f"{df['t'].min()} - {df['t'].max()} ч")
            with col3:
                st.metric("Диаметры", f"{df['d'].min():.1f} - {df['d'].max():.1f} мкм")
            with col4:
                st.metric("Номера зерен", f"{df['G'].nunique()} шт")
                
            st.dataframe(df.head())
        else:
            st.error("Отсутствуют необходимые колонки: G, T, t, d")
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")

# Анализ данных
if 'experimental_data' in st.session_state:
    df = st.session_state['experimental_data']
    df_enriched = df.merge(grain_df, left_on='G', right_on='grain_size', how='left')
    df_enriched['T_K'] = df_enriched['T'] + 273.15  # Температура в Кельвинах
    
    st.header("2. Подбор коэффициентов моделей с учетом температуры")
    
    # Выбор модели
    model_type = st.selectbox(
        "Выберите тип модели:",
        ["Индивидуальные модели для каждого зерна", "Универсальная модель для всех зерен"]
    )
    
    if model_type == "Индивидуальные модели для каждого зерна":
        st.subheader("Индивидуальные модели для каждого номера зерна")
        
        individual_results = {}
        
        for grain_size in sorted(df['G'].unique()):
            st.markdown(f"### 🔍 Анализ для зерна {grain_size}")
            
            grain_data = df_enriched[df_enriched['G'] == grain_size]
            
            if len(grain_data) >= 4:  # Минимум 4 точки для подбора
                try:
                    # Подготовка данных
                    X_grain = grain_data[['t', 'T_K']].values
                    y_grain = grain_data['d'].values
                    
                    # Подбор параметров модели с температурой
                    popt, pcov = curve_fit(grain_specific_model, 
                                         X_grain, 
                                         y_grain,
                                         p0=[1.0, 200000, 0.5],  # k0, Q, n
                                         bounds=([0.001, 100000, 0.1], 
                                                [1000, 500000, 2.0]))
                    
                    k0_opt, Q_opt, n_opt = popt
                    y_pred = grain_specific_model(X_grain, k0_opt, Q_opt, n_opt)
                    metrics = calculate_metrics(y_grain, y_pred)
                    
                    individual_results[grain_size] = {
                        'k0': k0_opt,
                        'Q': Q_opt,
                        'n': n_opt,
                        'metrics': metrics,
                        'predictions': y_pred,
                        'grain_area': grain_data['grain_area'].iloc[0]
                    }
                    
                    # Вывод результатов
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Параметры модели:**")
                        st.write(f"- k₀ = {k0_opt:.4f}")
                        st.write(f"- Q = {Q_opt:.0f} Дж/моль")
                        st.write(f"- n = {n_opt:.4f}")
                        st.write(f"- Площадь зерна = {grain_data['grain_area'].iloc[0]:.6f} мм²")
                    
                    with col2:
                        st.write("**Метрики качества:**")
                        for metric, value in metrics.items():
                            st.write(f"- {metric} = {value:.4f}")
                    
                    # График Аррениуса для этого зерна
                    # Сначала найдем k для каждой температуры
                    temp_k_values = []
                    temp_values = []
                    for temp in grain_data['T'].unique():
                        temp_data = grain_data[grain_data['T'] == temp]
                        if len(temp_data) >= 2:
                            # Оцениваем k из данных
                            try:
                                popt_temp, _ = curve_fit(lambda t, k: k * (t ** n_opt), 
                                                       temp_data['t'], temp_data['d'],
                                                       p0=[0.1])
                                temp_k_values.append(popt_temp[0])
                                temp_values.append(temp)
                            except:
                                pass
                    
                    if len(temp_k_values) >= 2:
                        arrhenius_fig = plot_arrhenius_analysis(
                            np.array(temp_values), np.array(temp_k_values), 
                            k0_opt, Q_opt, grain_size
                        )
                        st.pyplot(arrhenius_fig)
                    
                    # График модели vs эксперимента
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Разные цвета для разных температур
                    colors = plt.cm.viridis(np.linspace(0, 1, len(grain_data['T'].unique())))
                    
                    for i, temp in enumerate(grain_data['T'].unique()):
                        temp_data = grain_data[grain_data['T'] == temp]
                        temp_mask = grain_data['T'] == temp
                        temp_pred = y_pred[temp_mask]
                        
                        ax.scatter(temp_data['t'], temp_data['d'], 
                                  color=colors[i], label=f'{temp}°C', s=80, alpha=0.7)
                        
                        # Линия модели для этой температуры
                        t_range = np.linspace(temp_data['t'].min(), temp_data['t'].max(), 100)
                        T_K_const = temp + 273.15
                        d_pred_range = grain_specific_model(
                            np.column_stack([t_range, np.full_like(t_range, T_K_const)]), 
                            k0_opt, Q_opt, n_opt
                        )
                        ax.plot(t_range, d_pred_range, color=colors[i], linestyle='--', alpha=0.7)
                    
                    ax.set_xlabel('Время (часы)')
                    ax.set_ylabel('Диаметр σ-фазы (мкм)')
                    ax.set_title(f'Модель роста для зерна {grain_size}\n'
                                f'R² = {metrics["R²"]:.3f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Ошибка подбора для зерна {grain_size}: {e}")
            else:
                st.warning(f"Недостаточно данных для зерна {grain_size} (нужно минимум 4 точки)")
        
        # Сводная таблица по всем зернам
        if individual_results:
            st.subheader("📋 Сводная таблица параметров моделей")
            
            summary_data = []
            for grain_size, results in individual_results.items():
                summary_data.append({
                    'Номер зерна': grain_size,
                    'Площадь зерна': results['grain_area'],
                    'k₀': results['k0'],
                    'Q, Дж/моль': results['Q'],
                    'n': results['n'],
                    'R²': results['metrics']['R²'],
                    'RMSE': results['metrics']['RMSE']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.format({
                'Площадь зерна': '{:.6f}',
                'k₀': '{:.4f}',
                'Q, Дж/моль': '{:.0f}',
                'n': '{:.4f}',
                'R²': '{:.4f}',
                'RMSE': '{:.4f}'
            }))
            
            st.session_state['individual_results'] = individual_results
    
    else:  # Универсальная модель
        st.subheader("Универсальная модель для всех зерен")
        
        # Подготовка данных
        X = df_enriched[['t', 'T_K', 'grain_area']].values
        y = df_enriched['d'].values
        
        try:
            # Подбор параметров универсальной модели
            popt, pcov = curve_fit(universal_growth_model, X, y,
                                 p0=[1.0, 200000, 0.5, 0.01],
                                 bounds=([0.001, 100000, 0.1, 0], 
                                        [1000, 500000, 2.0, 1.0]))
            
            k0_uni, Q_uni, n_uni, alpha_uni = popt
            y_pred_uni = universal_growth_model(X, k0_uni, Q_uni, n_uni, alpha_uni)
            metrics_uni = calculate_metrics(y, y_pred_uni)
            
            # Вывод параметров
            st.write("**Параметры универсальной модели:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("k₀", f"{k0_uni:.4f}")
            with col2:
                st.metric("Q", f"{Q_uni:.0f} Дж/моль")
            with col3:
                st.metric("n", f"{n_uni:.4f}")
            with col4:
                st.metric("α", f"{alpha_uni:.4f}")
            
            st.write("**Метрики качества:**")
            metrics_cols = st.columns(4)
            for i, (metric, value) in enumerate(metrics_uni.items()):
                with metrics_cols[i]:
                    st.metric(metric, f"{value:.4f}")
            
            # Анализ Аррениуса для универсальной модели
            st.subheader("Анализ Аррениуса для универсальной модели")
            
            # Оценим k для каждой температуры из данных
            unique_temps = df_enriched['T'].unique()
            k_estimated = []
            for temp in unique_temps:
                temp_data = df_enriched[df_enriched['T'] == temp]
                if len(temp_data) >= 2:
                    try:
                        # Усредненная оценка k для этой температуры
                        k_temp = arrhenius_model(temp + 273.15, k0_uni, Q_uni)
                        k_estimated.append(k_temp)
                    except:
                        pass
            
            if len(k_estimated) >= 2:
                arrhenius_fig = plot_arrhenius_analysis(
                    unique_temps[:len(k_estimated)], np.array(k_estimated), 
                    k0_uni, Q_uni
                )
                st.pyplot(arrhenius_fig)
            
            # Визуализация предсказаний vs эксперимента
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_enriched['G'].unique())))
            
            for i, grain_size in enumerate(df_enriched['G'].unique()):
                mask = df_enriched['G'] == grain_size
                subset = df_enriched[mask]
                pred_subset = y_pred_uni[mask]
                
                ax.scatter(subset['d'], pred_subset, 
                          color=colors[i], label=f'Зерно {grain_size}', 
                          s=80, alpha=0.7)
            
            # Линия идеального предсказания
            min_val = min(df_enriched['d'].min(), y_pred_uni.min())
            max_val = max(df_enriched['d'].max(), y_pred_uni.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                   label='Идеальное предсказание', linewidth=2)
            
            ax.set_xlabel('Экспериментальные значения (мкм)')
            ax.set_ylabel('Предсказанные значения (мкм)')
            ax.set_title(f'Универсальная модель: Предсказания vs Эксперимент\n'
                        f'R² = {metrics_uni["R²"]:.3f}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Таблица сравнения
            st.subheader("📊 Таблица сравнения значений")
            
            comparison_df = df_enriched[['G', 'T', 't', 'd']].copy()
            comparison_df['Предсказание'] = y_pred_uni
            comparison_df['Ошибка'] = comparison_df['d'] - comparison_df['Предсказание']
            comparison_df['|Ошибка|'] = np.abs(comparison_df['Ошибка'])
            
            st.dataframe(comparison_df.style.format({
                'd': '{:.2f}',
                'Предсказание': '{:.2f}',
                'Ошибка': '{:.3f}',
                '|Ошибка|': '{:.3f}'
            }).background_gradient(subset=['|Ошибка|'], cmap='Reds'))
            
            st.session_state['universal_results'] = {
                'k0': k0_uni, 'Q': Q_uni, 'n': n_uni, 'alpha': alpha_uni,
                'metrics': metrics_uni, 'predictions': y_pred_uni
            }
            
        except Exception as e:
            st.error(f"Ошибка подбора универсальной модели: {e}")
            st.info("Попробуйте увеличить количество данных или изменить начальные приближения")

    # Выгрузка результатов
    st.header("3. Выгрузка результатов")
    
    if st.button("📤 Сгенерировать отчет"):
        output_buffer = io.BytesIO()
        
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            # Исходные данные
            df_enriched.to_excel(writer, sheet_name='Исходные_данные', index=False)
            
            # Параметры моделей
            if 'individual_results' in st.session_state:
                ind_results = []
                for grain_size, results in st.session_state['individual_results'].items():
                    ind_results.append({
                        'Номер_зерна': grain_size,
                        'k0': results['k0'],
                        'Q_Дж_моль': results['Q'],
                        'n': results['n'],
                        'Площадь_зерна': results['grain_area'],
                        'R2': results['metrics']['R²'],
                        'RMSE': results['metrics']['RMSE'],
                        'MAE': results['metrics']['MAE']
                    })
                pd.DataFrame(ind_results).to_excel(writer, sheet_name='Индивидуальные_модели', index=False)
            
            if 'universal_results' in st.session_state:
                uni_results = pd.DataFrame([st.session_state['universal_results']])
                uni_results.to_excel(writer, sheet_name='Универсальная_модель', index=False)
            
            # Сравнительная таблица
            if 'universal_results' in st.session_state:
                comparison_df.to_excel(writer, sheet_name='Сравнение_значений', index=False)
        
        output_buffer.seek(0)
        
        st.download_button(
            label="💾 Скачать полный отчет в Excel",
            data=output_buffer,
            file_name="отчет_модели_сигма_фаза.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Информация о физической модели
with st.expander("ℹ️ О физической модели с температурой"):
    st.markdown("""
    **Полная физическая модель роста σ-фазы:**
    
    ```
    d = k₀ · exp(-Q/RT) · (1 + α/a_i) · tⁿ
    ```
    
    **Параметры модели:**
    - **k₀**: Предэкспоненциальный множитель
    - **Q**: Энергия активации (Дж/моль) 
    - **R**: Универсальная газовая постоянная (8.314 Дж/(моль·К))
    - **T**: Температура в Кельвинах
    - **α**: Коэффициент влияния границ зерен
    - **a_i**: Площадь сечения зерна (мм²)
    - **n**: Показатель степенного закона роста
    - **t**: Время (часы)
    
    **Физический смысл:**
    - Температурная зависимость описывается **уравнением Аррениуса**
    - Влияние размера зерна учитывается через **плотность границ**
    - Временная зависимость следует **степенному закону роста**
    
    **Ожидаемые значения:**
    - Q ≈ 200-300 кДж/моль для диффузионных процессов в сталях
    - n ≈ 0.3-0.7 для роста фаз
    - α > 0 (положительное влияние мелкого зерна)
    """)
