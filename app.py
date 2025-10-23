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

# Модели роста
def basic_growth_model(t, k, n, d0=0):
    """Базовая модель роста: d = d0 + k * t^n"""
    return d0 + k * (t ** n)

def enhanced_growth_model(t, k, n, grain_area, alpha=0.5, d0=0):
    """Улучшенная модель с учетом площади зерна: d = d0 + k * (1 + alpha/grain_area) * t^n"""
    boundary_effect = 1 + alpha / grain_area
    return d0 + k * boundary_effect * (t ** n)

def universal_growth_model(X, k, n, beta, d0=0):
    """Универсальная модель для всех зерен: d = d0 + k * (1 + beta/grain_area) * t^n"""
    t, grain_area = X[:, 0], X[:, 1]
    boundary_effect = 1 + beta / grain_area
    return d0 + k * boundary_effect * (t ** n)

# Функции для оценки качества модели
def calculate_metrics(y_true, y_pred):
    """Расчет метрик качества модели"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
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

# Основная программа Streamlit
st.title("📊 Моделирование кинетики роста σ-фазы с подбором коэффициентов")

# Загрузка данных
st.header("1. Загрузка данных")

# Создаем шаблон
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
            st.dataframe(df.head())
        else:
            st.error("Отсутствуют необходимые колонки: G, T, t, d")
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")

# Анализ данных
if 'experimental_data' in st.session_state:
    df = st.session_state['experimental_data']
    df_enriched = df.merge(grain_df, left_on='G', right_on='grain_size', how='left')
    
    st.header("2. Подбор коэффициентов моделей")
    
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
            
            grain_data = df[df['G'] == grain_size]
            grain_area = grain_df[grain_df['grain_size'] == grain_size]['grain_area'].iloc[0]
            
            if len(grain_data) >= 3:  # Минимум 3 точки для подбора
                # Подбор параметров базовой модели
                try:
                    popt, pcov = curve_fit(basic_growth_model, 
                                         grain_data['t'], 
                                         grain_data['d'],
                                         p0=[0.1, 0.5],
                                         bounds=([0, 0], [10, 2]))
                    
                    k_opt, n_opt = popt
                    y_pred = basic_growth_model(grain_data['t'], k_opt, n_opt)
                    metrics = calculate_metrics(grain_data['d'], y_pred)
                    
                    individual_results[grain_size] = {
                        'k': k_opt,
                        'n': n_opt,
                        'grain_area': grain_area,
                        'metrics': metrics,
                        'predictions': y_pred
                    }
                    
                    # Вывод результатов
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Параметры модели:**")
                        st.write(f"- k = {k_opt:.4f}")
                        st.write(f"- n = {n_opt:.4f}")
                        st.write(f"- Площадь зерна = {grain_area:.6f} мм²")
                    
                    with col2:
                        st.write("**Метрики качества:**")
                        for metric, value in metrics.items():
                            st.write(f"- {metric} = {value:.4f}")
                    
                    # График
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Экспериментальные точки
                    ax.scatter(grain_data['t'], grain_data['d'], color='blue', 
                              label='Эксперимент', s=80, alpha=0.7)
                    
                    # Предсказания модели
                    t_range = np.linspace(grain_data['t'].min(), grain_data['t'].max(), 100)
                    d_pred_range = basic_growth_model(t_range, k_opt, n_opt)
                    ax.plot(t_range, d_pred_range, 'r-', label='Модель', linewidth=2)
                    
                    # Соединяем точки линиями
                    sorted_indices = np.argsort(grain_data['t'])
                    ax.plot(grain_data['t'].iloc[sorted_indices], 
                           grain_data['d'].iloc[sorted_indices], 
                           'b--', alpha=0.5)
                    
                    ax.set_xlabel('Время (часы)')
                    ax.set_ylabel('Диаметр σ-фазы (мкм)')
                    ax.set_title(f'Модель роста для зерна {grain_size}\n'
                                f'R² = {metrics["R²"]:.3f}, RMSE = {metrics["RMSE"]:.3f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # График остатков
                    resid_fig = plot_residuals(grain_data['d'], y_pred, f'Зерно {grain_size}')
                    st.pyplot(resid_fig)
                    
                except Exception as e:
                    st.error(f"Ошибка подбора для зерна {grain_size}: {e}")
            else:
                st.warning(f"Недостаточно данных для зерна {grain_size} (нужно минимум 3 точки)")
        
        # Сводная таблица по всем зернам
        if individual_results:
            st.subheader("📋 Сводная таблица параметров моделей")
            
            summary_data = []
            for grain_size, results in individual_results.items():
                summary_data.append({
                    'Номер зерна': grain_size,
                    'Площадь зерна': results['grain_area'],
                    'k': results['k'],
                    'n': results['n'],
                    'R²': results['metrics']['R²'],
                    'RMSE': results['metrics']['RMSE'],
                    'MAE': results['metrics']['MAE']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.format({
                'Площадь зерна': '{:.6f}',
                'k': '{:.4f}',
                'n': '{:.4f}',
                'R²': '{:.4f}',
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}'
            }))
            
            st.session_state['individual_results'] = individual_results
    
    else:  # Универсальная модель
        st.subheader("Универсальная модель для всех зерен")
        
        # Подготовка данных
        X = df_enriched[['t', 'grain_area']].values
        y = df_enriched['d'].values
        
        try:
            # Подбор параметров универсальной модели
            popt, pcov = curve_fit(universal_growth_model, X, y,
                                 p0=[0.1, 0.5, 0.1],
                                 bounds=([0, 0, 0], [10, 2, 10]))
            
            k_uni, n_uni, beta_uni = popt
            y_pred_uni = universal_growth_model(X, k_uni, n_uni, beta_uni)
            metrics_uni = calculate_metrics(y, y_pred_uni)
            
            # Вывод параметров
            st.write("**Параметры универсальной модели:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("k", f"{k_uni:.4f}")
            with col2:
                st.metric("n", f"{n_uni:.4f}")
            with col3:
                st.metric("β", f"{beta_uni:.4f}")
            
            st.write("**Метрики качества:**")
            metrics_cols = st.columns(4)
            for i, (metric, value) in enumerate(metrics_uni.items()):
                with metrics_cols[i]:
                    st.metric(metric, f"{value:.4f}")
            
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
            
            # График остатков
            resid_fig = plot_residuals(y, y_pred_uni, "Универсальная модель")
            st.pyplot(resid_fig)
            
            # Таблица сравнения экспериментальных и предсказанных значений
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
                'k': k_uni, 'n': n_uni, 'beta': beta_uni,
                'metrics': metrics_uni, 'predictions': y_pred_uni
            }
            
        except Exception as e:
            st.error(f"Ошибка подбора универсальной модели: {e}")
    
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
                        'k': results['k'],
                        'n': results['n'],
                        'Площадь_зерна': results['grain_area'],
                        'R2': results['metrics']['R²'],
                        'RMSE': results['metrics']['RMSE'],
                        'MAE': results['metrics']['MAE'],
                        'MAPE': results['metrics']['MAPE']
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

# Информация о моделях
with st.expander("ℹ️ О физических моделях"):
    st.markdown("""
    **Используемые физические модели:**
    
    1. **Индивидуальные модели для каждого зерна:**
       ```
       d = k · tⁿ
       ```
       - Подбираются отдельные коэффициенты k, n для каждого номера зерна
       - Учитывает специфику кинетики роста для разных микроструктур
    
    2. **Универсальная модель для всех зерен:**
       ```
       d = k · (1 + β/a_i) · tⁿ
       ```
       - Единые коэффициенты k, n, β для всех данных
       - Коэффициент β учитывает влияние границ зерен через площадь a_i
       - Более универсальная, но требует больше данных для калибровки
    
    **Метрики качества:**
    - **R²**: Коэффициент детерминации (ближе к 1 = лучше)
    - **RMSE**: Среднеквадратичная ошибка
    - **MAE**: Средняя абсолютная ошибка  
    - **MAPE**: Средняя абсолютная процентная ошибка
    """)
