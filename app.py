import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

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

def enhanced_growth_model(t, k, n, grain_area, alpha=0.5, d0=0):
    """
    Улучшенная модель роста с учетом площади зерна
    d = d0 + k * (1 + alpha/grain_area) * t^n
    где alpha - коэффициент влияния границ зерен
    """
    boundary_effect = 1 + alpha / grain_area
    return d0 + k * boundary_effect * (t ** n)

def boundary_density_model(t, k, n, grain_area, beta=0.1, d0=0):
    """
    Альтернативная модель: влияние через плотность границ
    d = d0 + k * (1 + beta * (1/grain_area)) * t^n
    """
    boundary_density_effect = 1 + beta * (1 / grain_area)
    return d0 + k * boundary_density_effect * (t ** n)

# Основная программа Streamlit
st.title("Улучшенная модель роста σ-фазы с учетом размера зерна")

# Показываем таблицу ГОСТ
with st.expander("Данные ГОСТ 5639-82 о размерах зерен"):
    st.dataframe(grain_df)
    st.markdown("""
    **Ключевая идея:** Меньшая площадь зерна → больше границ зерен → больше мест зарождения σ-фазы → ускоренный рост
    """)

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите CSV с колонками: G, T, t, d", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Данные загружены!")
    st.dataframe(df.head())
    
    # Объединяем с данными о размере зерна
    df_enriched = df.merge(grain_df, left_on='G', right_on='grain_size', how='left')
    
    # Анализ влияния размера зерна
    st.subheader("Анализ влияния размера зерна на рост σ-фазы")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График 1: Зависимость диаметра от времени для разных размеров зерна
    for grain_size in df_enriched['G'].unique():
        subset = df_enriched[df_enriched['G'] == grain_size]
        grain_area = subset['grain_area'].iloc[0]
        label = f'Зерно {grain_size} (S={grain_area:.4f} мм²)'
        
        ax1.scatter(subset['t'], subset['d'], label=label, alpha=0.7)
        
        # Линия тренда
        if len(subset) > 1:
            z = np.polyfit(subset['t'], subset['d'], 1)
            p = np.poly1d(z)
            ax1.plot(subset['t'], p(subset['t']), linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Время (часы)')
    ax1.set_ylabel('Диаметр σ-фазы (мкм)')
    ax1.set_title('Влияние размера зерна на кинетику роста')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Зависимость скорости роста от площади зерна
    growth_rates = []
    grain_areas = []
    
    for grain_size in df_enriched['G'].unique():
        subset = df_enriched[df_enriched['G'] == grain_size]
        if len(subset) > 1:
            # Оценка скорости роста (производная)
            time_sorted = np.sort(subset['t'].unique())
            if len(time_sorted) >= 2:
                diameters = [subset[subset['t'] == t]['d'].mean() for t in time_sorted]
                growth_rate = (diameters[-1] - diameters[0]) / (time_sorted[-1] - time_sorted[0])
                growth_rates.append(growth_rate)
                grain_areas.append(subset['grain_area'].iloc[0])
    
    if growth_rates:
        ax2.scatter(grain_areas, growth_rates, s=80, alpha=0.7)
        ax2.set_xlabel('Площадь зерна (мм²)')
        ax2.set_ylabel('Скорость роста (мкм/час)')
        ax2.set_title('Зависимость скорости роста от площади зерна')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Линия тренда
        if len(growth_rates) > 1:
            z = np.polyfit(np.log(grain_areas), growth_rates, 1)
            x_trend = np.logspace(np.log10(min(grain_areas)), np.log10(max(grain_areas)), 100)
            y_trend = z[0] * np.log(x_trend) + z[1]
            ax2.plot(x_trend, y_trend, 'r--', alpha=0.7, label='Тренд')
            ax2.legend()
    
    st.pyplot(fig)
    
    # Калибровка улучшенной модели
    st.subheader("Калибровка модели с учетом размера зерна")
    
    # Подготовка данных для подбора
    X = df_enriched[['t', 'grain_area']].values
    y = df_enriched['d'].values
    
    def model_for_fit(X, k, n, alpha):
        t, grain_area = X[:, 0], X[:, 1]
        return enhanced_growth_model(t, k, n, grain_area, alpha)
    
    try:
        # Подбор параметров
        popt, pcov = curve_fit(model_for_fit, X, y, 
                              p0=[0.1, 0.5, 0.1], 
                              bounds=([0, 0, 0], [10, 2, 10]))
        
        k_opt, n_opt, alpha_opt = popt
        
        st.success("Параметры модели успешно определены!")
        st.write(f"- Кинетический коэффициент k = {k_opt:.4f}")
        st.write(f"- Показатель степени n = {n_opt:.4f}")
        st.write(f"- Коэффициент влияния границ α = {alpha_opt:.4f}")
        
        # Визуализация предсказаний
        fig2, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_enriched['G'].unique())))
        
        for i, grain_size in enumerate(df_enriched['G'].unique()):
            subset = df_enriched[df_enriched['G'] == grain_size]
            grain_area = subset['grain_area'].iloc[0]
            
            # Экспериментальные данные
            ax.scatter(subset['t'], subset['d'], color=colors[i], 
                      label=f'Зерно {grain_size}', alpha=0.7)
            
            # Предсказания модели
            t_range = np.linspace(subset['t'].min(), subset['t'].max(), 100)
            d_pred = enhanced_growth_model(t_range, k_opt, n_opt, grain_area, alpha_opt)
            ax.plot(t_range, d_pred, color=colors[i], linestyle='--')
        
        ax.set_xlabel('Время (часы)')
        ax.set_ylabel('Диаметр σ-фазы (мкм)')
        ax.set_title('Сравнение экспериментальных данных и модели')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # Сохранение параметров
        st.session_state['enhanced_params'] = {
            'k': k_opt, 'n': n_opt, 'alpha': alpha_opt
        }
        
    except Exception as e:
        st.error(f"Ошибка при подборе параметров: {e}")

# Инструмент для прогнозирования
st.subheader("Прогнозирование роста σ-фазы")

col1, col2, col3 = st.columns(3)
with col1:
    time_pred = st.number_input("Время (часы)", value=5000)
with col2:
    temp_pred = st.number_input("Температура (°C)", value=600)
with col3:
    grain_pred = st.selectbox("Номер зерна", options=grain_df['grain_size'])

if st.button("Рассчитать прогноз") and 'enhanced_params' in st.session_state:
    params = st.session_state['enhanced_params']
    grain_area = grain_df[grain_df['grain_size'] == grain_pred]['grain_area'].iloc[0]
    
    diameter_pred = enhanced_growth_model(time_pred, params['k'], params['n'], 
                                        grain_area, params['alpha'])
    
    st.success(f"Прогнозируемый диаметр σ-фазы: **{diameter_pred:.2f} мкм**")
    
    # Дополнительный анализ
    st.write("**Анализ влияния размера зерна:**")
    st.write(f"- Площадь зерна: {grain_area:.6f} мм²")
    st.write(f"- Эффект границ: {params['alpha']/grain_area:.2f}x ускорение")
