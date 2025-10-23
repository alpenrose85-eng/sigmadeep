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

st.title("🔬 Поэтапный расчет кинетики роста σ-фазы")
st.markdown("""
**Физическая модель:** Диффузионно-контролируемое укрупнение частиц σ-фазы
- Степенной закон роста: $d^n - d_0^n = K \\cdot t$
- Температурная зависимость по Аррениусу: $K = K_0(G) \\cdot \\exp(-Q/RT)$
- Влияние размера зерна: $\\ln K_0(G) = a_0 + a_1 \\cdot Z(G)$
""")

# Загрузка данных
st.header("1. Загрузка и подготовка данных")

uploaded_file = st.file_uploader("Загрузите файл с данными (CSV или Excel)", 
                               type=['csv', 'xlsx', 'xls'])

# Вопросы к пользователю
st.subheader("Вопросы для уточнения модели:")
has_sigma_content = st.radio("Есть ли данные о процентном содержании σ-фазы?", 
                           ["Да", "Нет"])
initial_diameter = st.number_input("Начальный диаметр d₀ (мкм)", 
                                 value=0.0, min_value=0.0, step=0.1,
                                 help="Диаметр на минимальной наработке или близкий к 0")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        required_cols = ['G', 'T', 't', 'd']
        if has_sigma_content == "Да":
            required_cols.append('sigma_content')
        
        if all(col in df.columns for col in required_cols):
            st.session_state['experimental_data'] = df
            st.success("✅ Данные успешно загружены!")
            
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
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"❌ Отсутствуют колонки: {missing}")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")

# Основной расчет
if 'experimental_data' in st.session_state:
    df = st.session_state['experimental_data']
    
    # Подготовка данных
    st.header("2. Подготовка данных")
    df_prep = df.copy()
    df_prep['T_K'] = df_prep['T'] + 273.15  # Температура в Кельвинах
    df_prep = df_prep.merge(grain_df, left_on='G', right_on='grain_size', how='left')
    
    st.write("**Данные с температурой в Кельвинах и информацией о зернах:**")
    st.dataframe(df_prep.head())
    
    # Выбор показателя степени n
    st.header("3. Выбор показателя степени n")
    
    st.markdown("""
    **Тестируемые значения n:**
    - n = 3: Объемная диффузия (классический LSW)
    - n = 4: Диффузия по границам зерен (ожидается для σ-фазы)
    - n = 3.5: Промежуточный механизм
    """)
    
    n_candidates = [3.0, 3.5, 4.0]
    n_results = {}
    
    # Анализ для каждого кандидата n
    for n in n_candidates:
        st.subheader(f"Анализ для n = {n}")
        
        # Для каждого сочетания T и G
        combinations = df_prep.groupby(['T', 'G']).size().reset_index()[['T', 'G']]
        k_values = []
        
        for _, row in combinations.iterrows():
            temp, grain = row['T'], row['G']
            subset = df_prep[(df_prep['T'] == temp) & (df_prep['G'] == grain)]
            
            if len(subset) >= 2:
                # Вычисляем d^n - d₀^n
                d_transformed = subset['d']**n - initial_diameter**n
                
                # Линейная регрессия: (d^n - d₀^n) = K * t
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    subset['t'], d_transformed
                )
                
                k_values.append({
                    'T': temp,
                    'T_K': temp + 273.15,
                    'G': grain,
                    'K': max(slope, 1e-10),  # Избегаем отрицательных значений
                    'R2': r_value**2,
                    'std_err': std_err,
                    'grain_area': subset['grain_area'].iloc[0]
                })
        
        if k_values:
            k_df = pd.DataFrame(k_values)
            n_results[n] = k_df
            
            # Среднее R² для этого n
            mean_r2 = k_df['R2'].mean()
            st.write(f"Средний R² = {mean_r2:.4f}")
            
            # Визуализация для нескольких комбинаций
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # График 1: Линейность преобразованных данных
            for i, (temp, grain) in enumerate(combinations.head(4).itertuples(index=False)):
                subset = df_prep[(df_prep['T'] == temp) & (df_prep['G'] == grain)]
                if len(subset) >= 2:
                    d_transformed = subset['d']**n - initial_diameter**n
                    axes[0].scatter(subset['t'], d_transformed, 
                                  label=f'T={temp}°C, G={grain}', alpha=0.7)
                    
                    # Линия регрессии
                    slope = k_df[(k_df['T'] == temp) & (k_df['G'] == grain)]['K'].iloc[0]
                    t_range = np.linspace(subset['t'].min(), subset['t'].max(), 100)
                    axes[0].plot(t_range, slope * t_range, '--', alpha=0.7)
            
            axes[0].set_xlabel('Время t (часы)')
            axes[0].set_ylabel(f'$d^{{{n}}} - d_0^{{{n}}}$')
            axes[0].set_title(f'Линейность при n = {n}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # График 2: Качество подбора по всем точкам
            all_r2 = []
            for _, row in k_df.iterrows():
                all_r2.append(row['R2'])
            
            axes[1].hist(all_r2, bins=10, alpha=0.7, edgecolor='black')
            axes[1].axvline(np.mean(all_r2), color='red', linestyle='--', 
                          label=f'Среднее R² = {np.mean(all_r2):.3f}')
            axes[1].set_xlabel('R²')
            axes[1].set_ylabel('Частота')
            axes[1].set_title('Распределение R² по всем комбинациям T,G')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    # Выбор оптимального n
    st.subheader("Выбор оптимального n")
    if n_results:
        n_comparison = []
        for n, k_df in n_results.items():
            n_comparison.append({
                'n': n,
                'Средний R²': k_df['R2'].mean(),
                'Минимальный R²': k_df['R2'].min(),
                'Количество точек': len(k_df)
            })
        
        n_comp_df = pd.DataFrame(n_comparison)
        st.dataframe(n_comp_df.style.format({
            'Средний R²': '{:.4f}',
            'Минимальный R²': '{:.4f}'
        }).highlight_max(subset=['Средний R²']))
        
        # Автоматический выбор n с максимальным средним R²
        best_n_row = n_comp_df.loc[n_comp_df['Средний R²'].idxmax()]
        best_n = best_n_row['n']
        st.success(f"🎯 **Рекомендуемый показатель степени: n = {best_n}**")
        st.info(f"*Обоснование: максимальный средний R² = {best_n_row['Средний R²']:.4f}*")
        
        # Сохраняем лучший результат
        st.session_state['best_n'] = best_n
        st.session_state['k_values'] = n_results[best_n]
        
        # Анализ Аррениуса
        st.header("4. Температурная зависимость (уравнение Аррениуса)")
        
        k_df = n_results[best_n]
        
        # Для каждого номера зерна строим зависимость Аррениуса
        arrhenius_results = {}
        
        for grain in k_df['G'].unique():
            st.subheader(f"Анализ Аррениуса для зерна {grain}")
            
            grain_data = k_df[k_df['G'] == grain]
            
            if len(grain_data) >= 2:  # Нужно минимум 2 температуры
                # Линейная регрессия: ln(K) = ln(K₀) - (Q/R) * (1/T)
                x = 1 / grain_data['T_K']
                y = np.log(grain_data['K'])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                Q = -slope * R  # Энергия активации
                K0 = np.exp(intercept)
                
                arrhenius_results[grain] = {
                    'Q': Q,
                    'K0': K0,
                    'R2': r_value**2,
                    'grain_area': grain_data['grain_area'].iloc[0]
                }
                
                # График Аррениуса
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(x, y, s=80, label='Экспериментальные точки')
                
                # Линия регрессии
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r--', 
                       label=f'Регрессия: Q = {Q:.0f} Дж/моль\nR² = {r_value**2:.4f}')
                
                ax.set_xlabel('1/T (1/K)')
                ax.set_ylabel('ln(K)')
                ax.set_title(f'Уравнение Аррениуса для зерна {grain}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                st.write(f"**Результаты для зерна {grain}:**")
                st.write(f"- Энергия активации Q = {Q:.0f} Дж/моль")
                st.write(f"- Предэкспонента K₀ = {K0:.6f}")
                st.write(f"- R² = {r_value**2:.4f}")
        
        # Учет номера зерна
        st.header("5. Учет влияния номера зерна")
        
        if arrhenius_results:
            # Создаем DataFrame для регрессии
            grain_effect_data = []
            for grain, results in arrhenius_results.items():
                grain_effect_data.append({
                    'G': grain,
                    'ln_K0': np.log(results['K0']),
                    'grain_area': results['grain_area'],
                    'Q': results['Q']
                })
            
            grain_effect_df = pd.DataFrame(grain_effect_data)
            
            st.write("**Данные для анализа влияния зерна:**")
            st.dataframe(grain_effect_df)
            
            # Выбор метрики Z(G)
            z_metric = st.selectbox("Выберите метрику для Z(G):",
                                  ['grain_area', 'G'],
                                  format_func=lambda x: 'Площадь зерна' if x == 'grain_area' else 'Номер зерна')
            
            # Множественная регрессия
            X = grain_effect_df[z_metric].values
            y = grain_effect_df['ln_K0'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            
            st.success("**Результаты регрессии:**")
            st.write(f"- a₀ = {intercept:.4f}")
            st.write(f"- a₁ = {slope:.4f}")
            st.write(f"- R² = {r_value**2:.4f}")
            
            # График зависимости
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(X, y, s=80, label='Экспериментальные точки')
            
            x_fit = np.linspace(X.min(), X.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'r--', 
                   label=f'Регрессия: ln(K₀) = {intercept:.3f} + {slope:.3f}·Z(G)')
            
            x_label = 'Площадь зерна (мм²)' if z_metric == 'grain_area' else 'Номер зерна G'
            ax.set_xlabel(x_label)
            ax.set_ylabel('ln(K₀)')
            ax.set_title('Зависимость предэкспоненты от размера зерна')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Сохраняем финальные параметры
            st.session_state['final_params'] = {
                'n': best_n,
                'Q': grain_effect_df['Q'].mean(),  # Средняя энергия активации
                'a0': intercept,
                'a1': slope,
                'z_metric': z_metric,
                'd0': initial_diameter
            }
            
            # Обратный расчет температуры
            st.header("6. Обратный расчет температуры эксплуатации")
            
            st.markdown("""
            **Формула для обратного расчета:**
            $$
            T = \\frac{Q}{R \\cdot (a_0 + a_1 \\cdot Z(G) - \\ln\\left(\\frac{d^n - d_0^n}{t}\\right))}
            $$
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                d_obs = st.number_input("Наблюдаемый диаметр d (мкм)", 
                                      value=5.0, min_value=0.1, step=0.1)
            with col2:
                t_obs = st.number_input("Время эксплуатации t (часы)", 
                                      value=5000, min_value=1, step=100)
            with col3:
                g_obs = st.selectbox("Номер зерна G", 
                                   options=sorted(df_prep['G'].unique()))
            
            if st.button("Рассчитать температуру"):
                params = st.session_state['final_params']
                
                # Находим Z(G)
                if params['z_metric'] == 'grain_area':
                    z_value = grain_df[grain_df['grain_size'] == g_obs]['grain_area'].iloc[0]
                else:
                    z_value = g_obs
                
                # Вычисляем K_obs
                k_obs = (d_obs**params['n'] - params['d0']**params['n']) / t_obs
                
                # Вычисляем температуру
                denominator = R * (params['a0'] + params['a1'] * z_value - np.log(max(k_obs, 1e-10)))
                if denominator > 0:
                    T_K = params['Q'] / denominator
                    T_C = T_K - 273.15
                    
                    st.success(f"**Расчетная температура эксплуатации: {T_C:.1f}°C**")
                    
                    # Детали расчета
                    st.write("**Детали расчета:**")
                    st.write(f"- K_obs = {k_obs:.6f}")
                    st.write(f"- ln(K_obs) = {np.log(k_obs):.4f}")
                    st.write(f"- Z(G) = {z_value:.6f}")
                    st.write(f"- Знаменатель = {denominator:.4f}")
                else:
                    st.error("Ошибка расчета: отрицательный знаменатель")
            
            # Выгрузка результатов
            st.header("7. Выгрузка результатов")
            
            if st.button("📊 Сгенерировать полный отчет"):
                output_buffer = io.BytesIO()
                
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                    # Исходные данные
                    df_prep.to_excel(writer, sheet_name='Исходные_данные', index=False)
                    
                    # Параметры K для лучшего n
                    k_df.to_excel(writer, sheet_name='Кинетические_коэффициенты', index=False)
                    
                    # Результаты Аррениуса
                    arrhenius_df = pd.DataFrame([
                        {**{'G': g}, **v} for g, v in arrhenius_results.items()
                    ])
                    arrhenius_df.to_excel(writer, sheet_name='Аррениус_анализ', index=False)
                    
                    # Финальные параметры модели
                    final_params_df = pd.DataFrame([st.session_state['final_params']])
                    final_params_df.to_excel(writer, sheet_name='Финальные_параметры', index=False)
                
                output_buffer.seek(0)
                
                st.download_button(
                    label="💾 Скачать полный отчет Excel",
                    data=output_buffer,
                    file_name="полный_отчет_сигма_фаза.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Информация о модели
with st.expander("📚 Теоретическая справка"):
    st.markdown("""
    **Физическая модель диффузионно-контролируемого роста:**
    
    1. **Степенной закон роста:**
       $$
       d^n - d_0^n = K \\cdot t
       $$
    
    2. **Температурная зависимость (Аррениус):**
       $$
       K = K_0(G) \\cdot \\exp\\left(-\\frac{Q}{RT}\\right)
       $$
    
    3. **Влияние размера зерна:**
       $$
       \\ln K_0(G) = a_0 + a_1 \\cdot Z(G)
       $$
    
    **Объединенная модель:**
    $$
    \\ln\\left(\\frac{d^n - d_0^n}{t}\\right) = a_0 + a_1 \\cdot Z(G) - \\frac{Q}{RT}
    $$
    
    **Ожидаемые значения параметров:**
    - n ≈ 4.0 (диффузия по границам зерен)
    - Q ≈ 200-300 кДж/моль для сталей
    - a₁ > 0 (мелкое зерно ускоряет рост)
    """)
