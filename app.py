
**Расшифровка параметров:**
- `f_max = {f_max:.3f}%` - максимальное содержание сигма-фазы
- `K0 = {K0:.3e}` - предэкспоненциальный множитель
- `Q = {Q/1000:.1f} кДж/моль` - энергия активации
- `n = {n:.3f}` - показатель степени в модели Аврами
- `α = {alpha:.3f}` - коэффициент влияния размера зерна
- `w = {w:.3f}` - вес степенного компонента
- `β = {beta:.0f}` - температурный коэффициент в степенном законе
"""
    
    def predict_temperature(self, G, sigma_percent, t, method="bisection"):
        """Предсказание температуры разными методами"""
        if self.params is None:
            raise ValueError("Модель не обучена!")
        
        sigma = sigma_percent
        
        if method == "bisection":
            return self._predict_temperature_bisection(G, sigma, t)
        else:
            return self._predict_temperature_analytic(G, sigma, t)
    
    def _predict_temperature_bisection(self, G, sigma, t, tol=1.0, max_iter=100):
        """Бисекционный поиск температуры"""
        T_min, T_max = 500, 900  # Реалистичный диапазон
        
        for i in range(max_iter):
            T_mid = (T_min + T_max) / 2
            f_pred = self._evaluate_model(G, T_mid, t)
            
            if abs(f_pred - sigma) < tol:
                return T_mid
            
            if f_pred < sigma:
                T_min = T_mid
            else:
                T_max = T_mid
        
        return (T_min + T_max) / 2
    
    def _evaluate_model(self, G, T, t):
        """Вычисление модели для данных параметров"""
        if self.params is None:
            return 0.0
            
        T_kelvin = T + 273.15
        f_max, K0, Q, n, alpha, w, beta = self.params
        R = 8.314
        
        # Аврами компонент
        grain_effect_avrami = 1 + alpha * (G - 8)
        K_avrami = K0 * np.exp(-Q / (R * T_kelvin)) * grain_effect_avrami
        f_avrami = f_max * (1 - np.exp(-K_avrami * (t ** n)))
        
        # Степенной компонент
        temp_effect_power = np.exp(beta / (R * T_kelvin))
        f_power = w * temp_effect_power * (t ** 0.5) * (1 + 0.05 * (G - 8))
        
        return f_avrami + f_power
    
    def calculate_validation_metrics(self, data):
        """Расчет метрик валидации"""
        if self.params is None:
            return None
        
        G = data['G'].values
        T = data['T'].values
        t = data['t'].values
        f_exp = data['f_exp (%)'].values
        
        f_pred = np.array([self._evaluate_model(g, temp, time) for g, temp, time in zip(G, T, t)])
        
        residuals = f_pred - f_exp
        relative_errors = (residuals / f_exp) * 100
        
        # Фильтрация бесконечных значений
        valid_mask = np.isfinite(relative_errors) & (f_exp > 0.1)
        f_exp_valid = f_exp[valid_mask]
        f_pred_valid = f_pred[valid_mask]
        residuals_valid = residuals[valid_mask]
        relative_errors_valid = relative_errors[valid_mask]
        
        if len(f_exp_valid) == 0:
            return None
            
        mae = np.mean(np.abs(residuals_valid))
        rmse = np.sqrt(mean_squared_error(f_exp_valid, f_pred_valid))
        mape = np.mean(np.abs(relative_errors_valid))
        r2 = r2_score(f_exp_valid, f_pred_valid)
        
        validation_results = {
            'data': data.copy(),
            'predictions': f_pred,
            'residuals': residuals,
            'relative_errors': relative_errors,
            'metrics': {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
        }
        
        return validation_results

def main():
    # Инициализация сессии
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'excluded_points' not in st.session_state:
        st.session_state.excluded_points = set()
    
    # Боковая панель
    st.sidebar.header("🎯 Управление данными")
    
    # Пример данных
    sample_data = pd.DataFrame({
        'G': [8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        'T': [600, 600, 600, 600, 650, 650, 650, 650, 600, 600, 600, 600, 650, 650, 650, 650, 600, 600, 600, 600, 650, 650, 650, 650, 700, 700],
        't': [2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000, 6000, 8000, 2000, 4000],
        'f_exp (%)': [1.76, 0.68, 0.94, 1.09, 0.67, 1.2, 1.48, 1.13, 0.87, 1.28, 2.83, 3.25, 1.88, 2.29, 3.25, 2.89, 1.261, 2.04, 2.38, 3.3, 3.2, 4.26, 5.069, 5.41, 3.3, 5.0]
    })
    
    # Используем примерные данные
    if st.session_state.current_data is None:
        st.session_state.current_data = sample_data
    
    # Основной интерфейс
    st.header("🔧 Улучшенное моделирование с ручным выбором точек")
    
    st.info("""
    **Новые возможности:**
    - Ручное исключение точек из анализа
    - Показ финальной формулы модели
    - Сравнение результатов с исключенными точками и без
    """)
    
    # Разделение на две колонки
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Данные для анализа")
        
        # Создаем копию данных для редактирования
        display_data = st.session_state.current_data.copy()
        display_data['Включить'] = [i not in st.session_state.excluded_points for i in range(len(display_data))]
        
        # Редактор данных с возможностью исключения точек
        edited_data = st.data_editor(
            display_data,
            column_config={
                "Включить": st.column_config.CheckboxColumn(
                    "Включить в анализ",
                    help="Снимите галочку чтобы исключить точку из анализа"
                ),
                "f_exp (%)": st.column_config.NumberColumn(format="%.3f"),
                "G": st.column_config.NumberColumn(format="%d"),
                "T": st.column_config.NumberColumn(format="%.1f"),
                "t": st.column_config.NumberColumn(format="%d")
            },
            disabled=["G", "T", "t", "f_exp (%)"],
            use_container_width=True
        )
        
        # Обновляем список исключенных точек
        new_excluded = set()
        for i, included in enumerate(edited_data['Включить']):
            if not included:
                new_excluded.add(i)
        
        if new_excluded != st.session_state.excluded_points:
            st.session_state.excluded_points = new_excluded
            st.session_state.analyzer = None
            st.session_state.validation_results = None
            st.rerun()
    
    with col2:
        st.header("📈 Статистика данных")
        
        total_points = len(st.session_state.current_data)
        excluded_count = len(st.session_state.excluded_points)
        included_count = total_points - excluded_count
        
        st.metric("Всего точек", total_points)
        st.metric("Включено в анализ", included_count)
        st.metric("Исключено", excluded_count)
        
        if excluded_count > 0:
            st.info(f"Исключенные точки: {sorted(st.session_state.excluded_points)}")
            
            if st.button("🔄 Включить все точки"):
                st.session_state.excluded_points = set()
                st.session_state.analyzer = None
                st.session_state.validation_results = None
                st.rerun()
    
    # Подготовка данных для анализа (исключаем выбранные точки)
    analysis_data = st.session_state.current_data.copy()
    if st.session_state.excluded_points:
        analysis_data = analysis_data.drop(list(st.session_state.excluded_points)).reset_index(drop=True)
    
    # Подбор модели
    st.header("🎯 Подбор параметров модели")
    
    if st.button("🚀 Запустить подбор параметров", use_container_width=True):
        analyzer = AdvancedSigmaPhaseAnalyzer()
        
        with st.spinner("Подбираем параметры ансамблевой модели..."):
            success = analyzer.fit_ensemble_model(analysis_data)
        
        if success:
            st.session_state.analyzer = analyzer
            validation_results = analyzer.calculate_validation_metrics(analysis_data)
            st.session_state.validation_results = validation_results
            
            st.success(f"✅ Модель успешно обучена! R² = {analyzer.R2:.4f}")
    
    # Показ результатов
    if st.session_state.analyzer is not None and st.session_state.validation_results is not None:
        analyzer = st.session_state.analyzer
        validation = st.session_state.validation_results
        
        # Разделение на колонки для результатов
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📊 Результаты валидации")
            
            metrics = validation['metrics']
            
            st.metric("R²", f"{metrics['R2']:.4f}")
            st.metric("MAE", f"{metrics['MAE']:.3f}%")
            st.metric("RMSE", f"{metrics['RMSE']:.3f}%")
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            # Оценка качества
            if metrics['MAPE'] < 15:
                st.success("✅ Отличное качество модели!")
            elif metrics['MAPE'] < 25:
                st.warning("⚠️ Удовлетворительное качество модели")
            else:
                st.error("❌ Попробуйте исключить больше точек или проверить данные")
            
            # График предсказаний
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=validation['data']['f_exp (%)'],
                y=validation['predictions'],
                mode='markers',
                name='Предсказания',
                marker=dict(size=10, color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=[0, 6], y=[0, 6],
                mode='lines',
                name='Идеально',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title='Предсказание vs Эксперимент',
                xaxis_title='Экспериментальные значения (%)',
                yaxis_title='Расчетные значения (%)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.header("🧮 Финальная формула")
            st.markdown(analyzer.final_formula)
            
            # Калькулятор температуры
            st.header("🔍 Калькулятор температуры")
            
            G_input = st.number_input("Номер зерна (G)", value=8.0, min_value=-3.0, max_value=14.0, step=0.1)
            sigma_input = st.number_input("Содержание сигма-фазы (%)", value=2.0, min_value=0.0, max_value=20.0, step=0.1)
            t_input = st.number_input("Время эксплуатации (ч)", value=4000, min_value=100, max_value=500000, step=100)
            
            if st.button("Рассчитать температуру"):
                try:
                    T_pred = analyzer.predict_temperature(G_input, sigma_input, t_input)
                    if T_pred is not None:
                        st.success(f"**Расчетная температура:** {T_pred:.1f}°C")
                    else:
                        st.error("Не удалось рассчитать температуру")
                except Exception as e:
                    st.error(f"Ошибка расчета: {e}")
        
        # Детальная таблица сравнения
        st.header("📋 Детальное сравнение")
        
        comparison_df = validation['data'].copy()
        comparison_df['f_pred (%)'] = validation['predictions']
        comparison_df['Абс. ошибка (%)'] = validation['residuals']
        comparison_df['Отн. ошибка (%)'] = validation['relative_errors']
        comparison_df['f_pred (%)'] = comparison_df['f_pred (%)'].round(3)
        comparison_df['Абс. ошибка (%)'] = comparison_df['Абс. ошибка (%)'].round(3)
        comparison_df['Отн. ошибка (%)'] = comparison_df['Отн. ошибка (%)'].round(1)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Скачивание результатов
        st.header("💾 Экспорт результатов")
        
        if st.button("Скачать результаты анализа"):
            results = {
                'parameters': analyzer.params.tolist() if analyzer.params is not None else [],
                'metrics': validation['metrics'],
                'formula': analyzer.final_formula,
                'data_used': analysis_data.to_dict(),
                'excluded_points': list(st.session_state.excluded_points)
            }
            
            results_json = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="Скачать JSON",
                data=results_json,
                file_name="sigma_phase_analysis.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
