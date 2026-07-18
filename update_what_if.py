import re

with open('dashboard.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Locate the What-If simulator block
what_if_start = "        # Get baseline features"
what_if_end = "# ══════════════════════════════════════════════════════════════════════════════\n# PAGE 7: MODEL COMPARISON (BENCHMARK)"

old_what_if = """        # Get baseline features
        base_feat = get_forecast_features(sel_br, target_dt)
        
        is_holiday = st.checkbox("Is Public Holiday?", value=bool(base_feat.get('Is_Holiday', False)))
        is_salary = st.checkbox("Is Salary Day?", value=bool(base_feat.get('Is_Salary_Day', False)))
        
        # Sliders for continuous variables
        st.markdown("**Historical Volume Adjustments**")
        mult_30 = st.slider("30-Day Avg Volume Multiplier", 0.5, 2.0, 1.0, 0.1)
        mult_7 = st.slider("7-Day Avg Volume Multiplier", 0.5, 2.0, 1.0, 0.1)
        
        if st.button("🚀 Run Simulation", use_container_width=True):
            with st.spinner("Simulating AI Model..."):
                # Apply changes
                sim_feat = base_feat.copy()
                if 'Is_Holiday' in sim_feat: sim_feat['Is_Holiday'] = int(is_holiday)
                if 'Is_Salary_Day' in sim_feat: sim_feat['Is_Salary_Day'] = int(is_salary)
                if 'rolling_30_mean_debit' in sim_feat: sim_feat['rolling_30_mean_debit'] *= mult_30
                if 'rolling_7_mean_debit' in sim_feat: sim_feat['rolling_7_mean_debit'] *= mult_7
                
                # Check for V3 Model (Optuna + Log Transform)
                try:
                    v3_model = joblib.load('models/v3/model_Half_Day_Total_Debit.pkl')
                    # V3 feature structure is slightly different (half daily). We will simulate using the V2 model for now, 
                    # but if V3 gets fully integrated into dashboard.py, we will use it here.
                    sim_model = model
                except:
                    sim_model = model
                
                # Baseline Prediction
                X_base = pd.DataFrame([base_feat])[feature_cols]
                base_pred = float(sim_model.predict(X_base)[0])
                
                # Simulated Prediction
                X_sim = pd.DataFrame([sim_feat])[feature_cols]
                sim_pred = float(sim_model.predict(X_sim)[0])
                
                diff = sim_pred - base_pred
                pct_change = (diff / base_pred) * 100 if base_pred > 0 else 0
                
                # Render Results
                with col2:
                    st.subheader("📊 Simulation Results")
                    st.markdown(f"Scenario for **Branch {sel_br}** on **{target_dt.strftime('%A, %d %B %Y')}**")
                    
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Original Prediction", f"{base_pred/1e6:.1f}M")
                    sc2.metric("Simulated Prediction", f"{sim_pred/1e6:.1f}M", f"{diff/1e6:+.1f}M ({pct_change:+.1f}%)", delta_color="inverse")
                    
                    st.markdown("---")
                    st.markdown("### Why did it change?")
                    
                    if diff > 0:
                        st.success(f"The simulation caused an INCREASE of {diff/1e6:.1f} Million PKR.")
                    elif diff < 0:
                        st.info(f"The simulation caused a DECREASE of {abs(diff)/1e6:.1f} Million PKR.")
                    else:
                        st.warning("The simulation caused NO CHANGE in the predicted amount.")
                        
                    st.markdown("*Note: The model intelligently weights these features. For example, declaring a holiday on a weekend might have a different impact than on a weekday.*")"""


new_what_if = """        # Get baseline features
        base_feat_am, base_feat_pm = get_forecast_features(sel_br, target_dt)
        if base_feat_am is None:
            st.error("Feature data not found. Please select a valid date.")
            st.stop()
        
        is_holiday = st.checkbox("Is Public Holiday?", value=bool(base_feat_am.get('Is_Holiday', False)))
        is_salary = st.checkbox("Is Salary Day?", value=bool(base_feat_am.get('Is_Salary_Day', False)))
        
        # Sliders for continuous variables
        st.markdown("**Historical Volume Adjustments**")
        mult_14 = st.slider("14-Day Avg Volume Multiplier", 0.5, 2.0, 1.0, 0.1)
        
        if st.button("🚀 Run Simulation", use_container_width=True):
            with st.spinner("Simulating AI Model..."):
                # Apply changes to both AM and PM
                sim_feat_am = base_feat_am.copy()
                sim_feat_pm = base_feat_pm.copy()
                
                for sf in [sim_feat_am, sim_feat_pm]:
                    if 'Is_Holiday' in sf: sf['Is_Holiday'] = int(is_holiday)
                    if 'Is_Salary_Day' in sf: sf['Is_Salary_Day'] = int(is_salary)
                    if 'rolling_14_mean_Half_Day_Total_Debit' in sf: sf['rolling_14_mean_Half_Day_Total_Debit'] *= mult_14
                
                sim_model = model  # model is already V3
                
                # Baseline Prediction
                X_base_am = pd.DataFrame([base_feat_am])[feature_cols]
                X_base_pm = pd.DataFrame([base_feat_pm])[feature_cols]
                base_pred = np.expm1(sim_model.predict(X_base_am)[0]) + np.expm1(sim_model.predict(X_base_pm)[0])
                
                # Simulated Prediction
                X_sim_am = pd.DataFrame([sim_feat_am])[feature_cols]
                X_sim_pm = pd.DataFrame([sim_feat_pm])[feature_cols]
                sim_pred = np.expm1(sim_model.predict(X_sim_am)[0]) + np.expm1(sim_model.predict(X_sim_pm)[0])
                
                diff = sim_pred - base_pred
                pct_change = (diff / base_pred) * 100 if base_pred > 0 else 0
                
                # Render Results
                with col2:
                    st.subheader("📊 Simulation Results")
                    st.markdown(f"Scenario for **Branch {sel_br}** on **{target_dt.strftime('%A, %d %B %Y')}**")
                    
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Original Prediction", f"{base_pred/1e6:.1f}M")
                    sc2.metric("Simulated Prediction", f"{sim_pred/1e6:.1f}M", f"{diff/1e6:+.1f}M ({pct_change:+.1f}%)", delta_color="inverse")
                    
                    st.markdown("---")
                    st.markdown("### Why did it change?")
                    
                    if diff > 0:
                        st.success(f"The simulation caused an INCREASE of {diff/1e6:.1f} Million PKR.")
                    elif diff < 0:
                        st.info(f"The simulation caused a DECREASE of {abs(diff)/1e6:.1f} Million PKR.")
                    else:
                        st.warning("The simulation caused NO CHANGE in the predicted amount.")
                        
                    st.markdown("*Note: The model intelligently weights these features. For example, declaring a holiday on a weekend might have a different impact than on a weekday.*")"""

if old_what_if in code:
    code = code.replace(old_what_if, new_what_if)
else:
    print("Warning: old what if not found!")

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("What-If updated!")
