import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from prophet import Prophet
import numpy as np
import calendar

# --- Configuration ---
st.set_page_config(
    page_title="Dynamic Expense Tracker",
    page_icon="💰",
    layout="wide"
)

# --- Google Sheets Connection ---
try:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    client = gspread.authorize(creds)

    SPREADSHEET_NAME = st.secrets["gcp_service_account"]["sheet_name"]
    spreadsheet = client.open(SPREADSHEET_NAME)
    
    try:
        worksheet = spreadsheet.worksheet("Expenses")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title="Expenses", rows="1", cols=4)
        worksheet.append_row(["Date", "Category", "Amount", "Comments"])

    GDRIVE_CONNECTED = True

except Exception as e:
    st.error(f"Failed to connect to Google Sheets. Please check your secrets configuration. Error: {e}")
    GDRIVE_CONNECTED = False


def load_data():
    """Load expense data from the Google Sheet."""
    if not GDRIVE_CONNECTED:
        return pd.DataFrame(columns=["Date", "Category", "Amount", "Comments"])
        
    try:
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        for col in ["Date", "Category", "Amount", "Comments"]:
            if col not in df.columns:
                df[col] = None
        
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
        df['id'] = range(len(df))
        return df

    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame(columns=["Date", "Category", "Amount", "Comments"])


def save_data(df):
    """Save the DataFrame back to the Google Sheet."""
    if not GDRIVE_CONNECTED:
        st.error("Cannot save data. Connection to Google Drive failed.")
        return
    
    df_to_save = df.drop(columns=['id'], errors='ignore')
    if not df_to_save.empty:
        df_to_save['Date'] = pd.to_datetime(df_to_save['Date']).dt.strftime('%Y-%m-%d')
    
    worksheet.clear()
    set_with_dataframe(worksheet, df_to_save, include_index=False, resize=True)

# --- Data Science: Forecasting Function ---
def predict_month_end(df):
    """
    Uses Facebook Prophet to predict expenses for the REST of the current month.
    """
    # 1. Prepare data
    daily_expenses = df.groupby("Date")['Amount'].sum().reset_index()
    daily_expenses.columns = ['ds', 'y']
    
    if len(daily_expenses) < 14:
        return None, "Not enough data points (days) to forecast.", None

    # 1.5 Remove Outliers (IQR Method) - Keep this to stabilize the trend line
    Q1 = daily_expenses['y'].quantile(0.25)
    Q3 = daily_expenses['y'].quantile(0.75)
    IQR = Q3 - Q1
    cap = Q3 + 1.5 * IQR 
    train_data = daily_expenses[daily_expenses['y'] <= cap].copy()
    
    outliers_removed = len(daily_expenses) - len(train_data)
    debug_msg = f"Filtered {outliers_removed} outliers (> ₹{cap:,.0f}) for trend calculation."

    # 2. Initialize and Train
    # Reverting interval_width to default (0.80) because we fixed the aggregation math instead
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.001, # Flat trend assumption
        seasonality_prior_scale=0.1
    )
    m.fit(train_data)

    # 3. Calculate Days Remaining in Current Month
    today = datetime.now()
    last_day_of_month = calendar.monthrange(today.year, today.month)[1]
    days_remaining = last_day_of_month - today.day
    
    if days_remaining <= 0:
        return None, "It's the end of the month! No days left to forecast.", None

    # 4. Predict
    future = m.make_future_dataframe(periods=days_remaining)
    forecast = m.predict(future)
    
    # Clip negative predictions
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    return forecast, None, debug_msg


# --- Main App ---
st.title("💰 Dynamic Expense Tracker")
st.write("This app saves your expenses to a centralized Google Sheet, accessible from any device.")

if GDRIVE_CONNECTED:
    if 'expenses_df' not in st.session_state:
        st.session_state.expenses_df = load_data()

    # --- Define Tabs ---
    tab1, tab2, tab3 = st.tabs(["📝 Add Expense", "📊 Monthly Dashboard", "📈 Analysis & AI Forecast"])

    # --- Tab 1: Add Expense ---
    with tab1:
        st.header("Add a New Expense")
        with st.form("expense_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                expense_date = st.date_input("Date of Expense", datetime.now())
            with col2:
                expense_category = st.selectbox(
                    "Expense Category",
                    [
                        "❤️ Girlfriend", "⛽ Fuel & Bike Service", "📱 Recharge & Subscriptions", "☕ Chai & snacks", "⚽ Sports",
                        "🍔 Food & Dining", "🛒 Groceries", "🚗 Transportation", "💡 Utilities", 
                        "🏠 Housing", "🛍️ Shopping", "🎬 Entertainment", "💪 Health & Fitness", 
                        "💄 Personal Care", "🎓 Education", "🎁 Gifts & Donations", "✈️ Travel", 
                        "Miscellaneous"
                    ]
                )
            expense_amount = st.number_input("Amount", min_value=0.01, format="%.2f")
            expense_comments = st.text_area("Comments (Optional)")
            
            submitted = st.form_submit_button("Add Expense")
            if submitted:
                new_expense = pd.DataFrame([{
                    "Date": pd.to_datetime(expense_date),
                    "Category": expense_category,
                    "Amount": expense_amount,
                    "Comments": expense_comments
                }])
                updated_df = pd.concat([st.session_state.expenses_df, new_expense], ignore_index=True)
                st.session_state.expenses_df = updated_df
                save_data(st.session_state.expenses_df)
                st.success("Expense added successfully and saved to Google Sheets!")
                st.rerun()

    # --- Tab 2: Monthly Dashboard ---
    with tab2:
        st.header("Monthly Dashboard")
        if not st.session_state.expenses_df.empty:
            current_year = datetime.now().year
            year_list = sorted(st.session_state.expenses_df['Date'].dt.year.unique(), reverse=True)
            selected_year = st.selectbox("Select Year", year_list if year_list else [current_year])
            
            selected_month_name = st.selectbox("Select Month", [
                "January", "February", "March", "April", "May", "June", 
                "July", "August", "September", "October", "November", "December"
            ])
            month_num = datetime.strptime(selected_month_name, "%B").month

            monthly_df = st.session_state.expenses_df[
                (st.session_state.expenses_df['Date'].dt.year == selected_year) &
                (st.session_state.expenses_df['Date'].dt.month == month_num)
            ]

            if monthly_df.empty:
                st.info(f"No expenses recorded for {selected_month_name} {selected_year}.")
            else:
                total_monthly_expense = monthly_df['Amount'].sum()
                st.metric(label=f"Total Expenses for {selected_month_name} {selected_year}", value=f"₹{total_monthly_expense:,.2f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Expenses by Category (Pie)")
                    category_summary = monthly_df.groupby("Category")["Amount"].sum().reset_index()
                    fig_pie = px.pie(category_summary, values='Amount', names='Category')
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    st.subheader("Expenses by Category (Bar)")
                    fig_bar = px.bar(category_summary.sort_values('Amount', ascending=False), x='Category', y='Amount')
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No expenses recorded yet.")

    # --- Tab 3: Analysis & AI Forecast ---
    with tab3:
        st.header("Expense Analysis & AI Forecast")
        if not st.session_state.expenses_df.empty:
            
            # 1. Standard Bar Charts
            analysis_df = st.session_state.expenses_df.copy()
            analysis_df['Month'] = analysis_df['Date'].dt.to_period('M').astype(str)
            monthly_summary = analysis_df.groupby('Month')['Amount'].sum().reset_index()
            
            st.subheader("📊 Historical Trends")
            col1, col2 = st.columns(2)
            with col1:
                fig_monthly = px.bar(monthly_summary, x='Month', y='Amount', title='Monthly Expense Trend', labels={'Amount': 'Total Amount (₹)'})
                st.plotly_chart(fig_monthly, use_container_width=True)
            with col2:
                analysis_df['Quarter'] = analysis_df['Date'].dt.to_period('Q').astype(str)
                quarterly_summary = analysis_df.groupby('Quarter')['Amount'].sum().reset_index()
                fig_quarterly = px.bar(quarterly_summary, x='Quarter', y='Amount', title='Quarterly Expense Trend', labels={'Amount': 'Total Amount (₹)'})
                st.plotly_chart(fig_quarterly, use_container_width=True)
            
            st.divider()

            # 2. AI Forecasting Section (Current Month Projection)
            current_month_name = datetime.now().strftime("%B")
            st.subheader(f"🔮 End-of-Month Projection ({current_month_name})")
            
            unique_days_count = st.session_state.expenses_df['Date'].nunique()
            
            if unique_days_count < 14:
                st.info(f"⏳ **AI Forecast is unlocking... (Progress: {unique_days_count}/14 days)**\n\nWe need at least 14 days of spending history to reliably predict your month-end total.")
                st.progress(unique_days_count / 14)
            else:
                st.write(f"Based on your spending so far in {current_month_name}, here is where you are likely to end up.")
                
                if st.button("Generate Projection"):
                    with st.spinner("Analyzing current month trend..."):
                        forecast, error_msg, debug_msg = predict_month_end(st.session_state.expenses_df)
                    
                    if error_msg:
                        st.warning(error_msg)
                    else:
                        # --- Calculate The Projection Range ---
                        today = datetime.now()
                        current_month_start = today.replace(day=1)
                        
                        # 1. Spent So Far (Actuals)
                        current_month_data = st.session_state.expenses_df[
                            (st.session_state.expenses_df['Date'] >= current_month_start) &
                            (st.session_state.expenses_df['Date'] <= today)
                        ]
                        spent_so_far = current_month_data['Amount'].sum()
                        
                        # 2. Predicted Remaining (Future)
                        future_forecast = forecast[forecast['ds'] > pd.to_datetime(today)]
                        
                        predicted_remaining_mid = future_forecast['yhat'].sum()
                        
                        # Correct Statistical Aggregation (Square Root Rule)
                        # Instead of adding up all worst-case days, we calculate the standard error of the sum.
                        # This assumes errors cancel out partially over time.
                        daily_uncertainty_gap = future_forecast['yhat_upper'] - future_forecast['yhat']
                        avg_daily_uncertainty = daily_uncertainty_gap.mean()
                        days_remaining_count = len(future_forecast)
                        
                        # Uncertainty of the Total = Daily Uncertainty * Sqrt(Days)
                        total_uncertainty = avg_daily_uncertainty * np.sqrt(days_remaining_count)
                        
                        # 3. Total Projected Range
                        total_mid = spent_so_far + predicted_remaining_mid
                        total_low = total_mid - total_uncertainty
                        total_high = total_mid + total_uncertainty

                        # --- Display Results ---
                        st.metric(f"Already Spent in {current_month_name}", f"₹{spent_so_far:,.2f}")
                        
                        st.success(f"🎯 **Projected Total for {current_month_name}: ₹{total_low:,.0f} — ₹{total_high:,.0f}**")
                        st.caption(f"Most likely outcome: ~₹{total_mid:,.0f}")
                        
                        if debug_msg:
                            st.caption(f"ℹ️ {debug_msg}")

                        # --- Visualization ---
                        fig_forecast = go.Figure()

                        # Plot historical data for this month
                        daily_actuals = current_month_data.groupby("Date")['Amount'].sum().reset_index()
                        fig_forecast.add_trace(go.Scatter(
                            x=daily_actuals['Date'], 
                            y=daily_actuals['Amount'],
                            mode='lines+markers',
                            name='Actual Spend',
                            line=dict(color='green', width=3)
                        ))
                        
                        # Plot forecast
                        fig_forecast.add_trace(go.Scatter(
                            x=future_forecast['ds'], 
                            y=future_forecast['yhat'],
                            mode='lines',
                            name='Projected Path',
                            line=dict(color='blue', dash='dot')
                        ))
                        
                        # Confidence Interval
                        fig_forecast.add_trace(go.Scatter(
                            x=future_forecast['ds'], 
                            y=future_forecast['yhat_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        fig_forecast.add_trace(go.Scatter(
                            x=future_forecast['ds'], 
                            y=future_forecast['yhat_lower'],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0, 0, 255, 0.2)',
                            line=dict(width=0),
                            name='Likely Range'
                        ))

                        fig_forecast.update_layout(
                            title=f"{current_month_name} Trajectory",
                            xaxis_title="Date",
                            yaxis_title="Daily Amount (₹)",
                            hovermode="x"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)

        else:
            st.info("No expenses recorded yet.")

    # --- Delete Expenses Section ---
    st.header("Manage Expenses")
    if not st.session_state.expenses_df.empty:
        with st.form("delete_form"):
            display_df = st.session_state.expenses_df.copy()
            display_df['Select'] = False
            cols = ['Select'] + [col for col in display_df.columns if col != 'Select' and col != 'id']
            display_df = display_df[cols]
            
            st.write("Select expenses to delete:")
            edited_df = st.data_editor(
                display_df,
                hide_index=True,
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=[col for col in display_df.columns if col != 'Select']
            )

            col1, col2 = st.columns(2)
            with col1:
                delete_button = st.form_submit_button("Delete Selected Expenses")
            with col2:
                delete_all_button = st.form_submit_button("⚠️ Delete All Expenses", type="primary")

            if delete_button:
                selected_rows = edited_df[edited_df.Select]
                if not selected_rows.empty:
                    indices_to_delete = st.session_state.expenses_df[
                        st.session_state.expenses_df['Date'].isin(pd.to_datetime(selected_rows['Date'])) &
                        st.session_state.expenses_df['Amount'].isin(selected_rows['Amount']) &
                        st.session_state.expenses_df['Category'].isin(selected_rows['Category'])
                    ].index
                    
                    st.session_state.expenses_df = st.session_state.expenses_df.drop(indices_to_delete)
                    save_data(st.session_state.expenses_df)
                    st.success(f"Successfully deleted {len(indices_to_delete)} expense(s).")
                    st.rerun()
                else:
                    st.warning("No expenses selected for deletion.")
            
            if delete_all_button:
                empty_df = pd.DataFrame(columns=["Date", "Category", "Amount", "Comments"])
                st.session_state.expenses_df = empty_df
                save_data(st.session_state.expenses_df)
                st.success("All expenses have been deleted.")
                st.rerun()
    else:
        st.info("No expenses to manage.")
