import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from prophet import Prophet

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
def predict_next_30_days(df):
    """
    Uses Facebook Prophet to predict expenses for the next 30 days.
    """
    # 1. Prepare data for Prophet (needs 'ds' for date and 'y' for value)
    daily_expenses = df.groupby("Date")['Amount'].sum().reset_index()
    daily_expenses.columns = ['ds', 'y']
    
    # Redundant check (UI handles it), but good for safety
    if len(daily_expenses) < 14:
        return None, "Not enough data points (days) to forecast."

    # 2. Initialize and Train the Model
    # We tweak the parameters to be more conservative (less reactive to spikes)
    m = Prophet(
        daily_seasonality=True if len(daily_expenses) > 60 else False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.01  # Lower value = smoother trend, less overfitting to recent spikes
    )
    m.fit(daily_expenses)

    # 3. Create a dataframe for the future (30 days)
    future = m.make_future_dataframe(periods=30)
    
    # 4. Predict
    forecast = m.predict(future)
    
    # Clip negative predictions to 0 (impossible to spend negative money)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    return forecast, None


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

            # 2. AI Forecasting Section
            st.subheader("🔮 AI Spending Forecast (Next 30 Days)")
            
            # Data Validity Check for Prediction (Requires 14 Distinct Days)
            unique_days_count = st.session_state.expenses_df['Date'].nunique()
            
            if unique_days_count < 14:
                st.info(f"⏳ **AI Forecast is unlocking... (Progress: {unique_days_count}/14 days)**\n\nWe need at least 14 days of spending history (distinct dates) to understand your patterns and generate an accurate prediction. Keep tracking!")
                st.progress(unique_days_count / 14)
            else:
                st.write("This model uses Facebook Prophet to analyze your daily spending habits and predict future expenses.")
                if st.button("Generate Forecast"):
                    with st.spinner("Training AI Model... (This might take a moment)"):
                        forecast, error_msg = predict_next_30_days(st.session_state.expenses_df)
                    
                    if error_msg:
                        st.warning(error_msg)
                    else:
                        # Calculate Metrics
                        future_30_days = forecast[forecast['ds'] > pd.to_datetime("today")].head(30)
                        predicted_total = future_30_days['yhat'].sum()
                        
                        # Calculate Historical Average (Last 30 Days)
                        today = pd.to_datetime("today")
                        last_30_days_start = today - timedelta(days=30)
                        historical_df = st.session_state.expenses_df
                        last_30_total = historical_df[historical_df['Date'] >= last_30_days_start]['Amount'].sum()

                        # Display Comparison Metrics
                        m1, m2 = st.columns(2)
                        m1.metric("Actual Spending (Last 30 Days)", f"₹{last_30_total:,.2f}")
                        m2.metric("Predicted Spending (Next 30 Days)", f"₹{predicted_total:,.2f}", 
                                  delta=f"{predicted_total - last_30_total:,.2f}", delta_color="inverse")

                        # Create the Plotly chart for forecast
                        fig_forecast = go.Figure()

                        # Plot historical data (Black dots)
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['ds'], 
                            y=forecast['yhat'],
                            mode='lines',
                            name='Trend / Prediction',
                            line=dict(color='blue')
                        ))
                        
                        # Fill area for uncertainty (Confidence Interval)
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['ds'], 
                            y=forecast['yhat_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['ds'], 
                            y=forecast['yhat_lower'],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0, 0, 255, 0.2)',
                            line=dict(width=0),
                            name='Confidence Interval'
                        ))

                        # Update layout
                        fig_forecast.update_layout(
                            title="Projected Daily Spending",
                            xaxis_title="Date",
                            yaxis_title="Amount (₹)",
                            hovermode="x"
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        st.success("Note: The AI prediction is smoothed to avoid overreacting to large one-time expenses.")

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
