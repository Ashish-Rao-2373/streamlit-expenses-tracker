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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

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

# --- Data Science: NLP Category Prediction ---
def train_categorization_model(df):
    """
    Trains a TfidfVectorizer + Naive Bayes model to predict Category based on Comments.
    Uses TF-IDF to better handle unique keywords like 'petrol' vs generic words.
    """
    # We need valid comments and categories to train
    training_data = df.dropna(subset=['Comments', 'Category'])
    # Filter out empty comments
    training_data = training_data[training_data['Comments'].astype(str).str.strip() != '']
    
    # Needs a bit more data to be useful
    if len(training_data) < 5:
        return None 

    # Create a pipeline: TF-IDF Vectorizer -> Naive Bayes Classifier
    # TF-IDF reduces the impact of frequent words that appear everywhere (like 'the', 'a')
    # and boosts unique words that appear in specific categories.
    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    model.fit(training_data['Comments'].astype(str), training_data['Category'])
    return model

def predict_category(model, comment):
    """
    Predicts category from comment using a HYBRID approach:
    1. Keyword Matching (Hard rules for common synonyms)
    2. ML Model Prediction (Fallback if no keyword match)
    """
    if not comment or comment.strip() == "":
        return None
        
    comment_lower = comment.lower()
    
    # 1. HARD-CODED RULES (The "Cheat Sheet")
    # This overrides the AI for obvious matches
    keywords = {
        "⛽ Fuel & Bike Service": ["petrol", "diesel", "gas", "fuel", "bike", "service", "mechanic", "oil change", "air", "puncture"],
        "🍔 Food & Dining": ["swiggy", "zomato", "restaurant", "cafe", "coffee", "tea", "chai", "dinner", "lunch", "breakfast", "burger", "pizza", "biryani", "mandi"],
        "🛒 Groceries": ["milk", "eggs", "vegetables", "fruits", "grocery", "supermarket", "dmart", "blinkit", "bigbasket"],
        "🚗 Transportation": ["uber", "ola", "rapido", "bus", "metro", "train", "flight", "cab", "auto", "taxi", "ticket"],
        "📱 Recharge & Subscriptions": ["recharge", "jio", "airtel", "vi", "wifi", "broadband", "netflix", "prime", "spotify", "subscription"],
        "⚽ Sports": ["turf", "badminton", "cricket", "football", "gym", "court", "match"],
        "❤️ Girlfriend": ["gift", "date", "gf", "love"],
        "💄 Personal Care": ["haircut", "salon", "medicine", "doctor", "pharmacy", "trim", "shave"]
    }
    
    for category, tags in keywords.items():
        if any(tag in comment_lower for tag in tags):
            return category

    # 2. AI MODEL PREDICTION
    # If no keyword matched, ask the trained model
    if model:
        try:
            prediction = model.predict([comment])[0]
            return prediction
        except:
            return None
            
    return None

# --- Data Science: Forecasting Function ---
def predict_month_end(df):
    """
    Uses 'Average Burn Rate' + 'Standard Deviation' to handle volatile spending patterns.
    This effectively smooths out the 0 vs 1000 days into a steady prediction.
    """
    # 1. Prepare data & Filter Fixed Costs
    fixed_cost_threshold = 5000
    variable_expenses_df = df[df['Amount'] < fixed_cost_threshold].copy()
    
    daily_expenses = variable_expenses_df.groupby("Date")['Amount'].sum().reset_index()
    daily_expenses.columns = ['ds', 'y']
    
    if len(daily_expenses) < 14:
        return None, "Not enough data points (days) to forecast.", None, None

    # 2. Calculate "Burn Rate" and "Volatility"
    recent_cutoff = pd.to_datetime("today") - timedelta(days=90)
    recent_data = daily_expenses[daily_expenses['ds'] >= recent_cutoff]
    
    if recent_data.empty:
        recent_data = daily_expenses 

    total_spend_recent = recent_data['y'].sum()
    first_date = recent_data['ds'].min()
    last_date = recent_data['ds'].max()
    
    if pd.isna(first_date): 
        days_span = 1
    else:
        days_span = (last_date - first_date).days + 1
        
    daily_burn_rate = total_spend_recent / max(1, days_span)
    
    all_dates = pd.date_range(start=first_date, end=last_date)
    recent_data_indexed = recent_data.set_index('ds').reindex(all_dates, fill_value=0)
    daily_volatility = recent_data_indexed['y'].std()

    # 3. Calculate Days Remaining
    today = datetime.now()
    last_day_of_month = calendar.monthrange(today.year, today.month)[1]
    days_remaining = last_day_of_month - today.day
    
    if days_remaining <= 0:
        return None, "It's the end of the month! No days left to forecast.", None, None

    # 4. Generate Forecast
    future_dates = [today + timedelta(days=x) for x in range(1, days_remaining + 1)]
    forecast_df = pd.DataFrame({'ds': future_dates})
    
    forecast_df['yhat'] = daily_burn_rate
    forecast_df['yhat_lower'] = daily_burn_rate 
    forecast_df['yhat_upper'] = daily_burn_rate 
    
    return forecast_df, None, f"Based on your recent avg spend of ₹{daily_burn_rate:,.0f}/day (±₹{daily_volatility:,.0f})", daily_volatility


# --- Main App ---
st.title("💰 Dynamic Expense Tracker")
st.write("This app saves your expenses to a centralized Google Sheet, accessible from any device.")

if GDRIVE_CONNECTED:
    if 'expenses_df' not in st.session_state:
        st.session_state.expenses_df = load_data()
        # Train NLP model on startup
        st.session_state.nlp_model = train_categorization_model(st.session_state.expenses_df)

    # --- Define Tabs ---
    tab1, tab2, tab3 = st.tabs(["📝 Add Expense", "📊 Monthly Dashboard", "📈 Analysis & AI Forecast"])

    # --- Tab 1: Add Expense ---
    with tab1:
        st.header("Add a New Expense")
        
        # We use a container to organize the inputs, but NOT st.form
        # This allows the "Comments" field to trigger a reload for auto-categorization
        col1, col2 = st.columns(2)
        
        with col1:
            expense_date = st.date_input("Date of Expense", datetime.now())
            
            # 1. Comments Input (Key for NLP)
            # When user types here and hits enter, the app reloads and runs the prediction logic below
            expense_comments = st.text_input("Comments (Type this first for Magic!)", 
                                           placeholder="e.g., Uber to office, Lunch at cafe...")

        with col2:
            # 2. NLP Prediction Logic
            # Check if model exists and comment is present
            suggested_index = 0
            all_categories = [
                "❤️ Girlfriend", "⛽ Fuel & Bike Service", "📱 Recharge & Subscriptions", "☕ Chai & snacks", "⚽ Sports",
                "🍔 Food & Dining", "🛒 Groceries", "🚗 Transportation", "💡 Utilities", 
                "🏠 Housing", "🛍️ Shopping", "🎬 Entertainment", "💪 Health & Fitness", 
                "💄 Personal Care", "🎓 Education", "🎁 Gifts & Donations", "✈️ Travel", 
                "Miscellaneous"
            ]
            
            if expense_comments:
                # Use the session state model if available
                model_to_use = st.session_state.get('nlp_model')
                predicted_cat = predict_category(model_to_use, expense_comments)
                
                if predicted_cat and predicted_cat in all_categories:
                    suggested_index = all_categories.index(predicted_cat)
                    st.toast(f"🤖 AI detected category: **{predicted_cat}**", icon="✨")

            # Category Selectbox (defaults to AI prediction if available)
            expense_category = st.selectbox("Expense Category", all_categories, index=suggested_index)

        # Amount Input
        expense_amount = st.number_input("Amount", min_value=0.01, format="%.2f")
        
        # Submit Button (Manual because we removed st.form)
        if st.button("Add Expense", type="primary"):
            new_expense = pd.DataFrame([{
                "Date": pd.to_datetime(expense_date),
                "Category": expense_category,
                "Amount": expense_amount,
                "Comments": expense_comments
            }])
            updated_df = pd.concat([st.session_state.expenses_df, new_expense], ignore_index=True)
            st.session_state.expenses_df = updated_df
            save_data(st.session_state.expenses_df)
            
            # Retrain model with new data for next time
            st.session_state.nlp_model = train_categorization_model(st.session_state.expenses_df)
            
            st.success("Expense added successfully and saved to Google Sheets!")
            # Note: We can't clear inputs easily without st.form, but Streamlit reruns usually handle flow well.
            # Ideally we might use session state keys to clear, but for simplicity we let it persist or user clears.

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
                        forecast, error_msg, debug_msg, daily_std = predict_month_end(st.session_state.expenses_df)
                    
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
                        
                        # 2. Check for Future Fixed Costs (Rent)
                        has_paid_rent_this_month = any(current_month_data['Amount'] >= 5000)
                        estimated_future_fixed_costs = 0
                        if not has_paid_rent_this_month:
                             historical_rents = st.session_state.expenses_df[st.session_state.expenses_df['Amount'] >= 5000]
                             if not historical_rents.empty:
                                 estimated_future_fixed_costs = 7500
                                 st.info("ℹ️ We noticed you haven't paid Rent yet this month, so we added ₹7,500 to the projection.")

                        # 3. Predicted Remaining Variable Spend
                        future_forecast = forecast[forecast['ds'] > pd.to_datetime(today)]
                        days_remaining = len(future_forecast)
                        
                        predicted_variable_mid = future_forecast['yhat'].sum()
                        
                        # SQUARE ROOT RULE for Total Uncertainty
                        total_variable_uncertainty = daily_std * np.sqrt(days_remaining)
                        
                        # 4. Total Projected Range
                        total_mid = spent_so_far + predicted_variable_mid + estimated_future_fixed_costs
                        total_low = total_mid - total_variable_uncertainty
                        total_high = total_mid + total_variable_uncertainty

                        # --- Display Results ---
                        st.metric(f"Already Spent in {current_month_name}", f"₹{spent_so_far:,.2f}")
                        
                        st.success(f"🎯 **Projected Total for {current_month_name}: ₹{total_low:,.0f} — ₹{total_high:,.0f}**")
                        st.caption(f"Most likely outcome: ~₹{total_mid:,.0f}")
                        
                        if debug_msg:
                            st.caption(f"ℹ️ {debug_msg}")

                        # --- Visualization (CUMULATIVE) ---
                        fig_forecast = go.Figure()

                        # 1. Plot Cumulative Actuals
                        current_month_data = current_month_data.sort_values('Date')
                        current_month_data['Cumulative_Amount'] = current_month_data['Amount'].cumsum()
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=current_month_data['Date'], 
                            y=current_month_data['Cumulative_Amount'],
                            mode='lines+markers',
                            name='Actual Spend So Far',
                            line=dict(color='green', width=3)
                        ))
                        
                        # 2. Prepare Cumulative Forecast
                        start_amount = spent_so_far
                        
                        # Create visual forecast path
                        future_forecast = forecast.copy()
                        # Cumulative sum of daily predictions added to the current total
                        future_forecast['Cumulative_yhat'] = start_amount + future_forecast['yhat'].cumsum()
                        
                        # Calculate the uncertainty cone (expanding over time)
                        days_out = np.arange(1, len(future_forecast) + 1)
                        # Scale the volatility by sqrt(days) to create the correct cone shape
                        spread = daily_std * np.sqrt(days_out)
                        
                        future_forecast['Cumulative_yhat_lower'] = future_forecast['Cumulative_yhat'] - spread
                        future_forecast['Cumulative_yhat_upper'] = future_forecast['Cumulative_yhat'] + spread

                        # Connector
                        last_actual_date = current_month_data['Date'].max() if not current_month_data.empty else today
                        connector_row = pd.DataFrame({
                            'ds': [last_actual_date],
                            'Cumulative_yhat': [start_amount],
                            'Cumulative_yhat_lower': [start_amount],
                            'Cumulative_yhat_upper': [start_amount]
                        })
                        
                        plot_forecast = pd.concat([connector_row, future_forecast[['ds', 'Cumulative_yhat', 'Cumulative_yhat_lower', 'Cumulative_yhat_upper']]]).sort_values('ds')

                        # Plot Forecast Line
                        fig_forecast.add_trace(go.Scatter(
                            x=plot_forecast['ds'], 
                            y=plot_forecast['Cumulative_yhat'],
                            mode='lines',
                            name='Projected Trajectory',
                            line=dict(color='blue', dash='dot')
                        ))
                        
                        # Plot Confidence Interval
                        fig_forecast.add_trace(go.Scatter(
                            x=plot_forecast['ds'], 
                            y=plot_forecast['Cumulative_yhat_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        fig_forecast.add_trace(go.Scatter(
                            x=plot_forecast['ds'], 
                            y=plot_forecast['Cumulative_yhat_lower'],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(0, 0, 255, 0.1)',
                            line=dict(width=0),
                            name='Likely Range'
                        ))

                        fig_forecast.update_layout(
                            title=f"{current_month_name} Cumulative Spending Trajectory",
                            xaxis_title="Date",
                            yaxis_title="Total Spent (₹)",
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
