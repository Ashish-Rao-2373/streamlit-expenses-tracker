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
import google.generativeai as genai

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

# --- Gemini AI Configuration ---
GEMINI_AVAILABLE = False
if "GEMINI_API_KEY" in st.secrets:
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEMINI_AVAILABLE = True
    except Exception as e:
        pass # Fail silently and fall back to other methods

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

# --- Data Science: Improved Categorization ---
def get_historical_mapping(df):
    """Creates a dictionary mapping exact past comments to their categories."""
    if df.empty:
        return {}
    valid_data = df.dropna(subset=['Comments', 'Category'])
    valid_data = valid_data[valid_data['Comments'].astype(str).str.strip() != '']
    comment_map = valid_data.groupby(valid_data['Comments'].str.lower())['Category'].last().to_dict()
    return comment_map

def train_nlp_model(df):
    """Trains a backup NLP model for unseen comments."""
    training_data = df.dropna(subset=['Comments', 'Category'])
    training_data = training_data[training_data['Comments'].astype(str).str.strip() != '']
    if len(training_data) < 5: return None 
    model = make_pipeline(TfidfVectorizer(stop_words='english', ngram_range=(1,2)), MultinomialNB(alpha=0.1))
    model.fit(training_data['Comments'].astype(str), training_data['Category'])
    return model

def predict_category_smart(comment, history_map, nlp_model, all_categories):
    """
    Predicts category using a 4-Step Waterfall approach:
    1. EXACT HISTORY: Has user typed this before? (e.g. "Manaswi" -> "Girlfriend")
    2. LLM (Gemini): Use AI reasoning for new concepts ("face wash" -> Personal Care).
    3. KEYWORD RULES: Common synonyms.
    4. NLP MODEL: Statistical guess.
    """
    if not comment or comment.strip() == "":
        return None, None
        
    comment_lower = comment.lower().strip()
    
    # STEP 1: History Lookup (Personalization - FASTEST)
    if comment_lower in history_map:
        return history_map[comment_lower], "History Match"

    # STEP 2: Gemini LLM (Reasoning - SMARTEST)
    if GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            # The prompt string starts here. It uses triple quotes.
            # Make sure this block is closed properly with """
            prompt = f"""
            You are an expert expense categorizer. Map the expense comment to the most logical category from the list below.
            
            Categories: {", ".join(all_categories)}
            
            Expense Comment: "{comment}"
            
            Rules:
            - Use common sense (e.g., 'face wash' is Personal Care, 'Uber' is Transportation).
            - 'Manaswi' is a person, likely 'Girlfriend' or 'Gifts' if implies spending on her.
            - Return ONLY the exact category name from the list. Nothing else.
            """ 
            # End of prompt string. Make sure lines below this return to normal color.
            
            response = model.generate_content(prompt)
            predicted_cat = response.text.strip()
            
            # Verify the LLM returned a valid category
            for cat in all_categories:
                if cat in predicted_cat:
                    return cat, "Gemini AI"
        except Exception:
            pass # Fallback if API fails

    # STEP 3: Enhanced Rule-Based Matching (Common Logic - FALLBACK)
    keywords = {
        "❤️ Girlfriend": ["manaswi", "gift", "date", "gf", "love", "flower", "chocolate"],
        "☕ Chai & snacks": ["chai", "tea", "coffee", "snack", "biscuit", "samosa", "puff", "break"],
        "⚽ Sports": ["swimming", "football", "cricket", "badminton", "turf", "gym", "court", "match", "cap", "goggles", "jersey"],
        "💄 Personal Care": ["face", "wash", "soap", "shampoo", "hair", "salon", "medicine", "doctor", "pharmacy", "trim", "shave", "cream", "lotion", "dental"],
        "⛽ Fuel & Bike Service": ["petrol", "diesel", "gas", "fuel", "bike", "service", "mechanic", "oil", "air", "puncture", "honda", "yamaha"],
        "🍔 Food & Dining": ["swiggy", "zomato", "restaurant", "cafe", "dinner", "lunch", "breakfast", "burger", "pizza", "biryani", "mandi", "tiffin", "idli", "dosa"],
        "🛒 Groceries": ["milk", "eggs", "curd", "vegetables", "fruits", "grocery", "supermarket", "dmart", "blinkit", "bigbasket", "oil", "rice", "dal"],
        "🚗 Transportation": ["uber", "ola", "rapido", "bus", "metro", "train", "flight", "cab", "auto", "taxi", "ticket"],
        "📱 Recharge & Subscriptions": ["recharge", "jio", "airtel", "vi", "wifi", "broadband", "netflix", "prime", "spotify", "subscription", "plan"],
        "🏠 Housing": ["rent", "maintenance", "electricity", "water", "maid", "internet"],
        "🛍️ Shopping": ["amazon", "flipkart", "myntra", "clothes", "shoe", "shirt", "pant", "jeans"]
    }
    
    for category, tags in keywords.items():
        for tag in tags:
            if f" {tag} " in f" {comment_lower} " or comment_lower == tag:
                return category, "Keyword Match"

    # STEP 4: Simple NLP Prediction (Last Resort)
    if nlp_model:
        try:
            probs = nlp_model.predict_proba([comment])[0]
            max_prob = np.max(probs)
            if max_prob > 0.3:
                prediction = nlp_model.classes_[np.argmax(probs)]
                return prediction, "Statistical Guess"
        except:
            pass
            
    return None, None

# --- Data Science: Forecasting Function ---
def predict_month_end(df):
    """Uses 'Average Burn Rate' + 'Standard Deviation' logic."""
    fixed_cost_threshold = 5000
    variable_expenses_df = df[df['Amount'] < fixed_cost_threshold].copy()
    daily_expenses = variable_expenses_df.groupby("Date")['Amount'].sum().reset_index()
    daily_expenses.columns = ['ds', 'y']
    
    if len(daily_expenses) < 14: return None, "Not enough data points (days) to forecast.", None, None

    recent_cutoff = pd.to_datetime("today") - timedelta(days=90)
    recent_data = daily_expenses[daily_expenses['ds'] >= recent_cutoff]
    if recent_data.empty: recent_data = daily_expenses 

    total_spend_recent = recent_data['y'].sum()
    first_date = recent_data['ds'].min()
    last_date = recent_data['ds'].max()
    days_span = (last_date - first_date).days + 1 if not pd.isna(first_date) else 1
    daily_burn_rate = total_spend_recent / max(1, days_span)
    
    all_dates = pd.date_range(start=first_date, end=last_date)
    recent_data_indexed = recent_data.set_index('ds').reindex(all_dates, fill_value=0)
    daily_volatility = recent_data_indexed['y'].std()

    today = datetime.now()
    last_day_of_month = calendar.monthrange(today.year, today.month)[1]
    days_remaining = last_day_of_month - today.day
    
    if days_remaining <= 0: return None, "It's the end of the month!", None, None

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
        st.session_state.history_map = get_historical_mapping(st.session_state.expenses_df)
        st.session_state.nlp_model = train_nlp_model(st.session_state.expenses_df)

    tab1, tab2, tab3 = st.tabs(["📝 Add Expense", "📊 Monthly Dashboard", "📈 Analysis & AI Forecast"])

    with tab1:
        st.header("Add a New Expense")
        col1, col2 = st.columns(2)
        with col1:
            expense_date = st.date_input("Date of Expense", datetime.now())
            expense_comments = st.text_input("Comments (Type this first for Magic!)", placeholder="e.g., Manaswi gift, swimming cap, petrol...")

        with col2:
            suggested_index = 0
            all_categories = [
                "❤️ Girlfriend", "⛽ Fuel & Bike Service", "📱 Recharge & Subscriptions", "☕ Chai & snacks", "⚽ Sports",
                "🍔 Food & Dining", "🛒 Groceries", "🚗 Transportation", "💡 Utilities", 
                "🏠 Housing", "🛍️ Shopping", "🎬 Entertainment", "💪 Health & Fitness", 
                "💄 Personal Care", "🎓 Education", "🎁 Gifts & Donations", "✈️ Travel", 
                "Miscellaneous"
            ]
            prediction_source = None
            
            if expense_comments:
                hist_map = st.session_state.get('history_map', {})
                model = st.session_state.get('nlp_model')
                # Pass all_categories to the new function
                predicted_cat, source = predict_category_smart(expense_comments, hist_map, model, all_categories)
                
                if predicted_cat and predicted_cat in all_categories:
                    suggested_index = all_categories.index(predicted_cat)
                    prediction_source = source

            expense_category = st.selectbox("Expense Category", all_categories, index=suggested_index)
            if prediction_source:
                st.caption(f"✨ Auto-selected by: **{prediction_source}**")

        expense_amount = st.number_input("Amount", min_value=0.01, format="%.2f")
        
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
            
            # Retrain/Update memory
            st.session_state.history_map = get_historical_mapping(st.session_state.expenses_df)
            st.session_state.nlp_model = train_nlp_model(st.session_state.expenses_df)
            
            st.success("Expense added successfully and saved to Google Sheets!")

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

    with tab3:
        st.header("Expense Analysis & AI Forecast")
        if not st.session_state.expenses_df.empty:
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
            current_month_name = datetime.now().strftime("%B")
            st.subheader(f"🔮 End-of-Month Projection ({current_month_name})")
            
            unique_days_count = st.session_state.expenses_df['Date'].nunique()
            if unique_days_count < 14:
                st.info(f"⏳ **AI Forecast is unlocking... (Progress: {unique_days_count}/14 days)**")
                st.progress(unique_days_count / 14)
            else:
                st.write(f"Based on your spending so far in {current_month_name}, here is where you are likely to end up.")
                if st.button("Generate Projection"):
                    with st.spinner("Analyzing current month trend..."):
                        forecast, error_msg, debug_msg, daily_std = predict_month_end(st.session_state.expenses_df)
                    
                    if error_msg:
                        st.warning(error_msg)
                    else:
                        today = datetime.now()
                        current_month_start = today.replace(day=1)
                        current_month_data = st.session_state.expenses_df[
                            (st.session_state.expenses_df['Date'] >= current_month_start) &
                            (st.session_state.expenses_df['Date'] <= today)
                        ]
                        spent_so_far = current_month_data['Amount'].sum()
                        
                        has_paid_rent_this_month = any(current_month_data['Amount'] >= 5000)
                        estimated_future_fixed_costs = 0
                        if not has_paid_rent_this_month:
                             historical_rents = st.session_state.expenses_df[st.session_state.expenses_df['Amount'] >= 5000]
                             if not historical_rents.empty:
                                 estimated_future_fixed_costs = 7500
                                 st.info("ℹ️ We noticed you haven't paid Rent yet this month, so we added ₹7,500 to the projection.")

                        future_forecast = forecast[forecast['ds'] > pd.to_datetime(today)]
                        days_remaining = len(future_forecast)
                        predicted_variable_mid = future_forecast['yhat'].sum()
                        total_variable_uncertainty = daily_std * np.sqrt(days_remaining)
                        
                        total_mid = spent_so_far + predicted_variable_mid + estimated_future_fixed_costs
                        total_low = total_mid - total_variable_uncertainty
                        total_high = total_mid + total_variable_uncertainty

                        st.metric(f"Already Spent in {current_month_name}", f"₹{spent_so_far:,.2f}")
                        st.success(f"🎯 **Projected Total for {current_month_name}: ₹{total_low:,.0f} — ₹{total_high:,.0f}**")
                        st.caption(f"Most likely outcome: ~₹{total_mid:,.0f}")
                        if debug_msg: st.caption(f"ℹ️ {debug_msg}")

                        fig_forecast = go.Figure()
                        current_month_data = current_month_data.sort_values('Date')
                        current_month_data['Cumulative_Amount'] = current_month_data['Amount'].cumsum()
                        fig_forecast.add_trace(go.Scatter(x=current_month_data['Date'], y=current_month_data['Cumulative_Amount'], mode='lines+markers', name='Actual Spend', line=dict(color='green', width=3)))
                        
                        start_amount = spent_so_far
                        future_forecast = forecast.copy()
                        future_forecast['Cumulative_yhat'] = start_amount + future_forecast['yhat'].cumsum()
                        days_out = np.arange(1, len(future_forecast) + 1)
                        spread = daily_std * np.sqrt(days_out)
                        future_forecast['Cumulative_yhat_lower'] = future_forecast['Cumulative_yhat'] - spread
                        future_forecast['Cumulative_yhat_upper'] = future_forecast['Cumulative_yhat'] + spread

                        last_actual_date = current_month_data['Date'].max() if not current_month_data.empty else today
                        connector_row = pd.DataFrame({'ds': [last_actual_date], 'Cumulative_yhat': [start_amount], 'Cumulative_yhat_lower': [start_amount], 'Cumulative_yhat_upper': [start_amount]})
                        plot_forecast = pd.concat([connector_row, future_forecast[['ds', 'Cumulative_yhat', 'Cumulative_yhat_lower', 'Cumulative_yhat_upper']]]).sort_values('ds')

                        fig_forecast.add_trace(go.Scatter(x=plot_forecast['ds'], y=plot_forecast['Cumulative_yhat'], mode='lines', name='Projected Trajectory', line=dict(color='blue', dash='dot')))
                        fig_forecast.add_trace(go.Scatter(x=plot_forecast['ds'], y=plot_forecast['Cumulative_yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                        fig_forecast.add_trace(go.Scatter(x=plot_forecast['ds'], y=plot_forecast['Cumulative_yhat_lower'], mode='lines', fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', line=dict(width=0), name='Likely Range'))
                        fig_forecast.update_layout(title=f"{current_month_name} Cumulative Spending Trajectory", xaxis_title="Date", yaxis_title="Total Spent (₹)", hovermode="x")
                        st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.info("No expenses recorded yet.")

    st.header("Manage Expenses")
    if not st.session_state.expenses_df.empty:
        with st.form("delete_form"):
            display_df = st.session_state.expenses_df.copy()
            display_df['Select'] = False
            cols = ['Select'] + [col for col in display_df.columns if col != 'Select' and col != 'id']
            display_df = display_df[cols]
            st.write("Select expenses to delete:")
            edited_df = st.data_editor(display_df, hide_index=True, column_config={"Select": st.column_config.CheckboxColumn(required=True)}, disabled=[col for col in display_df.columns if col != 'Select'])
            col1, col2 = st.columns(2)
            with col1: delete_button = st.form_submit_button("Delete Selected Expenses")
            with col2: delete_all_button = st.form_submit_button("⚠️ Delete All Expenses", type="primary")

            if delete_button:
                selected_rows = edited_df[edited_df.Select]
                if not selected_rows.empty:
                    indices_to_delete = st.session_state.expenses_df[st.session_state.expenses_df['Date'].isin(pd.to_datetime(selected_rows['Date'])) & st.session_state.expenses_df['Amount'].isin(selected_rows['Amount']) & st.session_state.expenses_df['Category'].isin(selected_rows['Category'])].index
                    st.session_state.expenses_df = st.session_state.expenses_df.drop(indices_to_delete)
                    save_data(st.session_state.expenses_df)
                    st.success(f"Successfully deleted {len(indices_to_delete)} expense(s).")
                    st.rerun()
                else: st.warning("No expenses selected for deletion.")
            if delete_all_button:
                empty_df = pd.DataFrame(columns=["Date", "Category", "Amount", "Comments"])
                st.session_state.expenses_df = empty_df
                save_data(st.session_state.expenses_df)
                st.success("All expenses have been deleted.")
                st.rerun()
    else:
        st.info("No expenses to manage.")
```
```eof
