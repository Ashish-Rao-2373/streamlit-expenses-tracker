import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials

# --- Configuration ---
st.set_page_config(
    page_title="Dynamic Expense Tracker",
    page_icon="💰",
    layout="wide"
)

# --- Google Sheets Connection ---
# Use st.secrets for authentication
try:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    client = gspread.authorize(creds)

    # Open the Google Sheet by its name
    SPREADSHEET_NAME = st.secrets["gcp_service_account"]["sheet_name"]
    spreadsheet = client.open(SPREADSHEET_NAME)
    
    # Try to open the worksheet, create it if it doesn't exist
    try:
        worksheet = spreadsheet.worksheet("Expenses")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title="Expenses", rows="1", cols=4)
        worksheet.append_row(["Date", "Category", "Amount", "Comments"])

    # A flag to know if connection is successful
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
        # Ensure columns exist even if sheet is empty
        for col in ["Date", "Category", "Amount", "Comments"]:
            if col not in df.columns:
                df[col] = None
        
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            # Ensure Amount is numeric, coercing errors to NaN and then filling with 0
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
        # Add a unique identifier for each row for deletion
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
    
    # Drop the temporary id column before saving
    df_to_save = df.drop(columns=['id'], errors='ignore')

    # Convert datetime to string for CSV compatibility
    if not df_to_save.empty:
        df_to_save['Date'] = pd.to_datetime(df_to_save['Date']).dt.strftime('%Y-%m-%d')
    
    # Clear the sheet and write the new data
    worksheet.clear()
    set_with_dataframe(worksheet, df_to_save, include_index=False, resize=True)


# --- Main App ---
st.title("💰 Dynamic Expense Tracker")
st.write("This app saves your expenses to a centralized Google Sheet, accessible from any device.")

if GDRIVE_CONNECTED:
    # Load the data at the start
    if 'expenses_df' not in st.session_state:
        st.session_state.expenses_df = load_data()

    # --- Define Tabs ---
    tab1, tab2, tab3 = st.tabs(["📝 Add Expense", "📊 Monthly Dashboard", "📈 Analysis"])

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
                        "Manaswi","Recharge","Subscriptions", "Swimming", "Football", "🍔 Food & Dining", "🛒 Groceries", 
                        "🚗 Transportation", "💡 Utilities", "🏠 Housing", "🛍️ Shopping", 
                        "🎬 Entertainment", "💪 Health & Fitness", "💄 Personal Care", 
                        "🎓 Education", "🎁 Gifts & Donations", "✈️ Travel", "👶 Kids", 
                        "🐾 Pets", "💼 Business", "Miscellaneous"
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
                
                # Use session state to manage the dataframe
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
            # Use unique years from the dataframe for selection
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
            st.info("No expenses recorded yet. Add an expense in the first tab.")

    # --- Tab 3: Analysis ---
    with tab3:
        st.header("Expense Analysis")
        if not st.session_state.expenses_df.empty:
            analysis_df = st.session_state.expenses_df.copy()
            analysis_df['Month'] = analysis_df['Date'].dt.to_period('M').astype(str)
            monthly_summary = analysis_df.groupby('Month')['Amount'].sum().reset_index()
            st.subheader("Total Expenses Per Month")
            fig_monthly = px.bar(monthly_summary, x='Month', y='Amount', title='Monthly Expense Trend', labels={'Amount': 'Total Amount (₹)'})
            st.plotly_chart(fig_monthly, use_container_width=True)

            analysis_df['Quarter'] = analysis_df['Date'].dt.to_period('Q').astype(str)
            quarterly_summary = analysis_df.groupby('Quarter')['Amount'].sum().reset_index()
            st.subheader("Total Expenses Per Quarter")
            fig_quarterly = px.bar(quarterly_summary, x='Quarter', y='Amount', title='Quarterly Expense Trend', labels={'Amount': 'Total Amount (₹)'})
            st.plotly_chart(fig_quarterly, use_container_width=True)
        else:
            st.info("No expenses recorded yet. Add an expense in the first tab.")

    # --- Delete Expenses Section ---
    st.header("Manage Expenses")
    if not st.session_state.expenses_df.empty:
        with st.form("delete_form"):
            # Create a copy to display with a 'Select' column
            display_df = st.session_state.expenses_df.copy()
            display_df['Select'] = False
            
            # Reorder columns to put 'Select' first
            cols = ['Select'] + [col for col in display_df.columns if col != 'Select' and col != 'id']
            display_df = display_df[cols]
            
            st.write("Select expenses to delete:")
            edited_df = st.data_editor(
                display_df,
                hide_index=True,
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=[col for col in display_df.columns if col != 'Select']
            )

            delete_button = st.form_submit_button("Delete Selected Expenses", type="primary")

            if delete_button:
                selected_rows = edited_df[edited_df.Select]
                if not selected_rows.empty:
                    # Get the original indices to delete from the session state dataframe
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
    else:
        st.info("No expenses to manage.")
