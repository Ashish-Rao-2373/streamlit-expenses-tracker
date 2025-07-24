import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# --- Configuration ---
st.set_page_config(
    page_title="Simple Expense Tracker",
    page_icon="💰",
    layout="wide"
)

# --- Data Storage Setup ---
DATA_FILE = "expenses.csv"

def load_data():
    """Load expense data from a CSV file. If the file doesn't exist, create it."""
    if not os.path.exists(DATA_FILE):
        # Create an empty DataFrame with the correct columns and save it
        df = pd.DataFrame(columns=["Date", "Category", "Amount", "Comments"])
        df.to_csv(DATA_FILE, index=False)
    
    # Read the data, ensuring 'Date' is parsed correctly
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def save_data(df):
    """Save the DataFrame back to the CSV file."""
    df.to_csv(DATA_FILE, index=False)

# --- Main App ---

# Load the data at the start
expenses_df = load_data()

st.title("💰 Simple Expense Tracker")
st.write("A straightforward app to track your daily expenses. All data is saved to `expenses.csv`.")

# --- Define Tabs ---
tab1, tab2, tab3 = st.tabs(["📝 Add Expense", "📊 Monthly Dashboard", "📈 Analysis"])

# --- Tab 1: Add Expense ---
with tab1:
    st.header("Add a New Expense")
    
    # Create a form for user input
    with st.form("expense_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            expense_date = st.date_input("Date of Expense", datetime.now())
            
        with col2:
            expense_category = st.selectbox(
                "Expense Category",
                [
                    "Manaswi","🍔 Food & Dining", "🛒 Groceries", "🚗 Transportation", "💡 Utilities", 
                    "🏠 Housing", "🛍️ Shopping", "🎬 Entertainment", "💪 Health & Fitness",
                    "💄 Personal Care", "🎓 Education", "🎁 Gifts & Donations", "✈️ Travel",
                    "👶 Swimming", "🐾 Pets", "Football", " miscellaneous"
                ]
            )
            
        expense_amount = st.number_input("Amount", min_value=0.01, format="%.2f")
        expense_comments = st.text_area("Comments (Optional)")
        
        # Submit button for the form
        submitted = st.form_submit_button("Add Expense")
        
        if submitted:
            # Create a new DataFrame for the new entry
            new_expense = pd.DataFrame([{
                "Date": pd.to_datetime(expense_date),
                "Category": expense_category,
                "Amount": expense_amount,
                "Comments": expense_comments
            }])
            
            # Append the new expense and save
            expenses_df = pd.concat([expenses_df, new_expense], ignore_index=True)
            save_data(expenses_df)
            st.success("Expense added successfully!")

# --- Tab 2: Monthly Dashboard ---
with tab2:
    st.header("Monthly Dashboard")

    # Month and Year selection
    current_year = datetime.now().year
    selected_year = st.selectbox("Select Year", list(range(current_year, current_year - 5, -1)))
    selected_month = st.selectbox("Select Month", [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ])
    month_num = datetime.strptime(selected_month, "%B").month

    # Filter data for the selected month and year
    monthly_df = expenses_df[
        (expenses_df['Date'].dt.year == selected_year) &
        (expenses_df['Date'].dt.month == month_num)
    ]

    if monthly_df.empty:
        st.info(f"No expenses recorded for {selected_month} {selected_year}.")
    else:
        # Display total expenses for the month
        total_monthly_expense = monthly_df['Amount'].sum()
        st.metric(label=f"Total Expenses for {selected_month} {selected_year}", value=f"₹{total_monthly_expense:,.2f}")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart of expenses by category
            st.subheader("Expenses by Category")
            category_summary = monthly_df.groupby("Category")["Amount"].sum().reset_index()
            fig_pie = px.pie(
                category_summary,
                values='Amount',
                names='Category',
                title='Category Breakdown'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart of expenses by category
            st.subheader(" ") # For alignment
            fig_bar = px.bar(
                category_summary.sort_values('Amount', ascending=False),
                x='Category',
                y='Amount',
                title='Category Breakdown (Bar)'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# --- Tab 3: Analysis ---
with tab3:
    st.header("Expense Analysis")

    # Group data by month
    expenses_df['Month'] = expenses_df['Date'].dt.to_period('M').astype(str)
    monthly_summary = expenses_df.groupby('Month')['Amount'].sum().reset_index()

    st.subheader("Total Expenses Per Month")
    fig_monthly = px.bar(
        monthly_summary,
        x='Month',
        y='Amount',
        title='Monthly Expense Trend',
        labels={'Amount': 'Total Amount ($)'}
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Group data by quarter
    expenses_df['Quarter'] = expenses_df['Date'].dt.to_period('Q').astype(str)
    quarterly_summary = expenses_df.groupby('Quarter')['Amount'].sum().reset_index()

    st.subheader("Total Expenses Per Quarter")
    fig_quarterly = px.bar(
        quarterly_summary,
        x='Quarter',
        y='Amount',
        title='Quarterly Expense Trend',
        labels={'Amount': 'Total Amount ($)'}
    )
    st.plotly_chart(fig_quarterly, use_container_width=True)


# --- Display Raw Data ---
with st.expander("View All Expenses"):
    st.dataframe(expenses_df.sort_values(by="Date", ascending=False), use_container_width=True)

