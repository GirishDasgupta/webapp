# perfect

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import bcrypt
import sqlite3

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Database operations
def get_db_connection():
    return sqlite3.connect('example.db')

def create_users_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT,
                        email TEXT);''')
    conn.commit()
    conn.close()

def register_user(username, password, email):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    if c.fetchone():
        conn.close()
        return False
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', 
              (username, hashed_password, email))
    conn.commit()
    conn.close()
    return True

def authenticate(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
        return True
    return False

def logout_user():
    st.session_state['authenticated'] = False

# Data loading functions
@st.cache_data(hash_funcs={sqlite3.Connection: id})
def load_current_datasets():
    transactions_data = pd.read_csv('400_transactions.csv').rename(columns=str.strip)
    households_data = pd.read_csv('400_households.csv').rename(columns=str.strip)
    products_data = pd.read_csv('400_products.csv').rename(columns=str.strip)
    return transactions_data, households_data, products_data

def demographic_analysis(households_data, transactions_data):
    merged_data = transactions_data.merge(households_data, on='HSHD_NUM')
    return merged_data.groupby('HH_SIZE').agg(avg_transaction_amount=('SPEND', 'mean'))

def perform_customer_segmentation(transactions_data):
    with open('customer_segmentation_model.pkl', 'rb') as file:
        kmeans = pickle.load(file)
    features = transactions_data.groupby('HSHD_NUM').agg(
        total_spend=('SPEND', 'sum'),
        avg_spend=('SPEND', 'mean'),
        transaction_count=('SPEND', 'count')
    ).reset_index()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features[['total_spend', 'avg_spend', 'transaction_count']])
    segments = kmeans.predict(features_scaled)
    return segments

# Streamlit UI
def login_ui():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state['authenticated'] = True
        else:
            st.error("Invalid username/password")

def register_ui():
    st.header("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    new_email = st.text_input("New Email")
    if st.button("Register"):
        if register_user(new_username, new_password, new_email):
            st.success("Registration successful. You can now login.")
        else:
            st.error("Registration failed. Username already exists.")

def main_ui():
    # Add your main dashboard UI components here
    st.title("Retail Data Analysis Dashboard")

    # Sidebar navigation
    pages = {
        "Sample Data Pull": sample_data_pull_ui,
        "Search Data Pulls": search_data_pulls_ui,
        "Customer Segmentation": customer_segmentation_ui,
        "Customer Engagement Dashboard": customer_engagement_dashboard_ui
    }

    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))
    if selected_page:
        pages[selected_page]()

    # Logout button in the sidebar
    if st.sidebar.button("Logout"):
        logout_user()

def sample_data_pull_ui():
    st.header("Sample Data Pull for Household #10")
    transactions_data, _, _ = load_current_datasets()
    sample_data = transactions_data.head()  # Display the first few rows for sample
    st.dataframe(sample_data)
    # Visualization
    fig = px.bar(sample_data, x='PRODUCT_NUM', y='SPEND', title='Sample Data Pull Visualization')
    st.plotly_chart(fig)

def search_data_pulls_ui():
    st.header("Search Data Pulls")
    
    transactions_data, _, _ = load_current_datasets()
    
    # User input for household number
    hshd_num = st.number_input("Enter Household Number", min_value=1, step=1, format='%d')
    
    # Button to perform search
    if st.button('Search'):
        # Filter transactions by the input household number
        search_result = transactions_data[transactions_data['HSHD_NUM'] == hshd_num]
        
        if not search_result.empty:
            st.write(f"Transactions for Household Number: {hshd_num}")
            st.dataframe(search_result)
            
            # Optional: Visualization
            fig = px.bar(search_result, x='PRODUCT_NUM', y='SPEND', color='SPEND', title=f'Transactions for Household #{hshd_num}')
            st.plotly_chart(fig)
        else:
            st.write(f"No transactions found for Household Number: {hshd_num}")

def customer_segmentation_ui():
    st.header("Customer Segmentation")
    transactions_data, _, _ = load_current_datasets()
    segments = perform_customer_segmentation(transactions_data)
    st.write(pd.Series(segments).value_counts())
    fig = px.histogram(x=segments, title='Customer Segmentation')
    st.plotly_chart(fig)

def customer_engagement_dashboard_ui():
    st.header("Customer Engagement Dashboard")
    transactions_data, households_data, _ = load_current_datasets()
    demographic_factors = demographic_analysis(households_data, transactions_data)
    st.write(demographic_factors)
    fig = px.bar(demographic_factors, x=demographic_factors.index, y='avg_transaction_amount', title='Average Transaction Amount by Household Size')
    st.plotly_chart(fig)

# Start the app
def main():
    create_users_table()
    if st.session_state['authenticated']:
        main_ui()
    else:
        page = st.sidebar.selectbox("Select a page", ["Login", "Register"])
        if page == "Login":
            login_ui()
        elif page == "Register":
            register_ui()

if __name__ == "__main__":
    main()
