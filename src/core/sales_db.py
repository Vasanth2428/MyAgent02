import sqlite3
import os
import logging
import re
from datetime import datetime, timedelta
import random

logger = logging.getLogger("RAG.SalesDB")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sales.db")

def _initialize_db():
    """Create tables and populate with synthetic data if they don't exist."""
    is_new = not os.path.exists(DB_PATH)
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Create Tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            country TEXT NOT NULL,
            signup_date DATE NOT NULL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            unit_price REAL NOT NULL,
            stock_quantity INTEGER NOT NULL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            order_date DATE NOT NULL,
            total_amount REAL NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
        )
        """)
        
        if is_new:
            logger.info("Initializing synthetic enterprise sales database...")
            # Populate Customers
            companies = ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Stark Ind", "Wayne Ent", "Cyberdyne", "Massive Dynamic"]
            countries = ["USA", "UK", "Canada", "Germany", "Japan"]
            
            for comp in companies:
                signup = (datetime.now() - timedelta(days=random.randint(100, 1000))).strftime("%Y-%m-%d")
                cursor.execute(
                    "INSERT INTO customers (name, email, country, signup_date) VALUES (?, ?, ?, ?)",
                    (comp, f"contact@{comp.lower().replace(' ', '')}.com", random.choice(countries), signup)
                )
            
            # Populate Inventory
            products = [
                ("Enterprise Server Rack", "Hardware", 4500.00, 12),
                ("Cloud Storage 1TB (Annual)", "Software", 120.00, 9999),
                ("Load Balancer Pro", "Network", 850.00, 45),
                ("Cyber-security Firewall v4", "Software", 2999.99, 150),
                ("Gigabit Switch 48-port", "Hardware", 450.00, 32),
                ("AI Accelerator GPU", "Hardware", 12000.00, 5)
            ]
            for p in products:
                cursor.execute(
                    "INSERT INTO inventory (product_name, category, unit_price, stock_quantity) VALUES (?, ?, ?, ?)",
                    p
                )
                
            # Populate Orders
            statuses = ["Completed", "Processing", "Shipped", "Cancelled"]
            for _ in range(50):
                cid = random.randint(1, len(companies))
                odate = (datetime.now() - timedelta(days=random.randint(1, 100))).strftime("%Y-%m-%d")
                amount = round(random.uniform(100.0, 50000.0), 2)
                status = random.choice(statuses)
                cursor.execute(
                    "INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES (?, ?, ?, ?)",
                    (cid, odate, amount, status)
                )
            
            conn.commit()

# Ensure DB is created on module load
_initialize_db()

def execute_read_only_sql(query: str) -> str:
    """
    Executes a SELECT query against the sales database and formats the output.
    Returns an error string if the query is destructive or invalid.
    """
    # Security: rudimentary block on destructive operations
    forbidden_keywords = ["drop", "delete", "update", "insert", "alter", "create", "replace", "grant", "revoke"]
    query_lower = query.lower()
    
    # Check if any forbidden word is present as a whole word
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', query_lower):
            return f"Error: Only SELECT queries are permitted. The keyword '{kw.upper()}' is forbidden."
            
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            if not rows:
                return "Query executed successfully, but returned 0 rows."
                
            # Format as markdown table
            header = "| " + " | ".join(columns) + " |"
            separator = "|-" + "-|-".join(["-" * len(c) for c in columns]) + "-|"
            
            output_rows = []
            for row in rows[:50]: # Limit output to 50 rows to prevent blowing up context window
                output_rows.append("| " + " | ".join(str(val) for val in row) + " |")
                
            table = "\n".join([header, separator] + output_rows)
            
            if len(rows) > 50:
                table += f"\n\n... (Result truncated. Total rows: {len(rows)})"
                
            return table
            
    except sqlite3.Error as e:
        return f"SQL Error: {e}"
    except Exception as e:
        return f"Execution Error: {e}"
