"""Generate a separate synthetic SQLite database with 10,000+ rows.
Run from the project folder: python generate_large_database.py
This creates data/large_business.db and does NOT overwrite the demo DB.
"""
import random
import sqlite3
from pathlib import Path
from datetime import date, timedelta

random.seed(42)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB = DATA_DIR / "large_business.db"

cities = ["Bengaluru", "Mysuru", "Hyderabad", "Chennai", "Mumbai", "Pune", "Delhi"]
categories = ["Electronics", "Furniture", "Accessories", "Home"]
first_names = ["Asha", "Rahul", "Meera", "Arjun", "Kiran", "Divya", "Neha", "Ravi"]
products = [
    (i, f"Product_{i}", random.choice(categories), round(random.uniform(300, 90000), 2))
    for i in range(1, 501)
]

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.executescript("""
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
 customer_id INTEGER PRIMARY KEY, customer_name TEXT, city TEXT, age INTEGER
);
CREATE TABLE products (
 product_id INTEGER PRIMARY KEY, product_name TEXT, category TEXT, price REAL
);
CREATE TABLE orders (
 order_id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
 quantity INTEGER, order_date TEXT
);
""")
cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

customer_rows = []
for i in range(1, 5001):
    customer_rows.append((
        i, f"{random.choice(first_names)}_{i}",
        random.choice(cities), random.randint(18, 75)
    ))
cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customer_rows)

start = date(2025, 1, 1)
order_rows = []
for i in range(1, 20001):
    order_rows.append((
        i, random.randint(1, 5000), random.randint(1, 500),
        random.randint(1, 8),
        (start + timedelta(days=random.randint(0, 699))).isoformat()
    ))
cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", order_rows)
conn.commit()
conn.close()
print(f"Created {DB} with 5,000 customers, 500 products, and 20,000 orders.")
