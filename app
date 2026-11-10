import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "data" / "business.db"
MODEL_ID = "alpecevit/flan-t5-base-text2sql"

st.set_page_config(page_title="Context-Aware AI Database Analyst", page_icon="🧠", layout="wide")

def create_demo_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        customer_name TEXT NOT NULL,
        city TEXT,
        age INTEGER
    );
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT,
        price REAL
    );
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        order_date TEXT,
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY(product_id) REFERENCES products(product_id)
    );
    """)
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", [
        (1, "Asha", "Bengaluru", 24), (2, "Rahul", "Mysuru", 29),
        (3, "Meera", "Bengaluru", 32), (4, "Arjun", "Hyderabad", 26),
        (5, "Kiran", "Chennai", 35), (6, "Divya", "Bengaluru", 27)
    ])
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", [
        (1, "Laptop", "Electronics", 55000), (2, "Phone", "Electronics", 25000),
        (3, "Desk", "Furniture", 9000), (4, "Headphones", "Electronics", 2500),
        (5, "Office Chair", "Furniture", 7000), (6, "Keyboard", "Accessories", 1800)
    ])
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", [
        (1,1,1,1,"2026-01-10"), (2,2,2,2,"2026-01-12"),
        (3,3,3,1,"2026-02-02"), (4,1,4,3,"2026-02-15"),
        (5,4,1,1,"2026-03-01"), (6,5,5,2,"2026-03-05"),
        (7,6,2,1,"2026-03-11"), (8,2,6,4,"2026-03-18"),
        (9,3,4,2,"2026-04-02"), (10,4,2,1,"2026-04-10")
    ])
    con.commit()
    con.close()

def get_schema():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = con.cursor()
    tables = cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """).fetchall()
    lines = []
    for (table,) in tables:
        cols = cur.execute(f'PRAGMA table_info("{table}")').fetchall()
        col_desc = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        lines.append(f"{table}({col_desc})")
    con.close()
    return "\n".join(lines)

@st.cache_resource(show_spinner="Loading the Text-to-SQL model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

def generate_sql(question, schema, feedback=None):
    tokenizer, model, device = load_model()
    prompt = (
        "transform question and schema to SQL query. "
        f"question: {question} schema: {schema}"
    )
    if feedback:
        prompt += (
            f" The previous SQL failed with this SQLite error: {feedback}. "
            "Correct the SQL using only the tables and columns in the schema. "
            "Return only one SQLite SELECT query."
        )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=180, num_beams=4)
    sql = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    sql = re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"^\s*SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
    return sql

def validate_read_only(sql):
    cleaned = sql.strip()
    if not cleaned:
        raise ValueError("The model returned an empty query.")
    # Permit one read-only statement only.
    if ";" in cleaned.rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed.")
    if not re.match(r"^(SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
        re.IGNORECASE
    )
    if forbidden.search(cleaned):
        raise ValueError("A write or administrative SQL operation was blocked.")
    return cleaned.rstrip(";")

def execute_sql(sql):
    safe_sql = validate_read_only(sql)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(safe_sql)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        return pd.DataFrame([dict(r) for r in rows], columns=columns)
    finally:
        con.close()

def run_with_correction(question, schema, max_attempts=2):
    feedback = None
    attempts = []
    for attempt in range(1, max_attempts + 1):
        sql = generate_sql(question, schema, feedback)
        attempts.append(sql)
        try:
            result = execute_sql(sql)
            return sql, result, attempts, None
        except Exception as exc:
            feedback = str(exc)
    return attempts[-1] if attempts else "", pd.DataFrame(), attempts, feedback

create_demo_database()

st.title("🧠 Context-Aware AI Database Analyst")
st.caption("Ask questions in natural language. The system generates SQL, validates it, executes it safely, and retries once after an error.")

with st.sidebar:
    st.header("Database context")
    st.success("Demo SQLite database connected")
    st.markdown("**Available tables**")
    st.code(get_schema(), language="text")
    st.markdown("**Model**")
    st.write(MODEL_ID)
    st.caption("This prototype uses an existing model fine-tuned for Text-to-SQL. It is not trained from scratch in this application.")

examples = [
    "Show all customers from Bengaluru",
    "List the top 5 products by total quantity ordered",
    "What is the total number of orders?",
    "Show each customer's name and number of orders",
    "Find the total quantity sold for each product category"
]
question = st.text_area("Ask a question about the database", placeholder="Example: Show the top 5 products by total quantity ordered", height=90)
selected = st.selectbox("Or try an example", ["Choose an example..."] + examples)
if selected != "Choose an example..." and not question:
    question = selected

if st.button("Generate SQL and analyze", type="primary", use_container_width=True, disabled=not bool(question.strip())):
    schema = get_schema()
    with st.spinner("Generating SQL, checking it, and executing..."):
        try:
            sql, result, attempts, error = run_with_correction(question.strip(), schema)
            st.subheader("Generated SQL")
            st.code(sql, language="sql")
            if len(attempts) > 1:
                st.info(f"Self-correction attempted. Total generations: {len(attempts)}")
                with st.expander("View generation attempts"):
                    for i, item in enumerate(attempts, start=1):
                        st.markdown(f"**Attempt {i}**")
                        st.code(item, language="sql")
            if error:
                st.error(f"Query could not be executed after retry: {error}")
            else:
                st.success("SQL executed successfully.")
                st.subheader("Query results")
                st.dataframe(result, use_container_width=True)
                st.metric("Rows returned", len(result))
                if result.empty:
                    st.info("The query ran successfully but returned no rows.")
                else:
                    st.subheader("Automatic result summary")
                    st.write(f"The query returned **{len(result)} row(s)** across **{len(result.columns)} column(s)**.")
        except Exception as exc:
            st.error(f"Could not complete the request: {exc}")

st.divider()
st.caption("Academic prototype. Review generated SQL before using any real or sensitive database. The app executes read-only queries only.")
