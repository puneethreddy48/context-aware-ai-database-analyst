
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==================================================
# 1. APP CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Context-Aware AI Database Analyst",
    page_icon="🤖",
    layout="wide"
)

APP_DIR = Path("/content")
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "business.db"

MODEL_ID = "alpecevit/flan-t5-base-text2sql"


# ==================================================
# 2. DATABASE CREATION
# ==================================================

def create_demo_database():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT,
            city TEXT,
            age INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            price REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT
        )
    """)

    customer_count = cursor.execute(
        "SELECT COUNT(*) FROM customers"
    ).fetchone()[0]

    if customer_count == 0:

        customers = [
            (1, "Asha", "Bengaluru", 24),
            (2, "Rahul", "Mysuru", 28),
            (3, "Meera", "Hyderabad", 25),
            (4, "Arjun", "Chennai", 30),
            (5, "Kiran", "Bengaluru", 26),
            (6, "Divya", "Mysuru", 23)
        ]

        cursor.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            customers
        )

    product_count = cursor.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    if product_count == 0:

        products = [
            (1, "Laptop", "Electronics", 55000),
            (2, "Phone", "Electronics", 25000),
            (3, "Desk", "Furniture", 8000),
            (4, "Headphones", "Electronics", 2000),
            (5, "Office Chair", "Furniture", 6500),
            (6, "Keyboard", "Accessories", 1500)
        ]

        cursor.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)",
            products
        )

    order_count = cursor.execute(
        "SELECT COUNT(*) FROM orders"
    ).fetchone()[0]

    if order_count == 0:

        orders = [
            (1, 1, 1, 1, "2026-01-01"),
            (2, 2, 2, 2, "2026-01-02"),
            (3, 3, 3, 1, "2026-01-03"),
            (4, 4, 4, 3, "2026-01-04"),
            (5, 5, 5, 1, "2026-01-05"),
            (6, 6, 6, 2, "2026-01-06"),
            (7, 1, 2, 1, "2026-01-07"),
            (8, 2, 3, 2, "2026-01-08"),
            (9, 3, 1, 1, "2026-01-09"),
            (10, 4, 6, 1, "2026-01-10")
        ]

        cursor.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            orders
        )

    conn.commit()
    conn.close()


# ==================================================
# 3. DATABASE SCHEMA EXTRACTION
# ==================================================

def get_schema():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables = cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
    """).fetchall()

    schema = ""

    for (table_name,) in tables:

        columns = cursor.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()

        column_names = [
            f"{column[1]} {column[2]}"
            for column in columns
        ]

        schema += (
            f"Table: {table_name}\n"
            f"Columns: {', '.join(column_names)}\n\n"
        )

    conn.close()

    return schema


# ==================================================
# 4. LOAD PRETRAINED TEXT-TO-SQL MODEL
# ==================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ==================================================
# 5. AI SQL GENERATION
# ==================================================

def generate_sql(question, schema, tokenizer, model, device):

    prompt = f"""
    You are an expert SQLite query generator.

    Convert the user's question into one valid SQLite SELECT query.

    DATABASE SCHEMA:
    {schema}

    RULES:
    - Use exact table and column names.
    - For showing all records, use SELECT *.
    - For customers from a city, return all customer columns.
    - Use appropriate WHERE conditions.
    - For counting records, use COUNT(*).
    - Only generate SELECT queries.
    - Do not explain the SQL.
    - Return SQL only.

    USER QUESTION:
    {question}

    SQL:
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            num_beams=4
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


# ==================================================
# 6. CONTEXT-AWARE QUESTION HANDLING
# ==================================================

def normalize_question(question):

    question = question.lower().strip()
    question = re.sub(r"[?!.,]", " ", question)
    question = re.sub(r"\s+", " ", question)

    return question


def context_aware_query(question):

    q = normalize_question(question)

    city_aliases = {
        "bengluru": "Bengaluru",
        "bengalore": "Bengaluru",
        "banglore": "Bengaluru",
        "bangalore": "Bengaluru",
        "bengaluru": "Bengaluru",
        "mysore": "Mysuru",
        "mysuru": "Mysuru",
        "hyderabad": "Hyderabad",
        "chennai": "Chennai"
    }

    # CUSTOMER QUESTIONS

    if "customer" in q:

        for alias, city in city_aliases.items():

            if re.search(r"\b" + re.escape(alias) + r"\b", q):

                return (
                    "SELECT * FROM customers "
                    "WHERE LOWER(city) = LOWER(?)",
                    (city,),
                    f"Showing customers from {city}."
                )

        if any(word in q for word in [
            "how many", "count", "number of", "total"
        ]):

            return (
                "SELECT COUNT(*) AS total_customers FROM customers",
                (),
                "Here is the total number of customers."
            )

        if any(word in q for word in [
            "show", "display", "list", "all"
        ]):

            return (
                "SELECT * FROM customers",
                (),
                "Showing all customer records."
            )

    # PRODUCT QUESTIONS

    if "product" in q:

        if any(word in q for word in [
            "how many", "count", "number of", "total"
        ]):

            return (
                "SELECT COUNT(*) AS total_products FROM products",
                (),
                "Here is the total number of products."
            )

        if any(word in q for word in [
            "show", "list", "display", "all"
        ]):

            return (
                "SELECT * FROM products",
                (),
                "Showing all products."
            )

    # ORDER QUESTIONS

    if "order" in q:

        if any(word in q for word in [
            "how many", "count", "number of", "total"
        ]):

            return (
                "SELECT COUNT(*) AS total_orders FROM orders",
                (),
                "Here is the total number of orders."
            )

        if any(word in q for word in [
            "show", "list", "display", "all"
        ]):

            return (
                "SELECT * FROM orders",
                (),
                "Showing all orders."
            )

    return None


# ==================================================
# 7. READ-ONLY SQL VALIDATION
# ==================================================

def validate_read_only(sql):

    cleaned = sql.strip().rstrip(";").strip()

    if not cleaned:
        raise ValueError("Empty SQL query generated.")

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.I):
        raise ValueError("Only SELECT queries are permitted.")

    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed.")

    return cleaned


# ==================================================
# 8. SAFE SQL EXECUTION
# ==================================================

def execute_sql(sql, parameters=()):

    safe_sql = validate_read_only(sql)

    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    try:

        def authorizer(action, arg1, arg2, database, trigger):

            denied_actions = {
                sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_UPDATE,
                sqlite3.SQLITE_DELETE,
                sqlite3.SQLITE_CREATE_TABLE,
                sqlite3.SQLITE_DROP_TABLE,
                sqlite3.SQLITE_ALTER_TABLE,
                sqlite3.SQLITE_ATTACH,
                sqlite3.SQLITE_DETACH
            }

            if action in denied_actions:
                return sqlite3.SQLITE_DENY

            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorizer)

        result = pd.read_sql_query(
            safe_sql,
            conn,
            params=parameters
        )

        return result

    finally:
        conn.close()


# ==================================================
# 9. AI QUERY RETRY
# ==================================================

def run_ai_query(question, schema, tokenizer, model, device):

    sql = generate_sql(
        question, schema, tokenizer, model, device
    )

    for attempt in range(2):

        try:

            result = execute_sql(sql)

            return sql, result, None

        except Exception as error:

            if attempt == 0:

                correction_prompt = f"""
                Correct this SQLite SELECT query.

                Schema:
                {schema}

                Question:
                {question}

                Incorrect SQL:
                {sql}

                Error:
                {error}

                Return only one valid SELECT query.
                """

                inputs = tokenizer(
                    correction_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)

                with torch.no_grad():

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        num_beams=4
                    )

                sql = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip()

            else:

                return sql, None, str(error)

    return sql, None, "Query could not be executed."


# ==================================================
# 10. ANSWER DISPLAY
# ==================================================

def display_answer(result, explanation):

    if result.empty:

        st.warning(
            "The query executed successfully, but no matching records were found."
        )

        return

    st.subheader("🤖 AI Analyst Answer")

    st.success(explanation)

    if result.shape == (1, 1):

        value = result.iloc[0, 0]
        column = result.columns[0]

        st.metric(
            label=column.replace("_", " ").title(),
            value=value
        )

    elif len(result) == 1:

        st.write("Here is the matching record.")

    else:

        st.write(f"Found {len(result)} matching records.")

    st.subheader("📊 Query Results")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"{len(result)} row(s) and "
        f"{len(result.columns)} column(s) returned."
    )


# ==================================================
# 11. MAIN DASHBOARD
# ==================================================

def main():

    st.title("🤖 Context-Aware AI Database Analyst")

    st.markdown("""
    Ask questions about your business database in natural language.

    The system uses a pretrained Text-to-SQL model,
    database schema context, and context-aware handling
    for common business questions.
    """)

    create_demo_database()

    schema = get_schema()

    with st.sidebar:

        st.header("Database Context")

        st.success("SQLite database connected")

        st.write("Available tables:")

        st.code("customers\nproducts\norders")

        st.divider()

        st.subheader("AI Model")

        st.write(MODEL_ID)

        st.caption(
            "Pretrained Text-to-SQL model. "
            "This project does not train the model from scratch."
        )

        with st.expander("View database schema"):
            st.code(schema)

    st.subheader("Ask Your Database")

    examples = [
        "Show all customers",
        "Show customers from Bengaluru",
        "Show customers from Mysuru",
        "How many customers are there?",
        "Show all products",
        "How many products are available?",
        "Show all orders",
        "What is the total number of orders?"
    ]

    selected = st.selectbox(
        "Try an example question",
        ["Choose a question"] + examples
    )

    question = st.text_input(
        "Enter your question",
        value="" if selected == "Choose a question" else selected,
        placeholder="Example: Show customers from Bengaluru"
    )

    if st.button("Generate SQL and Analyze", type="primary"):

        if not question.strip():

            st.warning("Please enter a question.")
            return

        try:

            with st.spinner("Understanding your question..."):

                rule_result = context_aware_query(question)

                if rule_result is not None:

                    sql, parameters, explanation = rule_result

                    result = execute_sql(sql, parameters)

                    source = "Context-aware query handling"

                else:

                    with st.spinner(
                        "Loading AI model and generating SQL..."
                    ):

                        tokenizer, model, device = load_model()

                        sql, result, error = run_ai_query(
                            question,
                            schema,
                            tokenizer,
                            model,
                            device
                        )

                    if error:

                        st.subheader("Generated SQL")
                        st.code(sql, language="sql")

                        st.error(
                            f"Could not execute the generated query: {error}"
                        )
                        return

                    explanation = (
                        "The AI model generated a query and "
                        "the database executed it successfully."
                    )

                    source = "Pretrained Text-to-SQL model"

            st.subheader("Generated SQL")

            st.code(sql, language="sql")

            st.caption(f"Query method: {source}")

            st.success("SQL executed successfully!")

            display_answer(result, explanation)

        except Exception as error:

            st.error(f"Application error: {error}")


# ==================================================
# 12. RUN APPLICATION
# ==================================================

if __name__ == "__main__":
    main()
