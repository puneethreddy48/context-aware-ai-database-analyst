
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Context-Aware AI Database Analyst",
    page_icon="🤖",
    layout="wide"
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "business.db"

MODEL_ID = "alpecevit/flan-t5-base-text2sql"


# --------------------------------------------------
# DATABASE CREATION
# --------------------------------------------------

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

        products = [
            (1, "Laptop", "Electronics", 55000),
            (2, "Phone", "Electronics", 25000),
            (3, "Desk", "Furniture", 8000),
            (4, "Headphones", "Electronics", 2000),
            (5, "Office Chair", "Furniture", 6500),
            (6, "Keyboard", "Accessories", 1500)
        ]

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
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            customers
        )

        cursor.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)",
            products
        )

        cursor.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            orders
        )

    conn.commit()
    conn.close()


# --------------------------------------------------
# DATABASE SCHEMA EXTRACTION
# --------------------------------------------------

def get_schema():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    tables = cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name NOT LIKE 'sqlite_%'
    """).fetchall()

    schema = ""

    for table in tables:

        table_name = table[0]

        columns = cursor.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()

        column_names = [
            column[1] + " " + column[2]
            for column in columns
        ]

        schema += (
            f"Table: {table_name}\n"
            f"Columns: {', '.join(column_names)}\n\n"
        )

    conn.close()

    return schema


# --------------------------------------------------
# LOAD PRETRAINED TEXT-TO-SQL MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return tokenizer, model, device


# --------------------------------------------------
# GENERATE SQL
# --------------------------------------------------

def generate_sql(question, schema, tokenizer, model, device):

    prompt = f"""
    Convert the question into a SQLite SQL query.

    Database schema:
    {schema}

    Question:
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

    sql = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return sql.strip()


# --------------------------------------------------
# READ-ONLY SQL VALIDATION
# --------------------------------------------------

def validate_read_only(sql):

    cleaned = sql.strip().rstrip(";").strip()

    if not cleaned:
        raise ValueError("The model generated an empty SQL query.")

    if not (
        cleaned.upper().startswith("SELECT")
        or cleaned.upper().startswith("WITH")
    ):
        raise ValueError("Only SELECT queries are allowed.")

    blocked = [
        "INSERT", "UPDATE", "DELETE", "DROP",
        "ALTER", "CREATE", "REPLACE", "ATTACH",
        "DETACH", "PRAGMA", "VACUUM"
    ]

    upper_sql = cleaned.upper()

    for word in blocked:

        if word in upper_sql:
            raise ValueError(
                f"Blocked SQL operation detected: {word}"
            )

    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed.")

    return cleaned


# --------------------------------------------------
# EXECUTE SQL SAFELY
# --------------------------------------------------

def execute_sql(sql):

    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    try:

        result = pd.read_sql_query(sql, conn)

    finally:

        conn.close()

    return result


# --------------------------------------------------
# SQL ERROR CORRECTION / RETRY
# --------------------------------------------------

def run_with_correction(
    question,
    schema,
    tokenizer,
    model,
    device
):

    sql = generate_sql(
        question,
        schema,
        tokenizer,
        model,
        device
    )

    for attempt in range(2):

        try:

            safe_sql = validate_read_only(sql)

            result = execute_sql(safe_sql)

            return safe_sql, result, None

        except Exception as error:

            if attempt == 0:

                correction_prompt = f"""
                Fix the following SQLite query.

                Database schema:
                {schema}

                Original question:
                {question}

                Incorrect SQL:
                {sql}

                Error:
                {error}

                Return only a corrected SELECT SQL query.
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


# --------------------------------------------------
# DISPLAY ANSWER
# --------------------------------------------------

def display_answer(result):

    if result.empty:

        st.info(
            "The query ran successfully but returned no rows."
        )

        return

    st.subheader("🤖 AI Analyst Answer")

    # Single-value answer, such as COUNT or SUM
    if result.shape == (1, 1):

        answer = result.iloc[0, 0]

        st.success(f"Answer: {answer}")

    elif len(result) == 1:

        st.write(
            "The query returned the following result:"
        )

    else:

        st.write(
            f"The query returned {len(result)} matching records."
        )

    st.subheader("📊 Query Results")

    st.dataframe(
        result,
        use_container_width=True
    )

    st.caption(
        f"{len(result)} row(s), "
        f"{len(result.columns)} column(s) returned."
    )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    st.title("🤖 Context-Aware AI Database Analyst")

    st.write(
        "Ask questions about your business database "
        "using natural language."
    )

    create_demo_database()

    schema = get_schema()

    with st.sidebar:

        st.header("Database Context")

        st.success("Demo SQLite database connected")

        st.write("Available tables:")

        st.code("customers\norders\nproducts")

        st.divider()

        st.subheader("AI Model")

        st.write(MODEL_ID)

        st.caption(
            "This application uses a pretrained "
            "Text-to-SQL model. It is not trained "
            "from scratch in this project."
        )

        with st.expander("View database schema"):

            st.code(schema)

    st.subheader("Ask Your Database")

    examples = [
        "Show all customers",
        "Show customers from Mysuru",
        "How many customers are there?",
        "How many products are available?",
        "What is the total number of orders?",
        "Show all products",
        "Show all orders"
    ]

    selected_question = st.selectbox(
        "Choose an example question",
        ["Select an example"] + examples
    )

    question = st.text_input(
        "Enter your question",
        value=(
            selected_question
            if selected_question != "Select an example"
            else ""
        ),
        placeholder="Example: How many orders are there?"
    )

    if st.button(
        "Generate SQL and Analyze",
        type="primary"
    ):

        if not question.strip():

            st.warning("Please enter a question.")

            return

        with st.spinner(
            "Loading AI model and analyzing your question..."
        ):

            try:

                tokenizer, model, device = load_model()

                sql, result, error = run_with_correction(
                    question,
                    schema,
                    tokenizer,
                    model,
                    device
                )

                st.subheader("Generated SQL")

                st.code(sql, language="sql")

                if error:

                    st.error(
                        f"Unable to execute the generated query: {error}"
                    )

                else:

                    st.success("SQL executed successfully!")

                    display_answer(result)

            except Exception as error:

                st.error(f"Application error: {error}")


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    main()
