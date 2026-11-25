
import os
import re
import time
import sqlite3
import logging
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "alpecevit/flan-t5-base-text2sql"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "large_business.db"
DB_PATH = Path(os.environ.get("CONTEXT_DB_PATH", str(DEFAULT_DB_PATH)))

MAX_RESULT_ROWS = 1000
QUERY_TIMEOUT_SECONDS = 8
MAX_NEW_TOKENS = 128

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    uri = "file:" + quote(str(DB_PATH.resolve())) + "?mode=ro"

    conn = sqlite3.connect(
        uri,
        uri=True,
        timeout=QUERY_TIMEOUT_SECONDS
    )

    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={QUERY_TIMEOUT_SECONDS * 1000}")
    conn.execute("PRAGMA query_only = ON")

    return conn


def get_schema():
    conn = get_connection()

    try:
        tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()

        schema = {}

        for row in tables:
            table = row["name"]

            columns = conn.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()

            schema[table] = [
                {
                    "name": col["name"],
                    "type": col["type"],
                    "primary_key": bool(col["pk"])
                }
                for col in columns
            ]

        return schema

    finally:
        conn.close()


def get_database_stats():
    conn = get_connection()

    try:
        tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()

        return {
            row["name"]: conn.execute(
                f'SELECT COUNT(*) FROM "{row["name"]}"'
            ).fetchone()[0]
            for row in tables
        }

    finally:
        conn.close()


def schema_as_text():
    lines = []

    for table, columns in get_schema().items():
        column_names = ", ".join(
            column["name"] for column in columns
        )
        lines.append(f"{table}({column_names})")

    return ", ".join(lines)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    import torch

    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# MODEL SQL GENERATION
# ============================================================

def clean_generated_sql(text):
    text = str(text).strip()

    text = re.sub(
        r"```(?:sql)?",
        "",
        text,
        flags=re.I
    )

    text = text.replace("```", "").strip()

    match = re.search(
        r"\b(SELECT|WITH)\b.*",
        text,
        flags=re.I | re.S
    )

    if not match:
        return ""

    return match.group(0).strip()


def repair_common_table_names(sql):
    replacements = {
        "customer": "customers",
        "product": "products",
        "order": "orders"
    }

    for old, new in replacements.items():
        sql = re.sub(
            rf"\b(FROM|JOIN)\s+{old}\b",
            lambda match: f"{match.group(1)} {new}",
            sql,
            flags=re.I
        )

    return sql


def generate_sql(question, tokenizer, model, device):
    import torch

    schema = schema_as_text()

    prompt = (
        "transform question and schema to SQL query. "
        f"question: {question} "
        f"schema: {schema}."
    )

    logger.info("FLAN-T5 INPUT PROMPT: %s", prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_length = inputs["input_ids"].shape[1]

    logger.info(
        "FLAN-T5 INPUT TOKEN COUNT: %s",
        input_length
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS
        )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    logger.info(
        "FLAN-T5 RAW OUTPUT: %s",
        generated
    )

    sql = clean_generated_sql(generated)
    sql = repair_common_table_names(sql)

    logger.info(
        "FLAN-T5 CLEANED SQL: %s",
        sql
    )

    return sql


# ============================================================
# QUESTION HELPERS
# ============================================================

def has_any(text, words):
    return any(word in text for word in words)


def get_city(question):
    cities = {
        "bengaluru": "Bengaluru",
        "bangalore": "Bengaluru",
        "mysuru": "Mysuru",
        "mysore": "Mysuru",
        "hyderabad": "Hyderabad",
        "chennai": "Chennai",
        "mumbai": "Mumbai",
        "delhi": "Delhi",
        "pune": "Pune",
        "kolkata": "Kolkata"
    }

    for key, value in cities.items():
        if re.search(
            rf"\b{re.escape(key)}\b",
            question,
            re.I
        ):
            return value

    return None


def extract_limit(question, default=None):
    match = re.search(r"\b(\d+)\b", question)

    return int(match.group(1)) if match else default


def make_result(sql, explanation, params=None):
    result = {
        "sql": sql.strip(),
        "explanation": explanation
    }

    if params is not None:
        result["params"] = tuple(params)

    return result


# ============================================================
# RULE-BASED SQL GENERATION
# ============================================================

def context_aware_query(question):

    q = re.sub(
        r"\s+",
        " ",
        question.lower().strip()
    )

    city = get_city(q)

    each = has_any(
        q,
        ["each", "every", "per", "by"]
    )

    # AVERAGE AGE BY CITY

    if (
        has_any(q, ["average age", "avg age", "mean age"])
        and has_any(q, ["city", "cities"])
        and each
    ):
        return make_result("""
            SELECT city,
                   AVG(age) AS average_age
            FROM customers
            GROUP BY city
            ORDER BY average_age DESC
        """, "Calculates average customer age in each city.")

    if city and has_any(q, ["average age", "avg age", "mean age"]):
        return make_result("""
            SELECT AVG(age) AS average_age
            FROM customers
            WHERE LOWER(city) = LOWER(?)
        """, f"Calculates average customer age in {city}.", (city,))

    # CUSTOMER COUNT BY CITY

    if (
        "customer" in q
        and has_any(q, ["count", "number of", "how many"])
        and has_any(q, ["city", "cities"])
        and each
    ):
        return make_result("""
            SELECT city,
                   COUNT(*) AS total_customers
            FROM customers
            GROUP BY city
            ORDER BY total_customers DESC
        """, "Counts customers in each city.")

    # REVENUE BY PRODUCT

    if (
        has_any(q, ["revenue", "sales", "generated the most"])
        and "product" in q
        and each
    ):
        return make_result("""
            SELECT p.product_id,
                   p.product_name,
                   SUM(p.price * o.quantity) AS total_revenue
            FROM products p
            JOIN orders o
              ON p.product_id = o.product_id
            GROUP BY p.product_id, p.product_name
            ORDER BY total_revenue DESC
        """, "Calculates revenue generated by each product.")

    if (
        has_any(q, [
            "highest revenue",
            "most revenue",
            "highest sales"
        ])
        and "product" in q
    ):
        return make_result("""
            SELECT p.product_id,
                   p.product_name,
                   SUM(p.price * o.quantity) AS total_revenue
            FROM products p
            JOIN orders o
              ON p.product_id = o.product_id
            GROUP BY p.product_id, p.product_name
            ORDER BY total_revenue DESC
            LIMIT 1
        """, "Finds the product with the highest total revenue.")

    # REVENUE BY CATEGORY

    if (
        has_any(q, ["revenue", "total sales", "sales amount"])
        and "category" in q
    ):
        return make_result("""
            SELECT p.category,
                   SUM(p.price * o.quantity) AS total_revenue
            FROM products p
            JOIN orders o
              ON p.product_id = o.product_id
            GROUP BY p.category
            ORDER BY total_revenue DESC
        """, "Calculates revenue for each product category.")

    if has_any(q, ["total revenue", "total sales", "sales amount", "revenue"]):
        return make_result("""
            SELECT SUM(p.price * o.quantity) AS total_revenue
            FROM orders o
            JOIN products p
              ON o.product_id = p.product_id
        """, "Calculates total revenue from ordered products.")

    # QUANTITY AGGREGATIONS

    if "total quantity" in q and "product" in q and each:
        return make_result("""
            SELECT p.product_name,
                   SUM(o.quantity) AS total_quantity
            FROM orders o
            JOIN products p
              ON o.product_id = p.product_id
            GROUP BY p.product_id, p.product_name
        """, "Calculates total quantity ordered for each product.")

    if "total quantity" in q and "city" in q and each:
        return make_result("""
            SELECT c.city,
                   SUM(o.quantity) AS total_quantity
            FROM customers c
            JOIN orders o
              ON c.customer_id = o.customer_id
            GROUP BY c.city
        """, "Calculates total quantity ordered in each city.")

    if (
        "average" in q
        and "quantity" in q
        and "customer" in q
        and each
    ):
        return make_result("""
            SELECT c.customer_id,
                   c.customer_name,
                   AVG(o.quantity) AS average_quantity
            FROM customers c
            JOIN orders o
              ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.customer_name
        """, "Calculates average order quantity for each customer.")

    if (
        has_any(q, ["count", "number of"])
        and "order" in q
        and "product" in q
        and each
    ):
        return make_result("""
            SELECT p.product_name,
                   COUNT(o.order_id) AS total_orders
            FROM products p
            JOIN orders o
              ON p.product_id = o.product_id
            GROUP BY p.product_id, p.product_name
        """, "Counts orders for each product.")

    if (
        has_any(q, ["count", "number of"])
        and "order" in q
        and "customer" in q
        and each
    ):
        return make_result("""
            SELECT c.customer_name,
                   COUNT(o.order_id) AS total_orders
            FROM customers c
            JOIN orders o
              ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.customer_name
        """, "Counts orders for each customer.")

    # PRODUCT CATEGORY AGGREGATION

    if "most expensive" in q and "category" in q:
        return make_result("""
            SELECT category,
                   MAX(price) AS maximum_price
            FROM products
            GROUP BY category
        """, "Finds the maximum product price in each category.")

    if (
        has_any(q, ["count", "number of"])
        and "product" in q
        and "category" in q
        and each
    ):
        return make_result("""
            SELECT category,
                   COUNT(*) AS total_products
            FROM products
            GROUP BY category
        """, "Counts products in each category.")

    # CUSTOMER AND PRODUCT JOIN

    if "customer name" in q and "product name" in q:
        return make_result("""
            SELECT c.customer_name,
                   p.product_name
            FROM orders o
            JOIN customers c
              ON o.customer_id = c.customer_id
            JOIN products p
              ON o.product_id = p.product_id
        """, "Displays customer and product names for each order.")

    # CUSTOMER AND ORDER JOINS

    if "customer" in q and "order" in q and "date" in q:
        return make_result("""
            SELECT c.customer_name,
                   o.order_date
            FROM customers c
            JOIN orders o
              ON c.customer_id = o.customer_id
        """, "Displays customer names and order dates.")

    if (
        "customer" in q
        and "order" in q
        and has_any(q, ["order id", "order ids", "orders"])
    ):
        return make_result("""
            SELECT c.customer_name,
                   o.order_id
            FROM customers c
            JOIN orders o
              ON c.customer_id = o.customer_id
        """, "Displays customer names with their order IDs.")

    # PRODUCT AND ORDER JOIN

    if (
        "product" in q
        and "order" in q
        and has_any(q, ["quantity", "quantities"])
    ):
        return make_result("""
            SELECT p.product_name,
                   o.quantity
            FROM orders o
            JOIN products p
              ON o.product_id = p.product_id
        """, "Displays product names with order quantities.")

    # DISTINCT CITIES

    if has_any(q, [
        "list the cities",
        "show the cities",
        "distinct cities",
        "unique cities",
        "list cities"
    ]):
        return make_result(
            "SELECT DISTINCT city FROM customers",
            "Lists unique customer cities."
        )

    # CUSTOMER QUERIES

    if "customer" in q:

        if city:
            return make_result("""
                SELECT *
                FROM customers
                WHERE LOWER(city) = LOWER(?)
            """, f"Finds customers from {city}.", (city,))

        match = re.search(
            r"age\s+(?:is\s+)?between\s+(\d+)\s+and\s+(\d+)",
            q
        )

        if match:
            low, high = map(int, match.groups())

            return make_result(
                "SELECT * FROM customers WHERE age BETWEEN ? AND ?",
                f"Finds customers aged between {low} and {high}.",
                (low, high)
            )

        match = re.search(
            r"(?:older than|above|over|greater than)\s+(\d+)",
            q
        )

        if match:
            age = int(match.group(1))

            return make_result(
                "SELECT * FROM customers WHERE age > ?",
                f"Finds customers older than {age}.",
                (age,)
            )

        match = re.search(
            r"(?:younger than|under|below|less than)\s+(\d+)",
            q
        )

        if match:
            age = int(match.group(1))

            return make_result(
                "SELECT * FROM customers WHERE age < ?",
                f"Finds customers younger than {age}.",
                (age,)
            )

        if has_any(q, [
            "oldest",
            "highest age",
            "highest ages"
        ]):
            limit = extract_limit(q, 10)

            return make_result(
                f"SELECT * FROM customers ORDER BY age DESC LIMIT {limit}",
                f"Displays the {limit} oldest customers."
            )

        if has_any(q, [
            "youngest",
            "lowest age",
            "lowest ages"
        ]):
            limit = extract_limit(q, 5)

            return make_result(
                f"SELECT * FROM customers ORDER BY age ASC LIMIT {limit}",
                f"Displays the {limit} youngest customers."
            )

        if has_any(q, ["how many", "count", "number of customers"]):
            return make_result(
                "SELECT COUNT(*) AS total_customers FROM customers",
                "Counts customers."
            )

        if has_any(q, [
            "show all",
            "all customers",
            "list all",
            "display all",
            "find customers"
        ]):
            return make_result(
                "SELECT * FROM customers",
                "Displays customer records."
            )

    # PRODUCT QUERIES

    if "product" in q:

        match = re.search(
            r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)",
            q
        )

        if match and has_any(q, ["price", "priced", "cost"]):
            low, high = map(float, match.groups())

            return make_result(
                "SELECT * FROM products WHERE price BETWEEN ? AND ?",
                f"Finds products priced between {low:g} and {high:g}.",
                (low, high)
            )

        if has_any(q, [
            "most expensive",
            "highest price",
            "costliest"
        ]):
            limit = extract_limit(q, 1)

            return make_result(
                f"SELECT * FROM products ORDER BY price DESC LIMIT {limit}",
                f"Finds the {limit} most expensive products."
            )

        if has_any(q, [
            "cheapest",
            "lowest price",
            "least expensive"
        ]):
            limit = extract_limit(q, 10)

            return make_result(
                f"SELECT * FROM products ORDER BY price ASC LIMIT {limit}",
                f"Finds the {limit} cheapest products."
            )

        if "electronics" in q:
            return make_result(
                "SELECT * FROM products WHERE LOWER(category) = LOWER(?)",
                "Finds Electronics products.",
                ("Electronics",)
            )

        if "furniture" in q:
            return make_result(
                "SELECT * FROM products WHERE LOWER(category) = LOWER(?)",
                "Finds Furniture products.",
                ("Furniture",)
            )

        match = re.search(
            r"(?:more than|greater than|above|over)\s+(\d+(?:\.\d+)?)",
            q
        )

        if match:
            price = float(match.group(1))

            return make_result(
                "SELECT * FROM products WHERE price > ?",
                "Finds products above the specified price.",
                (price,)
            )

        match = re.search(
            r"(?:less than|below|under)\s+(\d+(?:\.\d+)?)",
            q
        )

        if match:
            price = float(match.group(1))

            return make_result(
                "SELECT * FROM products WHERE price < ?",
                "Finds products below the specified price.",
                (price,)
            )

        if has_any(q, ["how many", "count", "number of products"]):
            return make_result(
                "SELECT COUNT(*) AS total_products FROM products",
                "Counts products."
            )

        if has_any(q, [
            "show all",
            "all products",
            "list all",
            "display all",
            "find products"
        ]):
            return make_result(
                "SELECT * FROM products",
                "Displays product records."
            )

    # ORDER QUERIES

    if "order" in q:

        date_match = re.search(
            r"\b(20\d{2}-\d{2}-\d{2})\b",
            q
        )

        if date_match:
            date_value = date_match.group(1)

            if re.search(r"\bafter\b", q):
                operator, phrase = ">", "after"
            elif re.search(r"\bbefore\b", q):
                operator, phrase = "<", "before"
            else:
                operator, phrase = "=", "on"

            return make_result(
                f"SELECT * FROM orders WHERE order_date {operator} ?",
                f"Finds orders placed {phrase} {date_value}.",
                (date_value,)
            )

        match = re.search(
            r"quantity\s+(?:is\s+)?between\s+(\d+)\s+and\s+(\d+)",
            q
        )

        if match:
            low, high = map(int, match.groups())

            return make_result(
                "SELECT * FROM orders WHERE quantity BETWEEN ? AND ?",
                f"Finds orders with quantity between {low} and {high}.",
                (low, high)
            )

        if has_any(q, [
            "most recent",
            "recent orders",
            "latest orders"
        ]):
            limit = extract_limit(q, 10)

            return make_result(
                f"SELECT * FROM orders ORDER BY order_date DESC LIMIT {limit}",
                f"Displays the {limit} most recent orders."
            )

        if has_any(q, ["how many", "count", "number of orders"]):
            return make_result(
                "SELECT COUNT(*) AS total_orders FROM orders",
                "Counts orders."
            )

        if has_any(q, [
            "show all",
            "all orders",
            "list all",
            "display all"
        ]):
            return make_result(
                "SELECT * FROM orders",
                "Displays order records."
            )

    return None


# ============================================================
# SQL VALIDATION
# ============================================================

FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "REPLACE", "ATTACH", "DETACH",
    "VACUUM", "REINDEX", "PRAGMA"
}


def remove_sql_string_literals(sql):
    return re.sub(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
        " ",
        sql
    )


def validate_sql(sql):
    if not isinstance(sql, str) or not sql.strip():
        return False, "The model did not generate a SQL query."

    cleaned = sql.strip().rstrip(";").strip()

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.I):
        return False, "Only SELECT queries are allowed."

    without_strings = remove_sql_string_literals(cleaned)

    if ";" in without_strings:
        return False, "Multiple SQL statements are not allowed."

    if re.search(r"--|/\*|\*/", without_strings):
        return False, "SQL comments are not allowed."

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(
            rf"\b{keyword}\b",
            without_strings,
            re.I
        ):
            return False, f"Forbidden SQL keyword: {keyword}"

    return True, "SQL validation successful."


# ============================================================
# QUESTION AND SQL RELEVANCE CHECK
# ============================================================

def check_question_sql_alignment(question, sql):
    q = question.lower()
    s = sql.lower()

    expected_tables = set()

    if re.search(r"\bcustomer|client|age|city|cities\b", q):
        expected_tables.add("customers")

    if re.search(r"\bproduct|price|category|revenue|sales\b", q):
        expected_tables.add("products")

    if re.search(r"\border|quantity|revenue|sales\b", q):
        expected_tables.add("orders")

    # Check expected tables
    for table in expected_tables:
        if not re.search(rf"\b{table}\b", s):
            return False, (
                f"The generated SQL does not use the expected {table} table."
            )

    # Check average queries
    if has_any(q, ["average", "avg", "mean"]):
        if not re.search(r"\bAVG\s*\(", s, re.IGNORECASE):
            return False, (
                "The question asks for an average, but SQL has no AVG()."
            )

    # Check count queries
    if has_any(q, ["count", "how many", "number of"]):
        if not re.search(r"\bCOUNT\s*\(", s, re.IGNORECASE):
            return False, (
                "The question asks for a count, but SQL has no COUNT()."
            )

    # Check total revenue queries
    if has_any(q, ["total revenue", "total sales", "sum of"]):
        if not re.search(r"\bSUM\s*\(", s, re.IGNORECASE):
            return False, (
                "The question asks for a total, but SQL has no SUM()."
            )

    # Check grouped queries
    if has_any(q, [
        "each city", "every city", "per city", "by city",
        "each product", "every product", "per product", "by product"
    ]):
        if "group by" not in s:
            return False, (
                "The question asks for grouped results, but SQL has no GROUP BY."
            )

    return True, "Question and SQL appear relevant."


# ============================================================
# SQLITE SECURITY
# ============================================================

def sqlite_authorizer(action, arg1, arg2, database, source):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION
    }

    if hasattr(sqlite3, "SQLITE_RECURSIVE"):
        allowed.add(sqlite3.SQLITE_RECURSIVE)

    return (
        sqlite3.SQLITE_OK
        if action in allowed
        else sqlite3.SQLITE_DENY
    )


# ============================================================
# SQL EXECUTION
# ============================================================

def execute_sql(sql, params=None, use_validation=True):

    if use_validation:
        valid, message = validate_sql(sql)

        if not valid:
            raise ValueError(message)

    conn = get_connection()

    try:
        conn.set_authorizer(sqlite_authorizer)

        start_time = time.monotonic()

        conn.set_progress_handler(
            lambda: int(
                time.monotonic() - start_time > QUERY_TIMEOUT_SECONDS
            ),
            1000
        )

        cursor = conn.execute(sql, params or ())

        if cursor.description is None:
            raise ValueError("Query did not return a result set.")

        columns = [
            item[0] for item in cursor.description
        ]

        rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)

        truncated = len(rows) > MAX_RESULT_ROWS
        rows = rows[:MAX_RESULT_ROWS]

        dataframe = pd.DataFrame(
            [tuple(row) for row in rows],
            columns=columns
        )

        return dataframe, truncated

    finally:
        conn.close()


# ============================================================
# SQL CORRECTION
# ============================================================

def correct_sql(
    question,
    incorrect_sql,
    error_message,
    tokenizer,
    model,
    device
):
    import torch

    prompt = (
        "transform question and schema to SQL query. "
        f"question: {question}. "
        f"schema: {schema_as_text()}. "
        f"Previous SQL: {incorrect_sql}. "
        f"SQLite error: {error_message}. "
        "Generate one corrected SQLite SELECT query only."
    )

    logger.info("SQL CORRECTION PROMPT: %s", prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS
        )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    logger.info("SQL CORRECTION RAW OUTPUT: %s", generated)

    sql = clean_generated_sql(generated)
    sql = repair_common_table_names(sql)

    logger.info("SQL CORRECTION CLEANED SQL: %s", sql)

    return sql


def execute_with_optional_correction(
    question,
    sql,
    params=None,
    tokenizer=None,
    model=None,
    device=None,
    use_validation=True,
    allow_correction=True
):
    try:
        dataframe, truncated = execute_sql(
            sql,
            params=params,
            use_validation=use_validation
        )

        return sql, dataframe, truncated, False

    except Exception as first_error:

        logger.warning("SQL execution failed: %s", first_error)

        if (
            not allow_correction
            or tokenizer is None
            or model is None
        ):
            raise

        corrected_sql = correct_sql(
            question,
            sql,
            str(first_error),
            tokenizer,
            model,
            device
        )

        valid, message = validate_sql(corrected_sql)

        if not valid:
            raise ValueError(
                f"SQL correction produced an invalid query: {message}\n"
                f"Corrected SQL: {corrected_sql}"
            )

        aligned, alignment_message = check_question_sql_alignment(
            question,
            corrected_sql
        )

        if not aligned:
            raise ValueError(
                "The corrected SQL does not match the question. "
                + alignment_message
                + f"\nCorrected SQL: {corrected_sql}"
            )

        dataframe, truncated = execute_sql(
            corrected_sql,
            params=None,
            use_validation=True
        )

        return corrected_sql, dataframe, truncated, True


# ============================================================
# MAIN QUERY PIPELINE
# ============================================================

def run_ai_query(
    question,
    mode="Hybrid",
    tokenizer=None,
    model=None,
    device=None,
    use_validation=True,
    allow_correction=True
):

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    rule = context_aware_query(question)

    # RULES ONLY

    if mode == "Rules only":

        if rule is None:
            raise ValueError(
                "No matching rule found. Switch to Hybrid mode "
                "to process questions outside the example list."
            )

        sql, dataframe, truncated, corrected = (
            execute_with_optional_correction(
                question,
                rule["sql"],
                rule.get("params"),
                tokenizer,
                model,
                device,
                use_validation,
                False
            )
        )

        return {
            "sql": sql,
            "dataframe": dataframe,
            "method": "Rules",
            "explanation": rule["explanation"],
            "correction_used": corrected,
            "truncated": truncated
        }

    if mode not in ["Hybrid", "Model only"]:
        raise ValueError(f"Unknown execution mode: {mode}")

    # HYBRID: FIRST TRY RULES

    if mode == "Hybrid" and rule is not None:

        try:
            sql, dataframe, truncated, corrected = (
                execute_with_optional_correction(
                    question,
                    rule["sql"],
                    rule.get("params"),
                    tokenizer,
                    model,
                    device,
                    use_validation,
                    False
                )
            )

            return {
                "sql": sql,
                "dataframe": dataframe,
                "method": "Rules",
                "explanation": rule["explanation"],
                "correction_used": corrected,
                "truncated": truncated
            }

        except Exception as error:
            logger.warning(
                "Rule execution failed. Switching to model: %s",
                error
            )

    # FLAN-T5 GENERATION

    if tokenizer is None or model is None:
        tokenizer, model, device = load_model()

    sql = generate_sql(
        question,
        tokenizer,
        model,
        device
    )

    valid, message = validate_sql(sql)

    if not valid:
        raise ValueError(
            "FLAN-T5 could not generate a valid SQL query.\n"
            f"Reason: {message}\n"
            f"Generated SQL: {sql}"
        )

    aligned, alignment_message = check_question_sql_alignment(
        question,
        sql
    )

    if not aligned:
        raise ValueError(
            "I couldn't confidently match the generated SQL to your question.\n\n"
            f"Reason: {alignment_message}\n\n"
            f"Generated SQL:\n{sql}\n\n"
            "No unrelated results were returned."
        )

    sql, dataframe, truncated, corrected = (
        execute_with_optional_correction(
            question,
            sql,
            None,
            tokenizer,
            model,
            device,
            use_validation,
            allow_correction
        )
    )

    return {
        "sql": sql,
        "dataframe": dataframe,
        "method": (
            "FLAN-T5"
            if mode == "Model only"
            else "FLAN-T5 fallback"
        ),
        "explanation": "SQL generated by the FLAN-T5 text-to-SQL model.",
        "correction_used": corrected,
        "truncated": truncated
    }


# ============================================================
# EVALUATION COMPATIBILITY
# ============================================================

def sql_literal(value):

    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"

    raise TypeError(
        f"Unsupported parameter type: {type(value).__name__}"
    )


def generate_rule_sql(question):
    rule = context_aware_query(question)

    if rule is None:
        return None

    sql = rule["sql"]

    for value in rule.get("params", ()):

        if "?" not in sql:
            raise ValueError("Missing SQL parameter placeholder.")

        sql = sql.replace(
            "?",
            sql_literal(value),
            1
        )

    if "?" in sql:
        raise ValueError("Unfilled SQL parameter placeholder.")

    return sql


def generate_model_sql(question, tokenizer, model, device):
    return generate_sql(
        question,
        tokenizer,
        model,
        device
    )


# ============================================================
# VISUALIZATION
# ============================================================

def show_visualization(dataframe):

    if dataframe.empty or len(dataframe.columns) < 2:
        return

    numeric = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    text_columns = [
        column
        for column in dataframe.columns
        if column not in numeric
    ]

    if not numeric or not text_columns:
        return

    st.subheader("Data Visualization")

    chart_type = st.selectbox(
        "Chart type",
        ["Bar chart", "Line chart"]
    )

    x_column = st.selectbox(
        "Category",
        text_columns
    )

    y_column = st.selectbox(
        "Numerical value",
        numeric
    )

    chart_data = dataframe[
        [x_column, y_column]
    ].dropna()

    if chart_data.empty:
        st.info("No data available for this chart.")
        return

    chart_data = chart_data.set_index(x_column)

    if chart_type == "Bar chart":
        st.bar_chart(chart_data[y_column])
    else:
        st.line_chart(chart_data[y_column])


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def main():

    st.set_page_config(
        page_title="Context-Aware AI Database Analyst",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Context-Aware AI Database Analyst")

    st.write(
        "Ask questions about your database in natural language. "
        "You can type your own question even if it is not in the example list."
    )

    # DATABASE INFORMATION

    try:
        stats = get_database_stats()

        st.sidebar.header("Database Information")
        st.sidebar.success("Database connected")
        st.sidebar.write(f"Database: {DB_PATH.name}")

        st.sidebar.metric(
            "Total records",
            f"{sum(stats.values()):,}"
        )

        for table, count in stats.items():
            st.sidebar.write(
                f"**{table}:** {count:,} records"
            )

    except Exception as error:
        st.error(f"Database connection failed: {error}")
        st.stop()

    # DATABASE SCHEMA

    with st.expander("View Database Schema"):

        for table, columns in get_schema().items():

            st.markdown(f"**Table: {table}**")

            st.dataframe(
                pd.DataFrame(columns),
                width="stretch",
                hide_index=True
            )

    # EXECUTION MODE

    mode = st.sidebar.selectbox(
        "Execution method",
        ["Hybrid", "Rules only", "Model only"]
    )

    st.sidebar.caption(
        "Hybrid uses predefined rules when possible. "
        "For questions outside the examples, it uses FLAN-T5."
    )

    st.sidebar.divider()
    st.sidebar.write(f"Model: `{MODEL_ID}`")
    st.sidebar.write("Database: SQLite")
    st.sidebar.write("Query type: Read-only")

    # EXAMPLE QUESTIONS

    examples = [
        "Show all customers",
        "Show all products",
        "Show all orders",
        "Show customer names and cities",
        "Find customers from Bengaluru",
        "Find customers older than 40",
        "Find customers younger than 30",
        "Find customers whose age is between 30 and 50",
        "Find the average customer age",
        "Find the average age of customers in each city",
        "Find the average age of customers in Bengaluru",
        "List the cities where customers live",
        "Find the number of customers in each city, highest first",
        "Find products costing more than 10000",
        "Find products costing less than 5000",
        "Find products priced between 1000 and 10000",
        "Show the top 5 most expensive products",
        "Show the 10 products with the lowest prices",
        "Find the most expensive product in each category",
        "Find orders with quantity greater than 5",
        "Find orders placed after 2025-06-01",
        "Show the 10 most recent orders",
        "Show customer names with their orders",
        "Show customer names and their order dates",
        "Show product names with their order quantities",
        "Find the total quantity ordered for each product",
        "Count the number of orders for each customer",
        "Find the average order quantity for each customer",
        "Find the total quantity ordered in each city",
        "Find the total sales amount for each product category",
        "Calculate the total revenue from all products ordered",
        "Which product has generated the highest revenue?"
    ]

    st.subheader("Ask Your Database")

    selected = st.selectbox(
        "Try an example question",
        ["Choose an example"] + examples
    )

    question = st.text_area(
        "Enter your natural language question",
        value=selected if selected != "Choose an example" else "",
        placeholder="Type any database question here..."
    )

    # RUN QUERY

    if st.button(
        "Run Query",
        type="primary",
        width="stretch"
    ):

        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:

            with st.spinner(
                "Understanding your question and generating SQL..."
            ):

                result = run_ai_query(
                    question=question,
                    mode=mode,
                    use_validation=True,
                    allow_correction=True
                )

            st.success("Query executed successfully.")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Execution method",
                result["method"]
            )

            col2.metric(
                "Returned rows",
                len(result["dataframe"])
            )

            col3.metric(
                "Correction used",
                "Yes" if result["correction_used"] else "No"
            )

            st.subheader("Generated SQL")

            st.code(
                result["sql"],
                language="sql"
            )

            st.subheader("Explanation")
            st.write(result["explanation"])

            st.subheader("Query Results")

            st.dataframe(
                result["dataframe"],
                width="stretch",
                hide_index=True
            )

            if result["truncated"]:
                st.warning(
                    f"Only the first {MAX_RESULT_ROWS} rows are displayed."
                )

            show_visualization(result["dataframe"])

            csv_data = (
                result["dataframe"]
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "Download Results as CSV",
                data=csv_data,
                file_name="query_results.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(str(error))

            st.info(
                "Try rephrasing your question using the database tables "
                "and columns: customers, products, orders, age, city, "
                "price, quantity, order date or revenue. "
                "No unrelated results have been displayed."
            )


if __name__ == "__main__":
    main()
