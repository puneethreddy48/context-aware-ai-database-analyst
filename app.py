
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
MAX_NEW_TOKENS = 256

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
        names = ", ".join(
            f'{column["name"]} ({column["type"]})'
            for column in columns
        )
        lines.append(f"{table}({names})")

    return "\n".join(lines)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def clean_generated_sql(text):
    text = str(text).strip()
    text = re.sub(r"```(?:sql)?", "", text, flags=re.I)
    text = text.replace("```", "").strip()

    match = re.search(
        r"\b(SELECT|WITH)\b.*",
        text,
        flags=re.I | re.S
    )

    return match.group(0).strip() if match else text


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

    prompt = f"""
Translate the question into one valid SQLite SELECT query.

DATABASE SCHEMA:
{schema_as_text()}

TABLES:
customers(customer_id, customer_name, city, age)
products(product_id, product_name, category, price)
orders(order_id, customer_id, product_id, quantity, order_date)

JOIN RULES:
customers.customer_id = orders.customer_id
products.product_id = orders.product_id

INSTRUCTIONS:
- Use only the listed tables and columns.
- Use plural table names.
- Use SELECT * when complete records are requested.
- Use only requested columns for projection questions.
- Use COUNT for counts, SUM for totals, AVG for averages.
- Use MIN and MAX for minimum and maximum values.
- Use GROUP BY for each, every, or per.
- Use ORDER BY for sorting.
- Use LIMIT only when requested.
- Use order_date > 'DATE' for after a date.
- Use order_date < 'DATE' for before a date.
- Return one SELECT statement only.
- Do not explain the query.
- Do not invent columns or tables.

QUESTION:
{question}

SQL:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True
        )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return repair_common_table_names(clean_generated_sql(generated))


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
        if re.search(rf"\b{re.escape(key)}\b", question, re.I):
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
    q = re.sub(r"\s+", " ", question.lower().strip())
    city = get_city(q)

    # ----- Revenue and sales -----

    if (
        has_any(q, ["total sales", "sales amount", "revenue"])
        and has_any(q, ["category", "each", "per"])
    ):
        return make_result("""
            SELECT p.category,
                   SUM(p.price * o.quantity) AS total_sales
            FROM products p
            JOIN orders o ON p.product_id = o.product_id
            GROUP BY p.category
        """, "Calculates total sales amount for each product category.")

    if has_any(q, ["revenue", "total sales", "sales amount"]):
        return make_result("""
            SELECT SUM(p.price * o.quantity) AS total_revenue
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
        """, "Calculates total revenue from ordered products.")

    # ----- Aggregation across joins -----

    if (
        "total quantity" in q
        and "product" in q
        and has_any(q, ["each", "every", "per"])
    ):
        return make_result("""
            SELECT p.product_name,
                   SUM(o.quantity) AS total_quantity
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            GROUP BY p.product_id, p.product_name
        """, "Calculates total quantity ordered for each product.")

    if (
        "total quantity" in q
        and "city" in q
        and has_any(q, ["each", "every", "per"])
    ):
        return make_result("""
            SELECT c.city,
                   SUM(o.quantity) AS total_quantity
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.city
        """, "Calculates total ordered quantity for each city.")

    if (
        "average" in q and "quantity" in q
        and "customer" in q
        and has_any(q, ["each", "every", "per"])
    ):
        return make_result("""
            SELECT c.customer_id,
                   AVG(o.quantity) AS average_quantity
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id
        """, "Calculates average order quantity for each customer.")

    if (
        has_any(q, ["count", "number of"])
        and "order" in q and "product" in q
        and has_any(q, ["each", "every", "per"])
    ):
        return make_result("""
            SELECT p.product_name,
                   COUNT(o.order_id) AS total_orders
            FROM products p
            JOIN orders o ON p.product_id = o.product_id
            GROUP BY p.product_id, p.product_name
        """, "Counts orders for each product.")

    if (
        has_any(q, ["count", "number of"])
        and "order" in q and "customer" in q
        and has_any(q, ["each", "every", "per"])
    ):
        return make_result("""
            SELECT c.customer_name,
                   COUNT(o.order_id) AS total_orders
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.customer_name
        """, "Counts orders for each customer.")

    # ----- City aggregation -----

    if (
        "customer" in q
        and has_any(q, ["count", "number of"])
        and has_any(q, ["city", "cities"])
        and has_any(q, ["each", "every", "per"])
    ):
        sql = """
            SELECT city, COUNT(*) AS total_customers
            FROM customers
            GROUP BY city
        """

        if has_any(q, ["highest first", "descending", "largest first"]):
            sql += " ORDER BY total_customers DESC"

        return make_result(sql, "Counts customers in each city.")

    if city and has_any(q, ["average age", "avg age"]):
        return make_result("""
            SELECT AVG(age) AS average_age
            FROM customers
            WHERE LOWER(city) = LOWER(?)
        """, f"Calculates average customer age in {city}.", (city,))

    # ----- Product category aggregation -----

    if "most expensive" in q and "category" in q:
        return make_result("""
            SELECT category, MAX(price) AS maximum_price
            FROM products
            GROUP BY category
        """, "Finds the maximum product price in each category.")

    if (
        has_any(q, ["count", "number of"])
        and "product" in q and "category" in q
        and has_any(q, ["each", "every", "per"])
    ):
        return make_result("""
            SELECT category, COUNT(*) AS total_products
            FROM products
            GROUP BY category
        """, "Counts products in each category.")

    # ----- Customer and product join -----

    if "customer name" in q and "product name" in q:
        return make_result("""
            SELECT c.customer_name, p.product_name
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN products p ON o.product_id = p.product_id
        """, "Displays customer and product names for each order.")

    # ----- Customer and order joins -----

    if (
        "customer" in q and "order" in q
        and "date" in q
    ):
        return make_result("""
            SELECT c.customer_name, o.order_date
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
        """, "Displays customer names and their order dates.")

    if (
        "customer" in q and "order" in q
        and has_any(q, ["order id", "order ids", "orders"])
    ):
        return make_result("""
            SELECT c.customer_name, o.order_id
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
        """, "Displays customer names with their order IDs.")

    # ----- Product and order join -----

    if (
        "product" in q and "order" in q
        and has_any(q, ["quantity", "quantities"])
    ):
        return make_result("""
            SELECT p.product_name, o.quantity
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
        """, "Displays product names with their order quantities.")

    # ----- Distinct cities -----

    if has_any(q, [
        "list the cities", "show the cities",
        "distinct cities", "unique cities"
    ]):
        return make_result(
            "SELECT DISTINCT city FROM customers",
            "Lists unique customer cities."
        )

    # ----- Customer queries -----

    if "customer" in q:
        if city:
            return make_result("""
                SELECT *
                FROM customers
                WHERE LOWER(city) = LOWER(?)
            """, f"Finds customers from {city}.", (city,))

        match = re.search(
            r"age\s+(?:is\s+)?between\s+(\d+)\s+and\s+(\d+)", q
        )
        if match:
            low, high = map(int, match.groups())
            return make_result(
                "SELECT * FROM customers WHERE age BETWEEN ? AND ?",
                f"Finds customers aged between {low} and {high}.",
                (low, high)
            )

        match = re.search(
            r"age\s+(?:is\s+)?(?:exactly|equal to)\s+(\d+)", q
        )
        if match:
            age = int(match.group(1))
            return make_result(
                "SELECT * FROM customers WHERE age = ?",
                f"Finds customers aged exactly {age}.",
                (age,)
            )

        match = re.search(
            r"(?:older than|above|over|greater than)\s+(\d+)", q
        )
        if match:
            age = int(match.group(1))
            return make_result(
                "SELECT * FROM customers WHERE age > ?",
                f"Finds customers older than {age}.",
                (age,)
            )

        match = re.search(
            r"(?:younger than|under|below|less than)\s+(\d+)", q
        )
        if match:
            age = int(match.group(1))
            return make_result(
                "SELECT * FROM customers WHERE age < ?",
                f"Finds customers younger than {age}.",
                (age,)
            )

        if has_any(q, ["highest ages", "highest age", "oldest customers"]):
            limit = extract_limit(q, 10)
            return make_result(
                f"SELECT * FROM customers ORDER BY age DESC LIMIT {limit}",
                f"Displays the {limit} oldest customers."
            )

        if has_any(q, ["youngest customers", "lowest ages", "lowest age"]):
            limit = extract_limit(q, 5)
            return make_result(
                f"SELECT * FROM customers ORDER BY age ASC LIMIT {limit}",
                f"Displays the {limit} youngest customers."
            )

        if "sorted by age" in q or "sort by age" in q:
            direction = (
                "DESC" if has_any(q, ["descending", "high to low"]) else "ASC"
            )
            return make_result(
                f"SELECT * FROM customers ORDER BY age {direction}",
                f"Sorts customers by age {direction}."
            )

        if "customer names and cities" in q:
            return make_result(
                "SELECT customer_name, city FROM customers",
                "Displays customer names and cities."
            )

        if has_any(q, ["how many", "count", "number of customers"]):
            return make_result(
                "SELECT COUNT(*) AS total_customers FROM customers",
                "Counts customers."
            )

        if has_any(q, [
            "show all", "all customers", "list all",
            "display all", "find customers"
        ]):
            return make_result(
                "SELECT * FROM customers",
                "Displays customer records."
            )

    # ----- Product queries -----

    if "product" in q:
        match = re.search(
            r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", q
        )
        if match and has_any(q, ["price", "priced", "cost"]):
            low, high = map(float, match.groups())
            return make_result(
                "SELECT * FROM products WHERE price BETWEEN ? AND ?",
                f"Finds products priced between {low:g} and {high:g}.",
                (low, high)
            )

        if has_any(q, [
            "most expensive", "highest price",
            "highest priced", "costliest"
        ]):
            limit = extract_limit(q)
            if limit is not None:
                return make_result(
                    f"SELECT * FROM products ORDER BY price DESC LIMIT {limit}",
                    f"Finds the {limit} most expensive products."
                )

        if has_any(q, [
            "lowest prices", "lowest price",
            "cheapest products", "least expensive products"
        ]):
            limit = extract_limit(q, 10)
            return make_result(
                f"SELECT * FROM products ORDER BY price ASC LIMIT {limit}",
                f"Finds the {limit} lowest-priced products."
            )

        if "sorted by price" in q or "sort by price" in q:
            direction = (
                "DESC" if has_any(q, ["descending", "high to low"]) else "ASC"
            )
            return make_result(
                f"SELECT * FROM products ORDER BY price {direction}",
                f"Sorts products by price {direction}."
            )

        category_match = re.search(
            r"\b(?:in|under|within)\s+the\s+([a-z ]+?)\s+category\b", q
        )
        if category_match:
            category = category_match.group(1).strip().title()
            return make_result(
                "SELECT * FROM products WHERE LOWER(category) = LOWER(?)",
                f"Finds products in the {category} category.",
                (category,)
            )

        if "electronics" in q:
            return make_result(
                "SELECT * FROM products WHERE category = ?",
                "Finds Electronics products.",
                ("Electronics",)
            )

        if "furniture" in q:
            return make_result(
                "SELECT * FROM products WHERE category = ?",
                "Finds Furniture products.",
                ("Furniture",)
            )

        match = re.search(
            r"(?:more than|greater than|above|over|costing more than)\s+(\d+(?:\.\d+)?)",
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
            r"(?:less than|below|under|costing less than)\s+(\d+(?:\.\d+)?)",
            q
        )
        if match:
            price = float(match.group(1))
            return make_result(
                "SELECT * FROM products WHERE price < ?",
                "Finds products below the specified price.",
                (price,)
            )

        if "product names and prices" in q:
            return make_result(
                "SELECT product_name, price FROM products",
                "Displays product names and prices."
            )

        if "cheapest product" in q or "least expensive product" in q:
            return make_result("""
                SELECT *
                FROM products
                ORDER BY price ASC
                LIMIT 1
            """, "Finds the cheapest product.")

        if has_any(q, ["how many", "count", "number of products"]):
            return make_result(
                "SELECT COUNT(*) AS total_products FROM products",
                "Counts products."
            )

        if has_any(q, [
            "show all", "all products", "list all",
            "display all", "find products"
        ]):
            return make_result(
                "SELECT * FROM products",
                "Displays product records."
            )

    # ----- Order queries -----

    if "order" in q:
        date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
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
            r"quantity\s+(?:is\s+)?between\s+(\d+)\s+and\s+(\d+)", q
        )
        if match:
            low, high = map(int, match.groups())
            return make_result(
                "SELECT * FROM orders WHERE quantity BETWEEN ? AND ?",
                f"Finds orders with quantity between {low} and {high}.",
                (low, high)
            )

        match = re.search(
            r"quantity\s+(?:is\s+)?(greater than|more than|above|over|less than|below|under|equal to|equals?)\s+(\d+)",
            q
        )
        if match:
            phrase, number = match.groups()
            operators = {
                "greater than": ">", "more than": ">", "above": ">", "over": ">",
                "less than": "<", "below": "<", "under": "<",
                "equal to": "=", "equal": "=", "equals": "="
            }
            operator = operators[phrase]
            number = int(number)

            return make_result(
                f"SELECT * FROM orders WHERE quantity {operator} ?",
                f"Filters orders by quantity {phrase} {number}.",
                (number,)
            )

        if has_any(q, ["most recent", "recent orders"]):
            limit = extract_limit(q, 10)
            return make_result(
                f"SELECT * FROM orders ORDER BY order_date DESC LIMIT {limit}",
                f"Displays the {limit} most recent orders."
            )

        if "sorted by date" in q or "sort by date" in q:
            direction = (
                "DESC" if has_any(q, ["descending", "newest first"]) else "ASC"
            )
            return make_result(
                f"SELECT * FROM orders ORDER BY order_date {direction}",
                f"Sorts orders by date {direction}."
            )

        if has_any(q, ["how many", "count", "number of orders"]):
            return make_result(
                "SELECT COUNT(*) AS total_orders FROM orders",
                "Counts orders."
            )

        if has_any(q, ["show all", "all orders", "list all", "display all"]):
            return make_result(
                "SELECT * FROM orders",
                "Displays order records."
            )

    return None


# ============================================================
# SQL VALIDATION
# ============================================================

FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "REPLACE", "ATTACH", "DETACH", "VACUUM", "REINDEX", "PRAGMA"
}


def remove_sql_string_literals(sql):
    return re.sub(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"",
        " ",
        sql
    )


def validate_sql(sql):
    if not isinstance(sql, str) or not sql.strip():
        return False, "Empty SQL query."

    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        return False, "Empty SQL query."

    without_strings = remove_sql_string_literals(cleaned)

    if ";" in without_strings:
        return False, "Multiple SQL statements are not allowed."

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.I):
        return False, "Only SELECT queries are allowed."

    if re.search(r"--|/\*|\*/", without_strings):
        return False, "SQL comments are not allowed."

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", without_strings, re.I):
            return False, f"Forbidden SQL keyword: {keyword}"

    return True, "SQL validation successful."


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

        columns = [item[0] for item in cursor.description]
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

    prompt = f"""
Fix the SQLite query using this exact schema.

SCHEMA:
{schema_as_text()}

QUESTION:
{question}

INCORRECT SQL:
{incorrect_sql}

ERROR:
{error_message}

Rules:
- Use only existing tables and columns.
- Return one SELECT statement only.
- Return SQL only.
- Do not include explanations.

CORRECTED SQL:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return repair_common_table_names(clean_generated_sql(generated))


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
        if not allow_correction or tokenizer is None or model is None:
            raise

        corrected_sql = correct_sql(
            question,
            sql,
            str(first_error),
            tokenizer,
            model,
            device
        )

        # Corrected SQL is always checked before execution.
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
    rule = context_aware_query(question)
    params = rule.get("params") if rule else None

    if mode == "Rules only":
        if rule is None:
            raise ValueError("No matching rule found.")

        sql, df, truncated, corrected = execute_with_optional_correction(
            question, rule["sql"], params,
            tokenizer, model, device,
            use_validation, False
        )

        return {
            "sql": sql,
            "dataframe": df,
            "method": "Rules",
            "explanation": rule["explanation"],
            "correction_used": corrected,
            "truncated": truncated
        }

    if mode == "Model only":
        if tokenizer is None or model is None:
            tokenizer, model, device = load_model()

        sql = generate_sql(question, tokenizer, model, device)
        sql, df, truncated, corrected = execute_with_optional_correction(
            question, sql, None,
            tokenizer, model, device,
            use_validation, allow_correction
        )

        return {
            "sql": sql,
            "dataframe": df,
            "method": "FLAN-T5",
            "explanation": "SQL generated by FLAN-T5.",
            "correction_used": corrected,
            "truncated": truncated
        }

    if mode == "Hybrid":
        if rule is not None:
            try:
                sql, df, truncated, _ = execute_with_optional_correction(
                    question, rule["sql"], params,
                    tokenizer, model, device,
                    use_validation, False
                )

                return {
                    "sql": sql,
                    "dataframe": df,
                    "method": "Rules",
                    "explanation": rule["explanation"],
                    "correction_used": False,
                    "truncated": truncated
                }

            except Exception as error:
                logger.warning("Rule execution failed: %s", error)

        if tokenizer is None or model is None:
            tokenizer, model, device = load_model()

        sql = generate_sql(question, tokenizer, model, device)
        sql, df, truncated, corrected = execute_with_optional_correction(
            question, sql, None,
            tokenizer, model, device,
            use_validation, allow_correction
        )

        return {
            "sql": sql,
            "dataframe": df,
            "method": "FLAN-T5 fallback",
            "explanation": "The model was used because no usable rule matched.",
            "correction_used": corrected,
            "truncated": truncated
        }

    raise ValueError(f"Unknown execution mode: {mode}")


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
    raise TypeError(f"Unsupported parameter type: {type(value).__name__}")


def generate_rule_sql(question):
    rule = context_aware_query(question)
    if rule is None:
        return None

    sql = rule["sql"]
    for value in rule.get("params", ()):
        if "?" not in sql:
            raise ValueError("Missing SQL parameter placeholder.")
        sql = sql.replace("?", sql_literal(value), 1)

    if "?" in sql:
        raise ValueError("Unfilled SQL parameter placeholder.")

    return sql


def generate_model_sql(question, tokenizer, model, device):
    return generate_sql(question, tokenizer, model, device)


# ============================================================
# VISUALIZATION
# ============================================================

def show_visualization(dataframe):
    if dataframe.empty or len(dataframe.columns) < 2:
        return

    numeric = dataframe.select_dtypes(include="number").columns.tolist()
    text_columns = [
        column for column in dataframe.columns
        if column not in numeric
    ]

    if not numeric or not text_columns:
        return

    st.subheader("Data Visualization")

    chart_type = st.selectbox("Chart type", ["Bar chart", "Line chart"])
    x_column = st.selectbox("Category", text_columns)
    y_column = st.selectbox("Numerical value", numeric)

    chart_data = dataframe[[x_column, y_column]].dropna()
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
        "The system generates SQL and displays the results."
    )

    try:
        stats = get_database_stats()
        st.sidebar.header("Database Information")
        st.sidebar.success("Database connected")
        st.sidebar.write(f"Database: {DB_PATH.name}")
        st.sidebar.metric("Total records", f"{sum(stats.values()):,}")

        for table, count in stats.items():
            st.sidebar.write(f"**{table}:** {count:,} records")

    except Exception as error:
        st.error(f"Database connection failed: {error}")
        st.stop()

    with st.expander("View Database Schema"):
        for table, columns in get_schema().items():
            st.markdown(f"**Table: {table}**")
            st.dataframe(
                pd.DataFrame(columns),
                use_container_width=True,
                hide_index=True
            )

    mode = st.sidebar.selectbox(
        "Execution method",
        ["Hybrid", "Rules only", "Model only"]
    )

    st.sidebar.caption(
        "Hybrid tries rules first and uses FLAN-T5 when no usable rule matches."
    )
    st.sidebar.divider()
    st.sidebar.write(f"Model: `{MODEL_ID}`")
    st.sidebar.write("Database: SQLite")
    st.sidebar.write("Query type: Read-only")

    examples = [
        "Show all customers",
        "Show all products",
        "Show all orders",
        "Show customer names and cities",
        "Find customers from Bengaluru",
        "Find customers older than 40",
        "Find customers younger than 30",
        "Find customers whose age is between 30 and 50",
        "Find customers whose age is exactly 25",
        "Find the average customer age",
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
        "Find orders with quantity equal to 4",
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
        "Calculate the total revenue from all products ordered"
    ]

    st.subheader("Ask Your Database")

    selected = st.selectbox(
        "Try an example question",
        ["Choose an example"] + examples
    )

    question = st.text_area(
        "Enter your natural language question",
        value=selected if selected != "Choose an example" else "",
        placeholder="Example: Count customers in each city"
    )

    if st.button("Run Query", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        tokenizer = model = device = None

        if mode == "Model only":
            try:
                with st.spinner("Loading FLAN-T5 model..."):
                    tokenizer, model, device = load_model()
                st.success(f"Model loaded on {device.upper()}.")
            except Exception as error:
                st.error(f"Model loading failed: {error}")
                return

        try:
            with st.spinner("Processing your question..."):
                result = run_ai_query(
                    question,
                    mode,
                    tokenizer,
                    model,
                    device
                )

            st.success("Query executed successfully.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Execution method", result["method"])
            col2.metric("Returned rows", len(result["dataframe"]))
            col3.metric(
                "Correction used",
                "Yes" if result["correction_used"] else "No"
            )

            st.subheader("Generated SQL")
            st.code(result["sql"], language="sql")

            st.subheader("Explanation")
            st.write(result["explanation"])

            st.subheader("Query Results")
            st.dataframe(
                result["dataframe"],
                use_container_width=True,
                hide_index=True
            )

            if result["truncated"]:
                st.warning(
                    f"Only the first {MAX_RESULT_ROWS} rows are displayed."
                )

            show_visualization(result["dataframe"])

            csv_data = result["dataframe"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                data=csv_data,
                file_name="query_results.csv",
                mime="text/csv"
            )

        except Exception as error:
            import traceback
            st.error(f"Query execution failed: {error}")
            with st.expander("Technical error details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
