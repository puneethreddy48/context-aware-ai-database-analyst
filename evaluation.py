import time
from pathlib import Path
from collections import Counter

import pandas as pd

from app import (
    DB_PATH,
    load_model,
    generate_rule_sql,
    generate_model_sql,
    correct_sql,
    execute_sql,
    validate_sql,
    get_database_stats,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent

DETAILS_FILE = OUTPUT_DIR / "evaluation_details.csv"
SUMMARY_FILE = OUTPUT_DIR / "evaluation_summary.csv"
CATEGORY_FILE = OUTPUT_DIR / "evaluation_category_summary.csv"
CORRECTION_FILE = OUTPUT_DIR / "correction_ablation.csv"
SECURITY_FILE = OUTPUT_DIR / "security_test_results.csv"

RUN_CORRECTION_ABLATION = True


# ============================================================
# BENCHMARK DATASET: 40 QUESTIONS
# ============================================================

TEST_CASES = [
    ("selection", "Show all customers.",
     "SELECT * FROM customers"),
    ("selection", "Show all products.",
     "SELECT * FROM products"),
    ("selection", "Show all orders.",
     "SELECT * FROM orders"),

    ("projection", "Show customer names and cities.",
     "SELECT customer_name, city FROM customers"),
    ("projection", "Show product names and prices.",
     "SELECT product_name, price FROM products"),

    ("filtering", "Find customers from Bengaluru.",
     "SELECT * FROM customers WHERE city = 'Bengaluru'"),
    ("filtering", "Find customers from Mumbai.",
     "SELECT * FROM customers WHERE city = 'Mumbai'"),
    ("filtering", "Find customers from Chennai.",
     "SELECT * FROM customers WHERE city = 'Chennai'"),
    ("filtering", "Find customers older than 40.",
     "SELECT * FROM customers WHERE age > 40"),
    ("filtering", "Find customers younger than 30.",
     "SELECT * FROM customers WHERE age < 30"),
    ("filtering", "Find products costing more than 10000.",
     "SELECT * FROM products WHERE price > 10000"),
    ("filtering", "Find products costing less than 5000.",
     "SELECT * FROM products WHERE price < 5000"),
    ("filtering", "Find orders with quantity greater than 5.",
     "SELECT * FROM orders WHERE quantity > 5"),
    ("filtering", "Find orders with quantity less than 3.",
     "SELECT * FROM orders WHERE quantity < 3"),
    ("filtering", "Find customers from Hyderabad.",
     "SELECT * FROM customers WHERE city = 'Hyderabad'"),
    ("filtering", "Find customers from Pune.",
     "SELECT * FROM customers WHERE city = 'Pune'"),
    ("filtering", "Find customers from Mysuru.",
     "SELECT * FROM customers WHERE city = 'Mysuru'"),

    ("aggregation", "Count all customers.",
     "SELECT COUNT(*) FROM customers"),
    ("aggregation", "Count all products.",
     "SELECT COUNT(*) FROM products"),
    ("aggregation", "Find the average customer age.",
     "SELECT AVG(age) FROM customers"),
    ("aggregation", "Find the maximum product price.",
     "SELECT MAX(price) FROM products"),
    ("aggregation", "Find the minimum product price.",
     "SELECT MIN(price) FROM products"),
    ("aggregation", "Find the total quantity ordered.",
     "SELECT SUM(quantity) FROM orders"),
    ("aggregation", "Find the average product price.",
     "SELECT AVG(price) FROM products"),
    ("aggregation", "Find the total number of distinct cities.",
     "SELECT COUNT(DISTINCT city) FROM customers"),
    ("aggregation", "Find the maximum order quantity.",
     "SELECT MAX(quantity) FROM orders"),
    ("aggregation", "Find the minimum order quantity.",
     "SELECT MIN(quantity) FROM orders"),

    ("group_by", "Count customers in each city.",
     "SELECT city, COUNT(*) AS total_customers FROM customers GROUP BY city"),
    ("group_by", "Find the number of products in each category.",
     "SELECT category, COUNT(*) AS total_products FROM products GROUP BY category"),

    ("order_by", "Show customers sorted by age in ascending order.",
     "SELECT * FROM customers ORDER BY age ASC"),
    ("order_by", "Show customers sorted by age in descending order.",
     "SELECT * FROM customers ORDER BY age DESC"),
    ("order_by", "Show products sorted by price in ascending order.",
     "SELECT * FROM products ORDER BY price ASC"),
    ("order_by", "Show products sorted by price in descending order.",
     "SELECT * FROM products ORDER BY price DESC"),

    ("join", "Show customer names with their orders.",
     """SELECT c.customer_name, o.order_id
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id"""),
    ("join", "Show product names with their order quantities.",
     """SELECT p.product_name, o.quantity
        FROM orders o
        JOIN products p ON o.product_id = p.product_id"""),
    ("join", "Show customer names and their order dates.",
     """SELECT c.customer_name, o.order_date
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id"""),

    ("join_group_by", "Find total quantity ordered for each product.",
     """SELECT p.product_name, SUM(o.quantity) AS total_quantity
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        GROUP BY p.product_id, p.product_name"""),
    ("join_group_by", "Count the number of orders for each customer.",
     """SELECT c.customer_name, COUNT(o.order_id) AS total_orders
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.customer_name"""),

    ("order_by_limit", "Show the top 5 most expensive products.",
     "SELECT * FROM products ORDER BY price DESC LIMIT 5"),
    ("order_by_limit", "Show the 5 youngest customers.",
     "SELECT * FROM customers ORDER BY age ASC LIMIT 5"),
]


# ============================================================
# HOLDOUT DATASET: 20 QUESTIONS
# ============================================================

HOLDOUT_CASES = [
    ("holdout", "How many customers are there?",
     "SELECT COUNT(*) FROM customers"),
    ("holdout", "List the cities where customers live.",
     "SELECT DISTINCT city FROM customers"),
    ("holdout", "Find customers whose age is exactly 25.",
     "SELECT * FROM customers WHERE age = 25"),
    ("holdout", "Find products in the Electronics category.",
     "SELECT * FROM products WHERE category = 'Electronics'"),
    ("holdout", "Find orders placed after 2025-06-01.",
     "SELECT * FROM orders WHERE order_date > '2025-06-01'"),
    ("holdout", "Calculate the total revenue from all products ordered.",
     """SELECT SUM(p.price * o.quantity)
        FROM orders o
        JOIN products p ON o.product_id = p.product_id"""),
    ("holdout", "Find the average order quantity for each customer.",
     """SELECT c.customer_id, AVG(o.quantity)
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id"""),
    ("holdout", "Show the 10 most recent orders.",
     "SELECT * FROM orders ORDER BY order_date DESC LIMIT 10"),
    ("holdout", "Find the number of customers in each city, highest first.",
     """SELECT city, COUNT(*) AS total_customers
        FROM customers
        GROUP BY city
        ORDER BY total_customers DESC"""),
    ("holdout", "Find products priced between 1000 and 10000.",
     "SELECT * FROM products WHERE price BETWEEN 1000 AND 10000"),
    ("holdout", "Find orders with quantity equal to 4.",
     "SELECT * FROM orders WHERE quantity = 4"),
    ("holdout", "Find the total number of orders.",
     "SELECT COUNT(*) FROM orders"),
    ("holdout", "Find the average age of customers in Bengaluru.",
     "SELECT AVG(age) FROM customers WHERE city = 'Bengaluru'"),
    ("holdout", "Find the most expensive product in each category.",
     "SELECT category, MAX(price) FROM products GROUP BY category"),
    ("holdout", "Show the 10 products with the lowest prices.",
     "SELECT * FROM products ORDER BY price ASC LIMIT 10"),
    ("holdout", "Find the total quantity ordered in each city.",
     """SELECT c.city, SUM(o.quantity)
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.city"""),
    ("holdout", "Find customers whose age is between 30 and 50.",
     "SELECT * FROM customers WHERE age BETWEEN 30 AND 50"),
    ("holdout", "Count the number of orders for each product.",
     """SELECT p.product_name, COUNT(o.order_id)
        FROM products p
        JOIN orders o ON p.product_id = o.product_id
        GROUP BY p.product_id, p.product_name"""),
    ("holdout", "Show the 10 customers with the highest ages.",
     "SELECT * FROM customers ORDER BY age DESC LIMIT 10"),
    ("holdout", "Find the total sales amount for each product category.",
     """SELECT p.category, SUM(p.price * o.quantity)
        FROM products p
        JOIN orders o ON p.product_id = o.product_id
        GROUP BY p.category"""),
]


# ============================================================
# ADDITIONAL LLM-GENERATED CHALLENGE DATASET: 20 QUESTIONS
# ============================================================

CHALLENGE_CASES = [
    (
        "challenge",
        "Find customers who have placed more than 2 orders.",
        """SELECT c.customer_id, c.customer_name, COUNT(o.order_id) AS total_orders
           FROM customers c
           JOIN orders o ON c.customer_id = o.customer_id
           GROUP BY c.customer_id, c.customer_name
           HAVING COUNT(o.order_id) > 2"""
    ),
    (
        "challenge",
        "Find customers who have never placed an order.",
        """SELECT c.customer_id, c.customer_name
           FROM customers c
           LEFT JOIN orders o ON c.customer_id = o.customer_id
           WHERE o.order_id IS NULL"""
    ),
    (
        "challenge",
        "Calculate the total amount spent by each customer.",
        """SELECT c.customer_id, c.customer_name,
                  SUM(p.price * o.quantity) AS total_spent
           FROM customers c
           JOIN orders o ON c.customer_id = o.customer_id
           JOIN products p ON o.product_id = p.product_id
           GROUP BY c.customer_id, c.customer_name"""
    ),
    (
        "challenge",
        "Show the 5 customers who spent the most money.",
        """SELECT c.customer_id, c.customer_name,
                  SUM(p.price * o.quantity) AS total_spent
           FROM customers c
           JOIN orders o ON c.customer_id = o.customer_id
           JOIN products p ON o.product_id = p.product_id
           GROUP BY c.customer_id, c.customer_name
           ORDER BY total_spent DESC
           LIMIT 5"""
    ),
    (
        "challenge",
        "Find the most purchased product based on total quantity.",
        """SELECT p.product_name, SUM(o.quantity) AS total_quantity
           FROM products p
           JOIN orders o ON p.product_id = o.product_id
           GROUP BY p.product_id, p.product_name
           ORDER BY total_quantity DESC
           LIMIT 1"""
    ),
    (
        "challenge",
        "Calculate the total purchase value of orders placed in July 2025.",
        """SELECT SUM(p.price * o.quantity) AS total_purchase_value
           FROM orders o
           JOIN products p ON o.product_id = p.product_id
           WHERE o.order_date >= '2025-07-01'
             AND o.order_date < '2025-08-01'"""
    ),
    (
        "challenge",
        "Find the city with the highest average customer age.",
        """SELECT city, AVG(age) AS average_age
           FROM customers
           GROUP BY city
           ORDER BY average_age DESC
           LIMIT 1"""
    ),
    (
        "challenge",
        "Calculate the average product price in each category.",
        """SELECT category, AVG(price) AS average_price
           FROM products
           GROUP BY category"""
    ),
    (
        "challenge",
        "Count the distinct customers who purchased each product.",
        """SELECT p.product_name,
                  COUNT(DISTINCT o.customer_id) AS unique_customers
           FROM products p
           JOIN orders o ON p.product_id = o.product_id
           GROUP BY p.product_id, p.product_name"""
    ),
    (
        "challenge",
        "Find customers whose total spending is greater than 50000.",
        """SELECT c.customer_id, c.customer_name,
                  SUM(p.price * o.quantity) AS total_spent
           FROM customers c
           JOIN orders o ON c.customer_id = o.customer_id
           JOIN products p ON o.product_id = p.product_id
           GROUP BY c.customer_id, c.customer_name
           HAVING SUM(p.price * o.quantity) > 50000"""
    ),
    (
        "challenge",
        "Show the three most frequently ordered product categories.",
        """SELECT p.category, COUNT(o.order_id) AS total_orders
           FROM products p
           JOIN orders o ON p.product_id = o.product_id
           GROUP BY p.category
           ORDER BY total_orders DESC
           LIMIT 3"""
    ),
    (
        "challenge",
        "Find customers who placed orders during December 2025.",
        """SELECT DISTINCT c.customer_id, c.customer_name
           FROM customers c
           JOIN orders o ON c.customer_id = o.customer_id
           WHERE o.order_date >= '2025-12-01'
             AND o.order_date < '2026-01-01'"""
    ),
    (
        "challenge",
        "Find the youngest customer in each city.",
        """SELECT c.customer_id, c.customer_name, c.city, c.age
           FROM customers c
           WHERE c.age = (
               SELECT MIN(c2.age)
               FROM customers c2
               WHERE c2.city = c.city
           )"""
    ),
    (
        "challenge",
        "Find products that have never been ordered.",
        """SELECT p.product_id, p.product_name
           FROM products p
           LEFT JOIN orders o ON p.product_id = o.product_id
           WHERE o.order_id IS NULL"""
    ),
    (
        "challenge",
        "Count the number of orders placed in each month of 2025.",
        """SELECT strftime('%Y-%m', order_date) AS order_month,
                  COUNT(*) AS total_orders
           FROM orders
           WHERE order_date >= '2025-01-01'
             AND order_date < '2026-01-01'
           GROUP BY strftime('%Y-%m', order_date)
           ORDER BY order_month"""
    ),
    (
        "challenge",
        "Find the largest quantity in a single order.",
        """SELECT MAX(quantity) AS largest_quantity
           FROM orders"""
    ),
    (
        "challenge",
        "Find customers who purchased products in the Electronics category.",
        """SELECT DISTINCT c.customer_id, c.customer_name
           FROM customers c
           JOIN orders o ON c.customer_id = o.customer_id
           JOIN products p ON o.product_id = p.product_id
           WHERE p.category = 'Electronics'"""
    ),
    (
        "challenge",
        "Find the product category with the highest total sales value.",
        """SELECT p.category,
                  SUM(p.price * o.quantity) AS total_sales
           FROM products p
           JOIN orders o ON p.product_id = o.product_id
           GROUP BY p.category
           ORDER BY total_sales DESC
           LIMIT 1"""
    ),
    (
        "challenge",
        "Find customers whose total ordered quantity is above the average total quantity per customer.",
        """SELECT c.customer_id, c.customer_name,
                  SUM(o.quantity) AS total_quantity
           FROM customers c
           JOIN orders o ON c.customer_id = o.customer_id
           GROUP BY c.customer_id, c.customer_name
           HAVING SUM(o.quantity) > (
               SELECT AVG(customer_total)
               FROM (
                   SELECT SUM(quantity) AS customer_total
                   FROM orders
                   GROUP BY customer_id
               )
           )"""
    ),
    (
        "challenge",
        "Find the most expensive product in each category.",
        """SELECT p.category, p.product_name, p.price
           FROM products p
           WHERE p.price = (
               SELECT MAX(p2.price)
               FROM products p2
               WHERE p2.category = p.category
           )"""
    ),
]


# ============================================================
# RESULT COMPARISON
# ============================================================

def normalize_value(value):
    if pd.isna(value):
        return ("NULL",)

    if isinstance(value, (int, float)):
        return ("NUMBER", round(float(value), 8))

    return ("TEXT", str(value).strip())


def dataframe_rows(dataframe):
    return [
        tuple(
            normalize_value(value)
            for value in row
        )
        for row in dataframe.itertuples(
            index=False,
            name=None
        )
    ]


def execute_for_comparison(sql, use_validation=True):
    if not isinstance(sql, str) or not sql.strip():
        return None

    try:
        dataframe, _ = execute_sql(
            sql,
            params=None,
            use_validation=use_validation
        )

        return dataframe

    except Exception:
        return None


def results_equal(
    predicted_sql,
    gold_sql,
    use_validation=True
):
    predicted = execute_for_comparison(
        predicted_sql,
        use_validation
    )

    expected = execute_for_comparison(
        gold_sql,
        use_validation
    )

    if predicted is None or expected is None:
        return False

    if predicted.shape != expected.shape:
        return False

    predicted_rows = dataframe_rows(predicted)
    expected_rows = dataframe_rows(expected)

    # Preserve order when the gold SQL explicitly requests ordering.
    if re_has_order_by(gold_sql):
        return predicted_rows == expected_rows

    # Otherwise compare rows without requiring the same order.
    return Counter(predicted_rows) == Counter(expected_rows)


def re_has_order_by(sql):
    import re

    return bool(
        re.search(r"\bORDER\s+BY\b", sql, re.I)
    )


# ============================================================
# SAFE GENERATION HELPERS
# ============================================================

def safe_rule_sql(question):
    try:
        return generate_rule_sql(question)
    except Exception:
        return None


def safe_model_sql(question, tokenizer, model, device):
    try:
        return generate_model_sql(
            question,
            tokenizer,
            model,
            device
        )
    except Exception as error:
        print("Model generation error:", error)
        return None


def safe_validate(sql):
    try:
        return validate_sql(sql)
    except Exception as error:
        return False, str(error)


def try_execute(sql):
    """Execute SQL and return status, dataframe and error."""

    if not isinstance(sql, str) or not sql.strip():
        return False, None, "No SQL generated."

    try:
        dataframe, _ = execute_sql(
            sql,
            use_validation=True
        )

        return True, dataframe, None

    except Exception as error:
        return False, None, str(error)


# ============================================================
# ACTUAL HYBRID SELECTION
# ============================================================

def select_hybrid_sql(rule_sql, model_sql):
    """
    Follow the same decision as app.py:

    1. If a rule exists and executes successfully, use it.
    2. Otherwise, use the model-generated SQL.
    """

    if rule_sql:
        rule_success, _, rule_error = try_execute(rule_sql)

        if rule_success:
            return rule_sql, "Rules", None

    else:
        rule_error = "No matching rule."

    return model_sql, "FLAN-T5 fallback", rule_error


# ============================================================
# SECURITY TESTS
# ============================================================

def run_security_tests():
    print("\nRunning read-only security tests...")

    attacks = [
        ("DELETE attack", "DELETE FROM orders"),
        ("DROP attack", "DROP TABLE customers"),
        ("UPDATE attack", "UPDATE customers SET city='Test'"),
        (
            "INSERT attack",
            "INSERT INTO customers(customer_name) VALUES('Test')"
        ),
        (
            "Multiple statements",
            "SELECT * FROM customers; DROP TABLE customers"
        ),
        (
            "Line comment injection",
            "SELECT * FROM customers; -- hidden command\nDROP TABLE customers"
        ),
        (
            "Block comment injection",
            "SELECT * FROM customers /* hidden command */; DROP TABLE customers"
        ),
        (
            "PRAGMA attack",
            "PRAGMA table_info(customers)"
        ),
    ]

    results = []

    for name, attack_sql in attacks:
        valid, validation_message = safe_validate(attack_sql)

        try:
            execute_sql(
                attack_sql,
                use_validation=True
            )

            blocked_with_validation = False
            execution_message = "Executed"

        except Exception as error:
            blocked_with_validation = True
            execution_message = str(error)

        try:
            execute_sql(
                attack_sql,
                use_validation=False
            )

            blocked_without_validation = False
            authorizer_message = "Executed"

        except Exception as error:
            blocked_without_validation = True
            authorizer_message = str(error)

        results.append({
            "attack": name,
            "sql": attack_sql,
            "validation_passed": valid,
            "validation_message": validation_message,
            "blocked_with_validation": blocked_with_validation,
            "blocked_without_validation": blocked_without_validation,
            "security_test_passed": (
                blocked_with_validation
                and blocked_without_validation
            ),
            "execution_message": execution_message,
            "authorizer_message": authorizer_message
        })

    # Verify that the database tables still exist.
    try:
        stats_after = get_database_stats()

        database_intact = all(
            table in stats_after
            for table in [
                "customers",
                "products",
                "orders"
            ]
        )

    except Exception:
        stats_after = {}
        database_intact = False

    results.append({
        "attack": "Database integrity check",
        "sql": "Check required tables after security tests",
        "validation_passed": True,
        "validation_message": "Integrity check",
        "blocked_with_validation": database_intact,
        "blocked_without_validation": database_intact,
        "security_test_passed": database_intact,
        "execution_message": str(database_intact),
        "authorizer_message": str(stats_after)
    })

    security_df = pd.DataFrame(results)
    security_df.to_csv(SECURITY_FILE, index=False)

    print(
        security_df[
            ["attack", "security_test_passed"]
        ].to_string(index=False)
    )

    print(
        "\nSecurity tests passed:",
        int(security_df["security_test_passed"].sum()),
        "/",
        len(security_df)
    )

    return security_df


# ============================================================
# CORRECTION ABLATION
# ============================================================

def run_correction_ablation(
    details,
    tokenizer,
    model,
    device
):
    records = []
    corrected_cache = {}

    print("\nRunning correction ON/OFF ablation...")

    def get_corrected_sql(question, sql, error_message):
        cache_key = (question, sql)

        if cache_key not in corrected_cache:
            corrected_cache[cache_key] = correct_sql(
                question,
                sql,
                error_message,
                tokenizer,
                model,
                device
            )

        return corrected_cache[cache_key]

    for index, row in details.iterrows():

        question = row["question"]
        gold_sql = row["gold_sql"]
        model_sql = row["model_sql"]
        rule_sql = row["rule_sql"]

        print(
            f"[correction] {index + 1}/{len(details)}: {question}"
        )

        # ----------------------------------------------------
        # MODEL-ONLY
        # ----------------------------------------------------

        model_before = results_equal(
            model_sql,
            gold_sql
        )

        model_execution_success, _, model_initial_error = (
            try_execute(model_sql)
        )

        model_after_sql = model_sql
        model_correction_attempted = False
        model_correction_error = None
        model_corrected_execution_success = model_execution_success

        if (
            not model_execution_success
            and model_sql
            and RUN_CORRECTION_ABLATION
        ):
            model_correction_attempted = True

            try:
                model_after_sql = get_corrected_sql(
                    question,
                    model_sql,
                    model_initial_error
                )

                (
                    model_corrected_execution_success,
                    _,
                    correction_error
                ) = try_execute(model_after_sql)

                if not model_corrected_execution_success:
                    model_correction_error = correction_error

            except Exception as error:
                model_corrected_execution_success = False
                model_correction_error = str(error)

        model_after = results_equal(
            model_after_sql,
            gold_sql
        )

        # ----------------------------------------------------
        # HYBRID: USE THE ACTUAL SELECTED PATH
        # ----------------------------------------------------

        hybrid_before_sql, hybrid_path, rule_error = (
            select_hybrid_sql(
                rule_sql,
                model_sql
            )
        )

        hybrid_before_execution, _, hybrid_initial_error = (
            try_execute(hybrid_before_sql)
        )

        hybrid_before = results_equal(
            hybrid_before_sql,
            gold_sql
        )

        hybrid_after_sql = hybrid_before_sql
        hybrid_correction_attempted = False
        hybrid_correction_error = None
        hybrid_corrected_execution_success = hybrid_before_execution

        # App's hybrid mode does not correct a successful rule.
        # It only attempts correction for a failed model fallback.
        if (
            hybrid_path == "FLAN-T5 fallback"
            and not hybrid_before_execution
            and hybrid_before_sql
            and RUN_CORRECTION_ABLATION
        ):
            hybrid_correction_attempted = True

            try:
                hybrid_after_sql = get_corrected_sql(
                    question,
                    hybrid_before_sql,
                    hybrid_initial_error
                )

                (
                    hybrid_corrected_execution_success,
                    _,
                    correction_error
                ) = try_execute(hybrid_after_sql)

                if not hybrid_corrected_execution_success:
                    hybrid_correction_error = correction_error

            except Exception as error:
                hybrid_corrected_execution_success = False
                hybrid_correction_error = str(error)

        hybrid_after = results_equal(
            hybrid_after_sql,
            gold_sql
        )

        records.append({
            "split": row["split"],
            "question": question,
            "gold_sql": gold_sql,

            "model_sql": model_sql,
            "model_execution_success": model_execution_success,
            "model_initial_error": model_initial_error,
            "model_correct_without_correction": model_before,
            "model_correction_attempted": model_correction_attempted,
            "model_corrected_sql": (
                model_after_sql
                if model_correction_attempted else None
            ),
            "model_corrected_execution_success": (
                model_corrected_execution_success
            ),
            "model_correction_error": model_correction_error,
            "model_correct_with_correction": model_after,
            "model_improved": (
                not model_before and model_after
            ),

            "rule_sql": rule_sql,
            "rule_execution_success": (
                try_execute(rule_sql)[0]
                if rule_sql else False
            ),
            "rule_execution_error": rule_error,

            "hybrid_selected_path": hybrid_path,
            "hybrid_sql_without_correction": hybrid_before_sql,
            "hybrid_correct_without_correction": hybrid_before,
            "hybrid_correction_attempted": hybrid_correction_attempted,
            "hybrid_corrected_sql": (
                hybrid_after_sql
                if hybrid_correction_attempted else None
            ),
            "hybrid_corrected_execution_success": (
                hybrid_corrected_execution_success
            ),
            "hybrid_correction_error": hybrid_correction_error,
            "hybrid_correct_with_correction": hybrid_after,
            "hybrid_improved": (
                not hybrid_before and hybrid_after
            )
        })

    correction_df = pd.DataFrame(records)
    correction_df.to_csv(CORRECTION_FILE, index=False)

    print("\nCORRECTION ABLATION RESULTS")

    for label, before_col, after_col, attempt_col in [
        (
            "Model-only",
            "model_correct_without_correction",
            "model_correct_with_correction",
            "model_correction_attempted"
        ),
        (
            "Hybrid",
            "hybrid_correct_without_correction",
            "hybrid_correct_with_correction",
            "hybrid_correction_attempted"
        )
    ]:
        before = int(correction_df[before_col].sum())
        after = int(correction_df[after_col].sum())
        attempts = int(correction_df[attempt_col].sum())
        total = len(correction_df)

        print(f"\n{label}")
        print("Correct without correction:", before)
        print("Correct with correction:", after)
        print("Correction attempts:", attempts)

        print(
            "Accuracy without correction:",
            round(100 * before / total, 2),
            "%"
        )

        print(
            "Accuracy with correction:",
            round(100 * after / total, 2),
            "%"
        )

    return correction_df


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_split(
    cases,
    split_name,
    tokenizer,
    model,
    device
):
    records = []

    for index, (category, question, gold_sql) in enumerate(
        cases,
        1
    ):
        print(
            f"[{split_name}] {index}/{len(cases)}: {question}"
        )

        rule_sql = safe_rule_sql(question)

        model_sql = safe_model_sql(
            question,
            tokenizer,
            model,
            device
        )

        # The hybrid selection follows the actual application logic.
        hybrid_sql, hybrid_path, hybrid_rule_error = (
            select_hybrid_sql(
                rule_sql,
                model_sql
            )
        )

        rule_correct = results_equal(
            rule_sql,
            gold_sql
        )

        model_correct = results_equal(
            model_sql,
            gold_sql
        )

        hybrid_correct = results_equal(
            hybrid_sql,
            gold_sql
        )

        validation_on_correct = results_equal(
            hybrid_sql,
            gold_sql,
            use_validation=True
        )

        validation_off_correct = results_equal(
            hybrid_sql,
            gold_sql,
            use_validation=False
        )

        validation_passed, validation_message = (
            safe_validate(hybrid_sql)
            if hybrid_sql
            else (False, "No SQL generated.")
        )

        hybrid_execution_success, _, hybrid_execution_error = (
            try_execute(hybrid_sql)
        )

        records.append({
            "split": split_name,
            "category": category,
            "question": question,
            "gold_sql": gold_sql,

            "rule_sql": rule_sql,
            "model_sql": model_sql,
            "hybrid_sql": hybrid_sql,
            "hybrid_selected_path": hybrid_path,
            "hybrid_rule_error": hybrid_rule_error,
            "hybrid_execution_success": hybrid_execution_success,
            "hybrid_execution_error": hybrid_execution_error,

            "validation_passed": validation_passed,
            "validation_message": validation_message,

            "rules_only_correct": rule_correct,
            "model_only_correct": model_correct,
            "hybrid_correct": hybrid_correct,

            "validation_on_correct": validation_on_correct,
            "validation_off_correct": validation_off_correct
        })

    return pd.DataFrame(records)


# ============================================================
# SUMMARY TABLES
# ============================================================

def create_summary(details):
    methods = {
        "rules_only": "rules_only_correct",
        "model_only": "model_only_correct",
        "hybrid": "hybrid_correct",
        "validation_on": "validation_on_correct",
        "validation_off": "validation_off_correct"
    }

    rows = []

    for split_name, group in details.groupby("split"):
        for method, column in methods.items():
            correct = int(group[column].sum())
            total = len(group)

            rows.append({
                "split": split_name,
                "method": method,
                "correct": correct,
                "total": total,
                "accuracy_percent": round(
                    100 * correct / total,
                    2
                ) if total else 0
            })

    return pd.DataFrame(rows)


def create_category_summary(benchmark_df):
    return (
        benchmark_df.groupby("category")
        .agg(
            total_questions=("question", "count"),
            rules_correct=("rules_only_correct", "sum"),
            model_correct=("model_only_correct", "sum"),
            hybrid_correct=("hybrid_correct", "sum")
        )
        .reset_index()
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CONTEXT-AWARE AI DATABASE ANALYST - FULL EVALUATION")
    print("=" * 70)

    if not Path(DB_PATH).exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    print("\nDATABASE ROW COUNTS")
    print(get_database_stats())

    print("\nDATASET COUNTS")
    print("Benchmark questions:", len(TEST_CASES))
    print("Original holdout questions:", len(HOLDOUT_CASES))
    print("LLM-generated challenge questions:", len(CHALLENGE_CASES))
    print(
        "Total evaluation questions:",
        len(TEST_CASES) + len(HOLDOUT_CASES) + len(CHALLENGE_CASES)
    )

    start_time = time.time()

    print("\nLoading FLAN-T5 model...")

    tokenizer, model, device = load_model()

    print("Model loaded on:", device)

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    benchmark_df = run_split(
        TEST_CASES,
        "benchmark",
        tokenizer,
        model,
        device
    )

    # --------------------------------------------------------
    # ORIGINAL HOLDOUT
    # --------------------------------------------------------

    holdout_df = run_split(
        HOLDOUT_CASES,
        "holdout",
        tokenizer,
        model,
        device
    )

    # --------------------------------------------------------
    # ADDITIONAL CHALLENGE SET
    # --------------------------------------------------------

    challenge_df = run_split(
        CHALLENGE_CASES,
        "llm_generated_challenge",
        tokenizer,
        model,
        device
    )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    all_details = pd.concat(
        [
            benchmark_df,
            holdout_df,
            challenge_df
        ],
        ignore_index=True
    )

    summary_df = create_summary(all_details)

    category_df = create_category_summary(
        benchmark_df
    )

    all_details.to_csv(
        DETAILS_FILE,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    category_df.to_csv(
        CATEGORY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # CORRECTION ABLATION
    # --------------------------------------------------------

    correction_df = run_correction_ablation(
        all_details,
        tokenizer,
        model,
        device
    )

    # --------------------------------------------------------
    # SECURITY TESTS
    # --------------------------------------------------------

    security_df = run_security_tests()

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(summary_df.to_string(index=False))

    print("\nCATEGORY-WISE BENCHMARK RESULTS")

    print(category_df.to_string(index=False))

    print("\nHYBRID EXECUTION PATH COUNTS")

    print(
        all_details.groupby(
            ["split", "hybrid_selected_path"]
        ).size().to_string()
    )

    print("\nSECURITY TEST SUMMARY")

    print(
        security_df[
            ["attack", "security_test_passed"]
        ].to_string(index=False)
    )

    print(
        "\nTOTAL TIME:",
        round(elapsed, 2),
        "seconds"
    )

    print("\nFILES SAVED:")

    for file_path in [
        DETAILS_FILE,
        SUMMARY_FILE,
        CATEGORY_FILE,
        CORRECTION_FILE,
        SECURITY_FILE
    ]:
        print(file_path.resolve())

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()