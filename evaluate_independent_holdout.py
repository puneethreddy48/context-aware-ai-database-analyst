
import time
from pathlib import Path

import pandas as pd
import evaluation as ev


# ============================================================
# FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = BASE_DIR / "independent_holdout_questions.csv"
GOLD_FILE = BASE_DIR / "independent_holdout_gold.csv"

DETAILS_FILE = BASE_DIR / "independent_holdout_results.csv"
SUMMARY_FILE = BASE_DIR / "independent_holdout_summary.csv"


# ============================================================
# LOAD AND CHECK DATASETS
# ============================================================

def load_datasets():

    questions = pd.read_csv(QUESTIONS_FILE)
    gold = pd.read_csv(GOLD_FILE)

    required_questions = {"question_id", "question"}
    required_gold = {"question_id", "gold_sql"}

    if not required_questions.issubset(questions.columns):
        raise ValueError("Questions CSV has incorrect columns.")

    if not required_gold.issubset(gold.columns):
        raise ValueError("Gold CSV has incorrect columns.")

    if len(questions) != 25 or len(gold) != 25:
        raise ValueError("Expected exactly 25 questions and 25 gold SQL queries.")

    if questions["question_id"].duplicated().any():
        raise ValueError("Duplicate question IDs found.")

    if gold["question_id"].duplicated().any():
        raise ValueError("Duplicate gold SQL IDs found.")

    if set(questions["question_id"]) != set(gold["question_id"]):
        raise ValueError("Question IDs and gold SQL IDs do not match.")

    dataset = questions.merge(
        gold,
        on="question_id",
        how="inner",
        validate="one_to_one"
    )

    return dataset


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    start_time = time.time()

    print("=" * 65)
    print("NEW 25-QUESTION EVALUATION")
    print("=" * 65)

    if not Path(ev.DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {ev.DB_PATH}")

    dataset = load_datasets()

    print("\nQuestions loaded:", len(dataset))
    print("Question IDs matched:", True)

    # --------------------------------------------------------
    # VERIFY ALL GOLD SQL QUERIES BEFORE MODEL EVALUATION
    # --------------------------------------------------------

    print("\nChecking gold SQL queries...")

    for _, row in dataset.iterrows():

        result = ev.execute_for_comparison(row["gold_sql"])

        if result is None:
            raise RuntimeError(
                f"Gold SQL failed for {row['question_id']}"
            )

    print("All 25 gold SQL queries executed successfully.")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading FLAN-T5 model...")

    load_model_function = getattr(ev.load_model, "__wrapped__", ev.load_model)
    tokenizer, model, device = load_model_function()

    print("Model loaded on:", device)

    # --------------------------------------------------------
    # EVALUATE ALL THREE MODES
    # --------------------------------------------------------

    records = []

    for index, row in dataset.iterrows():

        question_id = row["question_id"]
        question = row["question"]
        gold_sql = row["gold_sql"]

        print(
            f"\n[{index + 1}/25] {question_id}: {question}"
        )

        # Rules-only
        rule_sql = ev.safe_rule_sql(question)

        # Model-only
        model_sql = ev.safe_model_sql(
            question,
            tokenizer,
            model,
            device
        )

        # Hybrid
        hybrid_sql, hybrid_path, hybrid_error = (
            ev.select_hybrid_sql(
                rule_sql,
                model_sql
            )
        )

        # Compare execution results with gold SQL
        rule_correct = ev.results_equal(
            rule_sql,
            gold_sql
        )

        model_correct = ev.results_equal(
            model_sql,
            gold_sql
        )

        hybrid_correct = ev.results_equal(
            hybrid_sql,
            gold_sql
        )

        # Execution status
        rule_execution, _, rule_error = ev.try_execute(rule_sql)
        model_execution, _, model_error = ev.try_execute(model_sql)
        hybrid_execution, _, hybrid_execution_error = (
            ev.try_execute(hybrid_sql)
        )

        print("Rules:", "CORRECT" if rule_correct else "INCORRECT")
        print("Model:", "CORRECT" if model_correct else "INCORRECT")
        print("Hybrid:", "CORRECT" if hybrid_correct else "INCORRECT")
        print("Hybrid path:", hybrid_path)

        records.append({
            "question_id": question_id,
            "question": question,
            "gold_sql": gold_sql,

            "rule_sql": rule_sql,
            "rule_execution_success": rule_execution,
            "rules_correct": rule_correct,
            "rule_error": rule_error,

            "model_sql": model_sql,
            "model_execution_success": model_execution,
            "model_correct": model_correct,
            "model_error": model_error,

            "hybrid_sql": hybrid_sql,
            "hybrid_selected_path": hybrid_path,
            "hybrid_correct": hybrid_correct,
            "hybrid_execution_success": hybrid_execution,
            "hybrid_error": hybrid_execution_error,
            "hybrid_rule_error": hybrid_error
        })

    # --------------------------------------------------------
    # SAVE DETAILED RESULTS
    # --------------------------------------------------------

    details = pd.DataFrame(records)

    details.to_csv(
        DETAILS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # CREATE SUMMARY
    # --------------------------------------------------------

    methods = {
        "Rules-only": "rules_correct",
        "Model-only": "model_correct",
        "Hybrid": "hybrid_correct"
    }

    summary = []

    for method, column in methods.items():

        correct = int(details[column].sum())
        total = len(details)

        summary.append({
            "method": method,
            "correct": correct,
            "incorrect": total - correct,
            "total": total,
            "accuracy_percent": round(
                correct / total * 100,
                2
            )
        })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("NEW HOLDOUT EVALUATION SUMMARY")
    print("=" * 65)

    print(summary_df.to_string(index=False))

    print("\nHYBRID PATH COUNTS")

    print(
        details["hybrid_selected_path"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nRESULT FILES:")
    print(DETAILS_FILE.resolve())
    print(SUMMARY_FILE.resolve())

    print(
        "\nTotal time:",
        round(time.time() - start_time, 2),
        "seconds"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
