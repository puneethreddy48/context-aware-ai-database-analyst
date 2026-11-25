# Context-Aware AI Database Analyst

**Review-2 MVP | Natural Language to SQL | AI-Powered Database Analysis**

## Project Overview

Context-Aware AI Database Analyst is an AI-powered application that allows users to interact with an SQLite database using natural language. Instead of manually writing SQL queries, users can ask questions in plain English and receive SQL queries and database results through an interactive Streamlit dashboard.

The system combines predefined context-aware SQL rules with a pretrained Text-to-SQL model. It supports three execution modes: **Hybrid, Rules Only, and Model Only**. Generated SQL queries are checked using read-only validation before execution.

## Key Features

- **Database Schema Inspection:** Automatically inspects SQLite tables and their structures.
- **Natural Language to SQL:** Uses the `alpecevit/flan-t5-base-text2sql` pretrained model.
- **Three Execution Modes:** Supports Hybrid, Rules Only, and Model Only.
- **Hybrid Query Processing:** Attempts predefined SQL rules first and uses model generation as a fallback.
- **Read-Only SQL Validation:** Restricts execution to permitted read-only SQL statements.
- **Error Feedback:** Includes a SQL correction mechanism using database execution feedback.
- **Interactive Dashboard:** Displays generated SQL queries and corresponding database results.
- **Synthetic Business Database:** Uses a reproducible SQLite dataset containing 25,500 records.
- **Evaluation Framework:** Includes benchmark and holdout question partitions and evaluation result files.

## System Architecture and Workflow

1. The user enters a question in natural language.
2. The application inspects the SQLite database schema.
3. The selected execution mode processes the question.
4. In Hybrid mode, predefined SQL rules are attempted first, followed by model fallback when required.
5. The generated SQL query passes through read-only validation.
6. Permitted queries are executed against the SQLite database.
7. The dashboard displays the generated SQL and query results.
8. If query execution fails, the application can attempt correction using execution-error feedback.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Database | SQLite |
| Text-to-SQL Model | `alpecevit/flan-t5-base-text2sql` |
| Model Platform | Hugging Face Transformers |
| SQL Processing | Python and SQLite |
| Evaluation | Pandas and custom evaluation scripts |

## Database Design

The project uses a synthetic business database generated using `generate_large_database.py`.

| Table | Number of Records |
|---|---:|
| Customers | 5,000 |
| Products | 500 |
| Orders | 20,000 |
| **Total** | **25,500** |

**Database file:** `data/large_business.db`

The database generation script uses a fixed random seed of **42** to support reproducibility.

To regenerate the database:

```bash
python generate_large_database.py
```

Regenerating the database may replace the existing database file.

## Pretrained Model Information

This project integrates the existing Hugging Face model:

`alpecevit/flan-t5-base-text2sql`

According to its model card, the pretrained checkpoint was fine-tuned using SParC, Spider, and CoSQL datasets.

The application supplies the natural-language question and database schema using the following prompt structure:

```text
transform question and schema to SQL query.
question: [USER QUESTION]
schema: [DATABASE SCHEMA]
```

**Important:** The model was not newly trained or fine-tuned as part of this project. The work focuses on application-level integration, context-aware SQL rule processing, query validation, and database analysis.

## Experimental Evaluation

The evaluation framework contains **60 manually prepared questions**, divided into two partitions:

| Evaluation Partition | Questions |
|---|---:|
| Benchmark | 40 |
| Holdout | 20 |
| **Total** | **60** |

### Evaluation Results

| Execution Mode | Benchmark (40) | Holdout (20) | Combined (60) |
|---|---:|---:|---:|
| Rules Only | 80% | 100% | 86.67% |
| Model Only | 37.5% | 15% | 30% |
| Hybrid | 97.5% | 100% | 98.33% |

The holdout partition was prepared within the same project evaluation process. Independent authorship and a formal rule-freezing procedure before holdout question creation were not established. Therefore, it should not be interpreted as a fully independent or blind evaluation.

These results represent the performance on the manually prepared evaluation questions and should not be interpreted as guaranteed performance on arbitrary databases or unseen question distributions.

### Evaluation and Reproducibility Files

- `evaluation.py` – Evaluation implementation.
- `evaluation_details.csv` – Contains all 60 questions, gold SQL queries, generated SQL queries, and correctness results.
- `evaluation_summary.csv` – Overall evaluation summary.
- `evaluation_category_summary.csv` – Category-wise evaluation results.
- `correction_ablation.csv` – Correction mechanism evaluation results.
- `security_test_results.csv` – SQL safeguard test results.

The application-level validation ablation did not change measured accuracy on the evaluated question partitions. Validation remains important for checking SQL structure and restricting query execution.

## Project Structure

```text
context-aware-ai-database-analyst/
│
├── .devcontainer/
├── data/
│   └── large_business.db
│
├── app.py
├── evaluation.py
├── generate_large_database.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── evaluation_details.csv
├── evaluation_summary.csv
├── evaluation_category_summary.csv
├── correction_ablation.csv
└── security_test_results.csv
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/puneethreddy48/context-aware-ai-database-analyst.git
```

### 2. Navigate to the project directory

```bash
cd context-aware-ai-database-analyst
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

The pretrained model may take additional time to download and load during the first execution.

## Reproducing the Evaluation

Ensure that the dependencies are installed and the database is available at `data/large_business.db`.

Run:

```bash
python evaluation.py
```

The evaluation script uses the predefined benchmark and holdout question sets and generates evaluation result files in the project directory.

**Note:** Running the evaluation can overwrite existing evaluation CSV files. Preserve a copy of the published results before rerunning experiments if needed.

## Limitations

- Model inference can be slow on CPU-based systems.
- Generated SQL queries may be incorrect.
- The evaluation uses a manually prepared question set and a synthetic SQLite database.
- The holdout partition is not independently authored or blind.
- The results do not establish generalization to arbitrary database schemas.
- Read-only SQL validation is an application-level safeguard, not a complete production-grade database security boundary.
- Query correction does not necessarily improve measured accuracy.
- The project currently focuses on SQLite rather than multiple database systems.

## Future Enhancements

- Improve SQL generation using richer schema and database context.
- Evaluate using independently authored and unseen question sets.
- Support additional relational database systems.
- Improve query validation and database security.
- Optimize model loading and inference speed.
- Expand result visualization and natural-language explanations.
- Explore more advanced SQL correction and verification strategies.

## Project Repository

GitHub: https://github.com/puneethreddy48/context-aware-ai-database-analyst

## Academic Project

Developed as an academic AI/ML project.

The project demonstrates how pretrained language models, rule-based SQL generation, database schema inspection, and read-only query validation can be integrated to make database exploration accessible through natural language.