# Context-Aware AI Database Analyst

**Review-2 MVP | Natural Language to SQL | AI-Powered Database Analysis**

## Project Overview

Context-Aware AI Database Analyst is an AI-powered application that allows users to interact with an SQLite database using natural language. Instead of manually writing SQL queries, users can ask questions in plain English and receive generated SQL queries and database results through an interactive Streamlit dashboard.

The prototype integrates a pretrained Text-to-SQL model, validates generated queries, executes permitted read-only SQL statements, and uses execution-error feedback for one retry when a query fails.

## Key Features

- **Database Schema Inspection:** Automatically inspects SQLite database tables and their structures.
- **Natural Language to SQL:** Converts user questions into SQL queries using the `alpecevit/flan-t5-base-text2sql` model.
- **Read-Only SQL Validation:** Validates generated queries and restricts execution to permitted read-only statements.
- **Self-Correction Mechanism:** Attempts one retry using database execution-error feedback.
- **Interactive Dashboard:** Displays generated SQL queries and corresponding database result rows using Streamlit.
- **Synthetic Sample Dataset:** Uses demonstration business data for database querying and analysis.

## System Workflow

1. User enters a question in natural language.
2. The application inspects the SQLite database schema.
3. The Text-to-SQL model generates an SQL query.
4. The generated query passes through read-only SQL validation.
5. The application executes the permitted query.
6. If execution fails, the system attempts one correction using the error feedback.
7. The dashboard displays the generated SQL and query results.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Database | SQLite |
| Text-to-SQL Model | `alpecevit/flan-t5-base-text2sql` |
| AI Model Platform | Hugging Face Transformers |
| Query Processing | SQL validation and execution |
| Data | Synthetic business demonstration dataset |

## Important Academic Note

This project integrates an existing pretrained Hugging Face model: `alpecevit/flan-t5-base-text2sql`.

According to its model card, the model is fine-tuned on SParC, Spider, and CoSQL datasets.

The project focuses on integrating Text-to-SQL generation, database schema inspection, read-only query validation, and an error-feedback-based self-correction mechanism.

**The model was not newly trained or fine-tuned as part of this prototype.** Model accuracy should not be claimed as an original experimental result unless it is measured using a clearly defined evaluation dataset and methodology.

## Project Structure

```text
context-aware-ai-database-analyst/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── business.db
│
└── evaluation/
    └── Evaluation results and reports
```

*The structure above is illustrative. Update it to match the actual files in the repository.*

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/puneethreddy48/context-aware-ai-database-analyst.git
```

### 2. Navigate to the project directory

```bash
cd context-aware-ai-database-analyst
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

## Deployment on Streamlit Community Cloud

1. Push the project files to your GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Select the `context-aware-ai-database-analyst` repository.
5. Choose the `main` branch.
6. Set `app.py` as the main application file.
7. Click **Deploy**.
8. Wait for the application and pretrained model to load.
9. Test the application using sample natural-language questions.

Do not upload API keys, passwords, private databases, or sensitive information.

## Limitations

- CPU-based deployment may be slow because the pretrained model must be loaded by the application.
- Generated SQL queries may be incorrect and should be inspected.
- The prototype uses SQLite and a small synthetic business dataset.
- Only SELECT/WITH queries are permitted by the implemented validation logic.
- This validation is a demonstration safeguard, not a complete production-grade database security boundary.
- The application provides a basic row and column summary rather than a verified natural-language explanation of every query result.

## Future Enhancements

- Improve SQL generation using more context-aware prompting.
- Support additional database systems.
- Improve query validation and database security.
- Expand the dataset and evaluate performance on a clearly defined benchmark.
- Improve result interpretation and natural-language explanations.
- Optimize model loading and inference speed.

## Project Purpose

The objective of this project is to demonstrate how natural-language processing and database technologies can be integrated to make database querying more accessible, while incorporating query validation and a basic self-correction mechanism.

---

**Developed as an academic AI/ML project.**