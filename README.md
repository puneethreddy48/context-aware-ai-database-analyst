# Context-Aware AI Database Analyst (Review-2 MVP)

## What this prototype demonstrates
- Database schema inspection (SQLite)
- Natural-language-to-SQL generation using `alpecevit/flan-t5-base-text2sql`
- Read-only SQL validation and execution
- One retry using execution-error feedback (self-correction loop)
- Streamlit dashboard showing SQL and result rows

## Important academic note
The Hugging Face model is an existing model fine-tuned on SParC, Spider and CoSQL according to its model card. This application integrates it; it does not fine-tune a new model. Do not report model accuracy as your own result unless you evaluate it on a clearly described test set.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app creates `data/business.db` automatically on first run. The sample data is synthetic demonstration data.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this README.
2. Do not upload private databases, API keys, passwords, or sensitive data.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose the repository, branch `main`, and main file `app.py`.
5. Deploy. The first start may take several minutes while the model downloads.
6. Open the public app URL and test the example questions.

## Limitations
- CPU deployment may be slow because the model is loaded by the app.
- Model-generated SQL may be incorrect; inspect it.
- The prototype uses SQLite and a small synthetic sales database.
- Only SELECT/WITH queries are allowed. This is a demonstration safeguard, not a complete security boundary for production.
- The app does not provide a verified natural-language explanation of every result; it provides a basic row/column summary.
