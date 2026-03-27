# Intelligent Hospital Bed Allocation

## Run locally

1. Install dependencies:
```powershell
pip install -r requirements.txt
```

2. Start the dashboard:
```powershell
streamlit run dashboard/app.py
```

## Deploy

### Streamlit Community Cloud

1. Push this project to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the main file path to:
```text
dashboard/app.py
```
4. Streamlit will install packages from `requirements.txt`.

### Render

1. Create a new Web Service from the repo.
2. Use:
```text
Build Command: pip install -r requirements.txt
Start Command: streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
```
3. The included `Procfile` matches this start command.

## Notes

- SQLite databases are stored in `database/`.
- Saved models are stored in `saved_models/`.
- For production demos, keep the same Python environment used for training/evaluation when possible.

## Demo Login

- `citycare / citycare123`
- `sunrise / sunrise123`
