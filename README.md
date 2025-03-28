# Bias Auditing and Explainable AI Prototype for Immigration Visa Decisioning

This project demonstrates a responsible AI system using:
- ✅ Fairness auditing (AIF360)
- ✅ Explainability (LIME)
- ✅ Visa decision prediction (Logistic Regression)
- ✅ Deployed as a REST API via FastAPI

### 🚀 API Endpoints
- `POST /predict` – Returns visa decision (Accepted/Rejected)
- `POST /explain` – Returns LIME explanation for the decision
- `GET /audit` – Returns fairness metrics (disparate impact, mean difference)

### 🧪 Example input
```json
{
  "age": 25,
  "priors_count": 1,
  "race": 0
}
```

---

### ⚙️ Deployment on Railway
1. Create a new project on [https://railway.app](https://railway.app)
2. Upload all files:
   - `visa_ai_fastapi_backend.py`
   - `requirements.txt`
   - `Procfile`
3. Set the root directory and deploy the service
4. Use the generated URL (e.g. `https://visa-ai.up.railway.app/docs`) to access the Swagger UI

---

### ✅ Notes
- The model is trained on a subset of the COMPAS dataset (age, priors_count, race)
- Bias auditing uses AIF360
- No database is required – model is trained at startup
- Adjust features easily to expand the prototype