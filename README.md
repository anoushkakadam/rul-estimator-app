
# 🛠️ Predictive Maintenance - Remaining Useful Life (RUL) Estimator

This project is a Streamlit-based machine learning app that estimates the **Remaining Useful Life (RUL)** of jet engines based on sensor data. It enables real-time failure prediction and alerts, helping to avoid unexpected breakdowns and optimize maintenance schedules.

> ⚡ Built with Python, Scikit-learn, XGBoost, Streamlit  
> 📊 Trained on the NASA C-MAPSS FD001 dataset

---

## 🚀 Live Demo

▶️ **[Click here to run the app](https://rul-estimator-app-lxmtgadfjk7zpmnpceh3kh.streamlit.app/)**  


---

## 📂 Project Structure

 🔧 Features

- 📁 Upload your own sensor data (.txt or .csv)
- 📈 Predict RUL using ML models (Linear Regression & XGBoost)
- 🧠 View model performance metrics (MAE, RMSE, R²)
- 🔔 Receive failure risk alerts based on RUL threshold
- ⬇️ Export predictions to CSV

---

## 🧠 Models Used

- **Linear Regression**: Simple, interpretable baseline model
- **XGBoost Regressor**: Advanced gradient boosting model for higher accuracy

---

## 📊 Dataset Info

NASA C-MAPSS FD001 Dataset:  
- Sensor readings for 100 engines across operational cycles
- Label: Remaining Useful Life (RUL) per time step

> Learn more: [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

---

## ▶️ How to Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/rul-estimator-app.git
cd rul-estimator-app

# Step 2: Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # or env\\Scripts\\activate on Windows

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Streamlit app
streamlit run streamlit_app.py

## 👩‍💻 Author
Anoushka Kadam
🔗 LinkedIn 
📫 anoushkak2002@gmail.com





