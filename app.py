from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.secret_key = "parkinson-ai-secret-key"


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
IMPUTER_FILE = os.path.join(BASE_DIR, "imputer.pkl")

DATABASE = os.path.join(BASE_DIR, "users.db")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.csv")


# =========================================================
# LOAD MACHINE LEARNING FILES
# =========================================================

try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    imputer = joblib.load(IMPUTER_FILE)

    print("Machine learning files loaded successfully.")

except Exception as e:

    print("ERROR loading machine learning files:")
    print(e)

    model = None
    scaler = None
    imputer = None


# =========================================================
# MODEL FEATURES
# These must match the order used when training the model
# =========================================================

columns = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]


# =========================================================
# FORM FIELD NAMES
# These must match name="" in index.html
# =========================================================

feature_names = [
    "Fo",
    "Fhi",
    "Flo",
    "Jitter",
    "JitterAbs",
    "RAP",
    "PPQ",
    "DDP",
    "Shimmer",
    "ShimmerDB",
    "APQ3",
    "APQ5",
    "APQ",
    "DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_db():

    conn = None

    try:

        conn = sqlite3.connect(DATABASE)

        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)

        conn.commit()

        print("Database initialized successfully.")

    except Exception as e:

        print("Database initialization error:")
        print(e)

    finally:

        if conn:
            conn.close()


init_db()


# =========================================================
# LOGIN CHECK
# =========================================================

def login_required():

    return "user_id" in session


# =========================================================
# DASHBOARD DATA
# =========================================================

def get_dashboard_data():

    total_predictions = 0
    parkinson_count = 0
    no_parkinson_count = 0
    average_confidence = 0

    prediction_dates = []
    prediction_counts = []

    # -----------------------------------------------------
    # NO HISTORY FILE
    # -----------------------------------------------------

    if not os.path.exists(HISTORY_FILE):

        return {
            "total_predictions": 0,
            "parkinson_count": 0,
            "no_parkinson_count": 0,
            "average_confidence": 0,
            "prediction_dates": [],
            "prediction_counts": []
        }

    try:

        df = pd.read_csv(HISTORY_FILE)

        # -------------------------------------------------
        # EMPTY FILE
        # -------------------------------------------------

        if df.empty:

            return {
                "total_predictions": 0,
                "parkinson_count": 0,
                "no_parkinson_count": 0,
                "average_confidence": 0,
                "prediction_dates": [],
                "prediction_counts": []
            }

        # -------------------------------------------------
        # TOTAL PREDICTIONS
        # -------------------------------------------------

        total_predictions = len(df)

        # -------------------------------------------------
        # PARKINSON COUNT
        # -------------------------------------------------

        if "Prediction" in df.columns:

            prediction_text = (
                df["Prediction"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            parkinson_count = int(
                prediction_text
                .eq("parkinson's disease detected")
                .sum()
            )

            no_parkinson_count = (
                total_predictions - parkinson_count
            )

        # -------------------------------------------------
        # AVERAGE CONFIDENCE
        # -------------------------------------------------

        if "Confidence" in df.columns:

            confidence_values = pd.to_numeric(
                df["Confidence"],
                errors="coerce"
            )

            if confidence_values.notna().any():

                average_confidence = round(
                    float(confidence_values.mean()),
                    2
                )

        # -------------------------------------------------
        # PREDICTION HISTORY CHART
        # -------------------------------------------------

        if "Date" in df.columns:

            dates = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            chart_df = pd.DataFrame({
                "Date": dates
            })

            chart_df = chart_df.dropna()

            if not chart_df.empty:

                chart_df["ChartDate"] = (
                    chart_df["Date"]
                    .dt.strftime("%d-%m-%Y")
                )

                grouped = (
                    chart_df
                    .groupby("ChartDate", sort=False)
                    .size()
                )

                prediction_dates = [
                    str(value)
                    for value in grouped.index
                ]

                prediction_counts = [
                    int(value)
                    for value in grouped.values
                ]

    except Exception as e:

        print("Dashboard error:")
        print(e)

    return {
        "total_predictions": total_predictions,
        "parkinson_count": parkinson_count,
        "no_parkinson_count": no_parkinson_count,
        "average_confidence": average_confidence,
        "prediction_dates": prediction_dates,
        "prediction_counts": prediction_counts
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    if login_required():

        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    # -----------------------------------------------------
    # ALREADY LOGGED IN
    # -----------------------------------------------------

    if login_required():

        return redirect(
            url_for("dashboard")
        )

    # -----------------------------------------------------
    # POST
    # -----------------------------------------------------

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        # -------------------------------------------------
        # EMPTY CHECK
        # -------------------------------------------------

        if not username or not email or not password:

            flash(
                "Please fill in all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # -------------------------------------------------
        # PASSWORD MATCH
        # -------------------------------------------------

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # -------------------------------------------------
        # PASSWORD LENGTH
        # -------------------------------------------------

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # -------------------------------------------------
        # HASH PASSWORD
        # -------------------------------------------------

        password_hash = generate_password_hash(
            password
        )

        conn = None

        try:

            conn = sqlite3.connect(
                DATABASE
            )

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO users
                (username, email, password)
                VALUES (?, ?, ?)
            """, (
                username,
                email,
                password_hash
            ))

            conn.commit()

            flash(
                "Registration successful! Please login.",
                "success"
            )

            return redirect(
                url_for("login")
            )

        except sqlite3.IntegrityError:

            flash(
                "Username or email already exists.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        except Exception as e:

            print("Registration error:")
            print(e)

            flash(
                "Registration failed. Please try again.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        finally:

            if conn:
                conn.close()

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN PAGE
# =========================================================

@app.route("/login", methods=["GET"])
def login():

    if login_required():

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "login.html"
    )


# =========================================================
# LOGIN PROCESS
# =========================================================

@app.route("/check-login", methods=["POST"])
def check_login():

    username = request.form.get(
        "username",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    # -----------------------------------------------------
    # EMPTY CHECK
    # -----------------------------------------------------

    if not username or not password:

        flash(
            "Please enter username and password.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    conn = None
    user = None

    try:

        conn = sqlite3.connect(
            DATABASE
        )

        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, username, email, password
            FROM users
            WHERE username = ?
        """, (
            username,
        ))

        user = cursor.fetchone()

    except Exception as e:

        print("Login database error:")
        print(e)

        flash(
            "Login error. Please try again.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    finally:

        if conn:
            conn.close()

    # -----------------------------------------------------
    # CHECK USER
    # -----------------------------------------------------

    if user:

        stored_password = user[3]

        try:

            password_correct = check_password_hash(
                stored_password,
                password
            )

        except Exception as e:

            print("Password checking error:")
            print(e)

            password_correct = False

        # -------------------------------------------------
        # LOGIN SUCCESS
        # -------------------------------------------------

        if password_correct:

            session.clear()

            session["user_id"] = user[0]
            session["username"] = user[1]
            session["email"] = user[2]

            flash(
                "Login successful!",
                "success"
            )

            return redirect(
                url_for("dashboard")
            )

    # -----------------------------------------------------
    # LOGIN FAILED
    # -----------------------------------------------------

    flash(
        "Invalid username or password.",
        "error"
    )

    return redirect(
        url_for("login")
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard", methods=["GET"])
def dashboard():

    if not login_required():

        return redirect(
            url_for("login")
        )

    data = get_dashboard_data()

    return render_template(
        "dashboard.html",
        **data
    )


# =========================================================
# PATIENT INFORMATION PAGE
# =========================================================

@app.route("/prediction", methods=["GET"])
def prediction():

    if not login_required():

        return redirect(
            url_for("login")
        )

    return render_template(
        "patient.html"
    )


# =========================================================
# START PREDICTION
# Receives patient information
# =========================================================

@app.route(
    "/start-prediction",
    methods=["GET", "POST"]
)
def start_prediction():

    if not login_required():

        return redirect(
            url_for("login")
        )

    # -----------------------------------------------------
    # IF OPENED DIRECTLY
    # -----------------------------------------------------

    if request.method == "GET":

        return redirect(
            url_for("prediction")
        )

    # -----------------------------------------------------
    # PATIENT INFORMATION
    # -----------------------------------------------------

    patient_name = request.form.get(
        "patient_name",
        ""
    ).strip()

    age = request.form.get(
        "age",
        ""
    ).strip()

    gender = request.form.get(
        "gender",
        ""
    ).strip()

    phone = request.form.get(
        "phone",
        ""
    ).strip()

    patient_id = request.form.get(
        "patient_id",
        ""
    ).strip()

    exam_date = request.form.get(
        "exam_date",
        ""
    ).strip()

    notes = request.form.get(
        "notes",
        ""
    ).strip()

    # -----------------------------------------------------
    # RENDER ML INPUT PAGE
    # -----------------------------------------------------

    return render_template(
        "index.html",
        patient_name=patient_name,
        age=age,
        gender=gender,
        phone=phone,
        patient_id=patient_id,
        exam_date=exam_date,
        notes=notes
    )


# =========================================================
# MACHINE LEARNING PREDICTION
# index.html submits to /predict
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if not login_required():

        return redirect(
            url_for("login")
        )

    try:

        # =================================================
        # CHECK MODEL FILES
        # =================================================

        if model is None:

            raise RuntimeError(
                "model.pkl could not be loaded."
            )

        if scaler is None:

            raise RuntimeError(
                "scaler.pkl could not be loaded."
            )

        if imputer is None:

            raise RuntimeError(
                "imputer.pkl could not be loaded."
            )

        # =================================================
        # PATIENT INFORMATION
        # =================================================

        patient_name = request.form.get(
            "patient_name",
            ""
        ).strip()

        age = request.form.get(
            "age",
            ""
        ).strip()

        gender = request.form.get(
            "gender",
            ""
        ).strip()

        phone = request.form.get(
            "phone",
            ""
        ).strip()

        patient_id = request.form.get(
            "patient_id",
            ""
        ).strip()

        exam_date = request.form.get(
            "exam_date",
            ""
        ).strip()

        notes = request.form.get(
            "notes",
            ""
        ).strip()

        # =================================================
        # MODEL FEATURES
        # =================================================

        values = []

        for name in feature_names:

            value = request.form.get(
                name,
                ""
            ).strip()

            if value == "":

                raise ValueError(
                    f"Missing feature: {name}"
                )

            try:

                numeric_value = float(
                    value
                )

                values.append(
                    numeric_value
                )

            except ValueError:

                raise ValueError(
                    f"Invalid value for {name}: {value}"
                )

        # =================================================
        # CHECK FEATURE COUNT
        # =================================================

        if len(values) != len(columns):

            raise ValueError(
                f"Expected {len(columns)} features, "
                f"but received {len(values)}."
            )

        # =================================================
        # CREATE DATAFRAME
        # =================================================

        input_data = pd.DataFrame(
            [values],
            columns=columns
        )

        # =================================================
        # IMPUTER
        # =================================================

        input_data = imputer.transform(
            input_data
        )

        # =================================================
        # SCALER
        # =================================================

        input_data = scaler.transform(
            input_data
        )

        # =================================================
        # MODEL PREDICTION
        # =================================================

        prediction_value = model.predict(
            input_data
        )[0]

        # =================================================
        # PROBABILITY
        # =================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = model.predict_proba(
                input_data
            )[0]

            classes = list(
                getattr(
                    model,
                    "classes_",
                    [0, 1]
                )
            )

            if 1 in classes:

                positive_index = classes.index(1)

                positive_probability = float(
                    probability[positive_index]
                )

            else:

                positive_probability = 0.0

        else:

            positive_probability = (
                1.0
                if prediction_value == 1
                else 0.0
            )

        # =================================================
        # RESULT
        # =================================================

        if prediction_value == 1:

            result = (
                "Parkinson's Disease Detected"
            )

            confidence = round(
                positive_probability * 100,
                2
            )

            risk_level = "High Risk"

            risk_class = "high"

        else:

            result = (
                "No Parkinson's Disease Detected"
            )

            confidence = round(
                (1 - positive_probability) * 100,
                2
            )

            risk_level = "Low Risk"

            risk_class = "low"

        # =================================================
        # SAVE HISTORY
        # =================================================

        history_data = {

            "Date":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "Examination Date":
                exam_date,

            "Patient ID":
                patient_id,

            "Patient Name":
                patient_name,

            "Age":
                age,

            "Gender":
                gender,

            "Phone":
                phone,

            "Prediction":
                result,

            "Confidence":
                confidence,

            "Risk Level":
                risk_level,

            "Notes":
                notes
        }

        # =================================================
        # READ EXISTING HISTORY
        # =================================================

        if os.path.exists(
            HISTORY_FILE
        ):

            try:

                df = pd.read_csv(
                    HISTORY_FILE
                )

            except Exception:

                df = pd.DataFrame()

        else:

            df = pd.DataFrame()

        # =================================================
        # ADD NEW RECORD
        # =================================================

        new_record = pd.DataFrame(
            [history_data]
        )

        df = pd.concat(
            [
                df,
                new_record
            ],
            ignore_index=True
        )

        # =================================================
        # SAVE CSV
        # =================================================

        df.to_csv(
            HISTORY_FILE,
            index=False
        )

        # =================================================
        # RECORD ID
        # =================================================

        record_id = len(df) - 1

        # =================================================
        # RESULT PAGE
        # =================================================

        return render_template(
            "result.html",

            record_id=record_id,

            prediction=result,

            confidence=confidence,

            risk_level=risk_level,

            risk_class=risk_class,

            patient_name=patient_name,

            age=age,

            gender=gender,

            phone=phone,

            patient_id=patient_id,

            exam_date=exam_date,

            notes=notes
        )

    except Exception as e:

        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)
        print(e)
        print("=" * 60)

        return render_template(
            "error.html",
            error=str(e)
        )


# =========================================================
# HISTORY
# =========================================================

@app.route("/history")
def history():

    if not login_required():

        return redirect(
            url_for("login")
        )

    records = []

    if os.path.exists(
        HISTORY_FILE
    ):

        try:

            df = pd.read_csv(
                HISTORY_FILE
            )

            # Replace NaN values
            df = df.fillna("")

            for index, row in df.iterrows():

                record = row.to_dict()

                record["_original_index"] = int(
                    index
                )

                records.append(
                    record
                )

        except Exception as e:

            print("History error:")
            print(e)

    return render_template(
        "history.html",
        records=records
    )


# =========================================================
# DELETE ONE RECORD
# =========================================================

@app.route(
    "/delete-record/<int:record_id>",
    methods=["POST"]
)
def delete_record(record_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    if not os.path.exists(
        HISTORY_FILE
    ):

        flash(
            "No prediction history found.",
            "error"
        )

        return redirect(
            url_for("history")
        )

    try:

        df = pd.read_csv(
            HISTORY_FILE
        )

        # -------------------------------------------------
        # CHECK RECORD
        # -------------------------------------------------

        if record_id < 0 or record_id >= len(df):

            flash(
                "Record not found.",
                "error"
            )

            return redirect(
                url_for("history")
            )

        # -------------------------------------------------
        # DELETE RECORD
        # -------------------------------------------------

        df = df.drop(
            index=record_id
        )

        df = df.reset_index(
            drop=True
        )

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        if df.empty:

            os.remove(
                HISTORY_FILE
            )

        else:

            df.to_csv(
                HISTORY_FILE,
                index=False
            )

        flash(
            "Prediction record deleted successfully.",
            "success"
        )

    except Exception as e:

        print("Delete error:")
        print(e)

        flash(
            "Unable to delete prediction record.",
            "error"
        )

    return redirect(
        url_for("history")
    )


# =========================================================
# CLEAR ALL HISTORY
# =========================================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    if not login_required():

        return redirect(
            url_for("login")
        )

    try:

        if os.path.exists(
            HISTORY_FILE
        ):

            os.remove(
                HISTORY_FILE
            )

            flash(
                "All prediction history deleted.",
                "success"
            )

        else:

            flash(
                "There is no history to delete.",
                "error"
            )

    except Exception as e:

        print("Clear history error:")
        print(e)

        flash(
            "Unable to clear history.",
            "error"
        )

    return redirect(
        url_for("history")
    )


# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():

    if not login_required():

        return redirect(
            url_for("login")
        )

    return render_template(
        "about.html"
    )


# =========================================================
# ERROR HANDLER - 404
# =========================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "error.html",
        error="Page not found."
    ), 404


# =========================================================
# ERROR HANDLER - 500
# =========================================================

@app.errorhandler(500)
def internal_server_error(error):

    return render_template(
        "error.html",
        error="Internal server error."
    ), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PARKINSON AI PREDICTION SYSTEM")
    print("=" * 60)
    print("Server starting...")
    print("Open: http://127.0.0.1:5000/")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )