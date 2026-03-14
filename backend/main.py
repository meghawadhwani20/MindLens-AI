import json
import logging
from fastapi import FastAPI, HTTPException, status, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from fastapi import Request
from typing import List
from .models import UserSignup, UserLogin, UserResetPassword, HabitCreate, ChatMessage
from .security import get_password_hash, verify_password
from .validation import validate_user_exists
from .chatbot import init_chatbot, get_chat_response
from .serial_reader import start_serial_reader, get_latest
from .routes.realtime import router as realtime_router
from .services.model_service import ensure_models_loaded, update_habit_prediction, update_sensor_state
from .storage import count_rows, fetch_rows, find_user_by_email, insert_row, update_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Chatbot on startup
@app.on_event("startup")
async def startup_event():
    init_chatbot()
    start_serial_reader(port="COM5", baudrate=115200)
    await ensure_models_loaded()
    logger.info("Application startup completed.")

# Custom Validation Handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Check if it's a password length error or missing field
    errors = exc.errors()
    for error in errors:
        loc = error.get('loc', [])
        msg = error.get('msg', '')
        
        if 'password' in loc and ('max_length' in error.get('type', '') or 'maximum 10 characters' in str(exc)):
             return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Password must be maximum 10 characters"}
            )
        if error.get('type') == 'value_error.missing':
             return JSONResponse(
                status_code=400,
                content={"success": False, "message": "All fields are required"}
            )
    
    return JSONResponse(
        status_code=400,
        content={"success": False, "message": "All fields are required"}
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"
PUBLIC_DIR = FRONTEND_DIR / "public"

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# Jinja2 Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- HTML Routes ---
@app.get("/", response_class=HTMLResponse, name="index")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse, name="signup")
async def read_signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/forgot-password.html", response_class=HTMLResponse, name="forgot_password")
async def read_forgot_password(request: Request):
    return templates.TemplateResponse("forgot-password.html", {"request": request})

@app.get("/reset-password", response_class=HTMLResponse, name="reset_password")
async def read_reset_password(request: Request):
    return templates.TemplateResponse("reset_password.html", {"request": request})

@app.get("/homepage.html", response_class=HTMLResponse, name="homepage")
async def read_homepage(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})

@app.get("/chatbot.html", response_class=HTMLResponse, name="chatbot")
async def read_chatbot(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@app.get("/verify-otp", response_class=HTMLResponse, name="verify_otp")
async def read_verify_otp(request: Request):
    return templates.TemplateResponse("verify-otp.html", {"request": request})

@app.get("/habit-tracker", response_class=HTMLResponse, name="habit_tracker")
async def read_habit_tracker(request: Request):
    return templates.TemplateResponse("habit-tracker.html", {"request": request})

@app.get("/stress-scanner", response_class=HTMLResponse, name="stress_scanner")
async def read_stress_scanner(request: Request):
    return templates.TemplateResponse("stress-scanner.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse, name="dashboard")
async def read_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# --- API Endpoints ---

@app.get("/latest-data")
async def latest_data():
    return get_latest()

class SensorSubmit(BaseModel):
    heart_rate: float
    spo2: float
    temperature: float

@app.post("/submit")
async def submit_sensor_values(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            form = await request.form()
            body = dict(form)
        hr = float(body.get("heart_rate"))
        sp = float(body.get("spo2"))
        tp = float(body.get("temperature"))
        print(f"Received data:\nheart_rate = {hr}\nspo2 = {sp}\ntemperature = {tp}")
        record = {
            "heart_rate": hr,
            "spo2": sp,
            "temperature": tp,
            "created_at": datetime.utcnow().isoformat()
        }
        insert_row("sensor_records", record)
        await update_sensor_state(record)
        logger.info("Sensor record stored successfully.")
        return {"success": True}
    except Exception as e:
        logger.exception("Failed to save sensor record.")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Failed to save sensor record"}
        )

@app.post("/signup")
async def signup(user: UserSignup):
    try:
        existing_user = find_user_by_email(user.email)
        if existing_user:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "User already exists, please sign in"}
            )
        
        hashed_pwd = get_password_hash(user.password)
        data = {
            "full_name": user.full_name,
            "email": user.email,
            "password": hashed_pwd
        }
        insert_row("users", data)
        return {"success": True, "message": "Signup successful"}
    except Exception as e:
        print(f"Signup error: {e}")
        if "Password must be maximum 10 characters" in str(e):
             return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Password must be maximum 10 characters"}
            )
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Internal server error"}
        )

@app.post("/login")
async def login(user: UserLogin):
    try:
        db_user = find_user_by_email(user.email)
        if not db_user:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Email does not exist, please signup first"}
            )

        if not verify_password(user.password, db_user["password"]):
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Wrong password"}
            )
            
        return {
            "success": True, 
            "message": "Login successful",
            "redirect": "/homepage.html",
            "user": {"email": db_user["email"], "full_name": db_user.get("full_name")}
        }
    except Exception as e:
        print(f"Login error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Internal server error"}
        )

@app.post("/reset-password")
async def reset_password(data: UserResetPassword):
    try:
        existing_user = find_user_by_email(data.email)
        if not existing_user:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Email not registered, please signup first"}
            )

        hashed_pwd = get_password_hash(data.new_password)
        update_rows("users", {"email": data.email}, {"password": hashed_pwd})
        return {"success": True, "message": "Password reset successful, please sign in"}
    except ValueError as e:
         return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(e)}
        )
    except Exception as e:
        print(f"Reset pwd error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Internal server error"}
        )

@app.post("/submit-habits")
async def submit_habits(habit: HabitCreate):
    try:
        # Validate user exists before inserting
        validate_user_exists(habit.email)
        
        data = habit.dict()
        data["timestamp"] = datetime.utcnow().isoformat()
        insert_row("habits", data)
        await update_habit_prediction(data)
        logger.info("Habit data stored successfully for %s", habit.email)
        return {"success": True, "message": "Habits saved successfully"}
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(e)}
        )
    except Exception as e:
        logger.exception("Habit submission error for %s", habit.email)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Failed to save habits"}
        )

@app.get("/api/habits")
async def get_habits(email: str, limit: int = 5):
    try:
        data = fetch_rows("habits", email=email, limit=limit, order_candidates=("timestamp", "created_at"))
        total_count = count_rows("habits", email=email)
        payload = {"success": True, "data": data, "count": total_count}
        logger.info("Habits API fetched for %s: rows=%s total=%s", email, len(data), total_count)
        logger.info("Habits API JSON for %s: %s", email, json.dumps(payload, default=str))
        return payload
    except Exception as e:
        logger.exception("Fetch habits error for %s", email)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Failed to fetch habits"}
        )

@app.post("/scan")
async def scan_stress(email: str = Form(...), images: List[UploadFile] = File(...)):
    try:
        email = (email or "").strip()
        if not email:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Please sign in before starting a stress scan."}
            )
        if len(images) != 3:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Exactly 3 frames are required for a scan."}
            )

        # Validate user exists before processing
        validate_user_exists(email)

        from .stress_scanner import process_stress_scan
        analysis = await process_stress_scan(email, images)
        logger.info("Stress scan completed successfully for %s", email)
        return {"success": True, "data": analysis}
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(e)}
        )
    except Exception as e:
        logger.exception("Scan error for %s", email)
        return JSONResponse(
            status_code=500, 
            content={"success": False, "message": "Internal processing error"}
        )

@app.get("/api/latest-scan")
async def get_latest_scan_endpoint(email: str):
    try:
        from .stress_scanner import get_latest_stress_scan
        data = await get_latest_stress_scan(email)
        payload = {"success": True, "data": data}
        logger.info("Latest scan API JSON for %s: %s", email, json.dumps(payload, default=str))
        return payload
    except Exception as e:
        logger.exception("Fetch latest scan error for %s", email)
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to fetch scan"})

@app.get("/api/scans")
async def get_scans_endpoint(email: str, limit: int = 5):
    try:
        from .stress_scanner import get_all_stress_scans
        data = await get_all_stress_scans(email, limit)
        total_count = count_rows("stress_scans", email=email)
        payload = {"success": True, "data": data, "count": total_count}
        logger.info("Scans API fetched for %s: rows=%s total=%s", email, len(data), total_count)
        logger.info("Scans API JSON for %s: %s", email, json.dumps(payload, default=str))
        return payload
    except Exception as e:
        logger.exception("Fetch scans error for %s", email)
        return JSONResponse(status_code=500, content={"success": False, "message": "Failed to fetch scans"})

@app.post("/chat")
async def chat_endpoint(chat_data: ChatMessage):
    try:
        # Validate User
        validate_user_exists(chat_data.email)
        
        # Get Response
        response_text = await get_chat_response(chat_data.message)
        return {"success": True, "message": response_text}
    except ValueError as e:
        # Handle known errors (User validation, API key missing, API error)
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(e)}
        )
    except Exception as e:
        print(f"Chat error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Internal chat error"}
        )


def create_app() -> FastAPI:
    if not getattr(app.state, "realtime_router_registered", False):
        app.include_router(realtime_router)
        app.state.realtime_router_registered = True
    return app


app = create_app()
