from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ✅ Enable CORS — this must be before route definitions!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],  # Add both variants
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load model
model = joblib.load("fake_profile_model.pkl")

class Profile(BaseModel):
    account_age_days: int
    followers: int
    following: int
    posts_per_day: float

@app.post("/predict_profile")
def predict_profile(profile: Profile):
    ratio = profile.followers / (profile.following + 1e-5)
    input_df = pd.DataFrame([{
        "account_age_days": profile.account_age_days,
        "posts_per_day": profile.posts_per_day,
        "follower_following_ratio": ratio
    }])
    prediction = model.predict(input_df)[0]
    return {"is_fake": bool(prediction)}
