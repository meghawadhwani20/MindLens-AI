from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional

class UserSignup(BaseModel):
    full_name: str
    email: EmailStr
    password: str = Field(..., max_length=10)

    @validator('password')
    def validate_password_length(cls, v):
        if len(v) > 10:
            raise ValueError('Password must be maximum 10 characters')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., max_length=10)

class UserResetPassword(BaseModel):
    email: EmailStr
    new_password: str = Field(..., max_length=10)
    confirm_password: str = Field(..., max_length=10)

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

class HabitCreate(BaseModel):
    email: EmailStr
    age: int
    sleep_hours: float
    work_hours: float
    screen_time: float
    water_intake: float
    exercise: bool
    meals_per_day: int
    social_interaction: str
    caffeine_intake: bool

class ChatMessage(BaseModel):
    email: EmailStr
    message: str

