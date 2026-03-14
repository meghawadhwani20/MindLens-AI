from .storage import find_user_by_email

def validate_email_format(email: str) -> bool:
    if not email or not isinstance(email, str):
        return False
    email = email.strip()
    if email.lower() in ["undefined", "null", "none", ""]:
        return False
    if "@" not in email or "." not in email:
        return False
    return True

def validate_user_exists(email: str):
    """
    Checks if the user exists in the database.
    Raises ValueError if email is invalid or user does not exist.
    """
    if not validate_email_format(email):
        raise ValueError("Invalid email address provided.")
    
    user = find_user_by_email(email)
    if not user:
        raise ValueError(f"User with email {email} does not exist. Please sign up or log in.")
