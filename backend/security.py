import bcrypt

def verify_password(plain_password, hashed_password):
    # bcrypt.checkpw expects bytes
    if isinstance(plain_password, str):
        plain_password = plain_password.encode('utf-8')
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    
    return bcrypt.checkpw(plain_password, hashed_password)

def get_password_hash(password):
    # bcrypt.hashpw expects bytes and returns bytes
    if isinstance(password, str):
        password = password.encode('utf-8')
    
    hashed = bcrypt.hashpw(password, bcrypt.gensalt())
    return hashed.decode('utf-8')
