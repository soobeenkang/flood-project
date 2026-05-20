import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(encoding='utf-8')


def get_engine():
    return create_engine(
        f"postgresql://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 5432)}"
        f"/{os.getenv('DB_NAME')}"
    )