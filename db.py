import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "career_chatbot"),
    )

def save_chat(user_query: str, bot_response: str, sources: list[str]):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_query, bot_response, sources) VALUES (%s, %s, %s)",
        (user_query, bot_response, ", ".join(sources)),
    )
    conn.commit()
    cursor.close()
    conn.close()