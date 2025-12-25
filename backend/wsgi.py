# ... new file ...

from app import app, initialize_models

# โหลดโมเดลตอนคอนเทนเนอร์สตาร์ท
initialize_models()

# Gunicorn จะใช้ตัวแปร `app` จากโมดูลนี้
# app = app