FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs (จำเป็นสำหรับ OpenCV/NumPy/Scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# คัดลอกโค้ด backend ทั้งหมด (รวม pyproject.toml)
COPY . /app

# ติดตั้ง dependency จาก pyproject และติดตั้งแบบ editable
# และติดตั้ง gunicorn สำหรับ production server
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir gunicorn

# สร้างโฟลเดอร์โมเดล (เมาท์มาจากภายนอกตอนรัน)
RUN mkdir -p /app/models

# ภายในคอนเทนเนอร์ใช้พอร์ต 8000
EXPOSE 8000

# รันด้วย Gunicorn บน :8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "120", "wsgi:app"]