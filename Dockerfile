# Sử dụng base image Python 3.10 trên nền Debian Bookworm (ổn định)
FROM python:3.10-slim-bookworm

# Đặt thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết.
# THÊM 'git' để có thể clone repo khi pip install từ requirements.txt.
# Cài thêm 'build-essential' để build các gói C++ như pybullet nếu cần.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements.txt vào trước để tận dụng Docker cache
COPY requirements.txt .

# Cài đặt PyTorch phiên bản CPU trước.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện Python còn lại từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ code của dự án vào thư mục làm việc trong container
COPY . .

# Lệnh mặc định khi container khởi chạy: mở một shell bash
CMD ["bash"]
