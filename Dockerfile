# Use Python 3 base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /src

# Copy dependency list first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install -y make \
    texlive-latex-base texlive-latex-extra \
    texlive-latex-recommended latexmk
RUN rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

ENV PYTHONPATH="/booknet/src"

# Run application
CMD ["python", "src/main.py"]
