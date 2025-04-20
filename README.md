# credit_risk_assist
Run "python -m venv venv" <br>
Run "pip install -r requirements.txt" <br>
Run "python app.py"<br>

Install pytesseract from: https://github.com/UB-Mannheim/tesseract/wiki
change the tesseract path in load_images.py


docker run -p 6333:6333 -p 6334:6334 -v %cd%\qdrant_data:/qdrant/storage qdrant/qdrant