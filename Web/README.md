# EduNER-TR Web-Based NER Application

This folder contains the web-based Named Entity Recognition (NER) application for the EduNER-TR project.

## Project Purpose
This project aims to make a machine learning model developed to automatically identify entities such as university, department, first name, last name, and ID number in Turkish texts accessible via the web. Users can enter text through the web interface and instantly see the model's predictions.

## Use Case
- Document verification in academic applications
- Automatic form filling
- Student information processing in educational institutions
- Information extraction from Turkish texts

## How the Web Application Works
- The user enters text into the web interface.
- The model in `ner_uygulama.py` is loaded on the server side and makes predictions.
- Results are visually presented to the user.

## Server Deployment Steps
1. Install the required libraries (e.g., Flask, torch, numpy, etc.).
2. Run the `app.py` file in the `Web` folder.
3. Go to `http://localhost:5000` in your browser.

### Step-by-Step Setup and Deployment

1. Open a terminal in the main project folder:
   ```bash
   cd Web
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python app.py
   ```
4. Open in browser:
   - [http://localhost:5000](http://localhost:5000)

> Note: The model file and `ner_uygulama.py` should be in the main directory. The model is automatically loaded when the server starts. If the model cannot be loaded, demo mode is activated.

## File Descriptions
- `app.py`: Flask-based web server
- `templates/index.html`: Web interface template
- `requirements.txt`: Required Python libraries

## Scientific Contribution and Purpose of Use
This web application makes the task of named entity recognition (NER) in Turkish texts easily accessible and reproducible. It is suitable for academic, institutional, and individual uses. The model automatically detects entities such as university, department, first name, last name, and ID number, speeding up information extraction processes and reducing human error. 