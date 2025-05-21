# EduNER-TR

Named Entity Recognition Project for Turkish Text

## Table of Contents

- [About the Project](#about-the-project)
- [Project Motivation](#project-motivation)
- [Installation](#installation)
- [Usage](#usage)
- [Web Interface](#web-interface)
- [Model File](#model-file)

## About the Project

EduNER-TR is a natural language processing project developed for identifying named entities in Turkish texts. This project automatically detects special entities such as person names, location names, and organization names in text.

## Project Motivation

This project was initiated to address specific needs in Turkish natural language processing, particularly in the educational domain:

- **Lack of Turkish NER Resources**: The absence of ready-made training datasets for Turkish language created a need to develop an original solution in this field.

- **Educational Data Processing**: In educational institutions, automatically extracting student and program information (name, surname, university, department, number) from texts is important to speed up data processing and reduce human errors.

- **Research Contributions**: The project enriches NER resources and research for the Turkish language, contributing to the field of Turkish NLP.

- **Data Mining and Automation**: The system enables extracting structured data from unstructured texts and facilitates automatic processing of academic documents and records.

As stated in the project report:

> "The developed NER system provides significant contributions in the following areas: Turkish NLP Research, Educational Data Processing, Data Mining, and Document Automation."

The model achieved impressive performance metrics, with validation accuracy improving from 76.75% to 99.18% after optimization of the label structure.

## Installation

Follow these steps to run the project:

1. Clone or download the project:
```bash
git clone [repo-url]
cd EduNER-TR
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # For Linux/Mac
# or
.venv\Scripts\activate  # For Windows

pip install -r requirements.txt
```

## Usage

To run the project from the console:

```bash
python ner_aplication.py
```

This command will load the NER model and make it ready to process texts.

To run the data processing steps:

```bash
python 01_data_proces.py
python 02_vocabulary_builder.py
python 03_data_processor.py
python 04_training_modell.py
```

## Web Interface

To run the web interface:

1. Copy the `best_model.pt` file to the `Web` folder.
2. Start the web server:

```bash
cd Web
python app.py
```

You can access the web interface by going to `http://localhost:5000` in your browser.

Note: The necessary files are already in the Web folder.

You can also access the live demo at: [ner.akpinar.dev](https://ner.akpinar.dev)

## Model File

The trained model file (`best_model.pt`) can be downloaded by following these steps:

1. Download the `GEN-AI.zip` file from [Google Drive link](https://drive.google.com/drive/folders/1A-RiqPyJgV_oFBVPUTNzkqpkVwcp_KLA?usp=drive_link).
2. Extract the ZIP file and locate the `best_model.pt` file.
3. Place the file in one of the following locations depending on your use case:
   - For console application: In the project root directory (`EduNER-TR/`)
   - For web interface: In the `Web/` folder
   - For using the old model version: Copy to the `Versions/V1` directory and run `model_uygulama.py`.

Note: The ZIP file contains the best model trained so far. The file size is large (2.9 GB), so it's not included in the GitHub repository. 