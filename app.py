import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx', 'txt'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

from PyPDF2 import PdfReader

def extract_text_from_file(filepath):
    try:
        if filepath.lower().endswith('.pdf'):
            with open(filepath, 'rb') as file:
                pdf = PdfReader(file)
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
        else:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
    except UnicodeDecodeError:
        # If utf-8 fails, try latin-1 encoding
        with open(filepath, 'r', encoding='latin-1') as file:
            text = file.read()
    return text



def score_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    scores = cosine_matrix[0, 1:]
    return scores

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.form['job_description']
        num_shortlist = int(request.form['num_shortlist'])
        uploaded_files = request.files.getlist('resumes')
        
        resumes = []
        filenames = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                text = extract_text_from_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                resumes.append(text)
                filenames.append(filename)
        
        scores = score_resumes(job_description, resumes)
        df = pd.DataFrame({'filename': filenames, 'score': scores})
        df = df.sort_values(by='score', ascending=False).head(num_shortlist)
        
        return render_template('results.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
