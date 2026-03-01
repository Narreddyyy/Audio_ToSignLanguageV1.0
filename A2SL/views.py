from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.staticfiles import finders
from django.http import JsonResponse  
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator
import re 
import io

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    # Expand common English contractions
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    # Remove any remaining punctuation (keeping only letters and numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def home_view(request):
    return render(request, 'home.html')

def about_view(request):
    return render(request, 'about.html')

def contact_view(request):
    return render(request, 'contact.html')

@login_required(login_url="login")
def animation_view(request):
    if request.method == 'POST':
        original_text = request.POST.get('sen')
        if not original_text:
            return render(request, 'animation.html')

        # Translation Layer (Multilingual Support)
        try:
            text_en = GoogleTranslator(source='auto', target='en').translate(original_text)
        except Exception:
            text_en = original_text

        # Text Cleaning 
        clean_en = clean_text(text_en)

        # Tokenization & Tagging
        words = word_tokenize(clean_en)
        tagged = nltk.pos_tag(words)

        # Glossing (English -> ASL Logic)
        lr = WordNetLemmatizer()
        gloss_words = []

        for w, tag in tagged:
            # Skip stopwords
            if w in ['a', 'an', 'the', 'is', 'am', 'are', 'was', 'were', 'to', 'do', 'does', 'did', 'done']:
                continue

            if w == 'i':
                gloss_words.append('me')
                continue    
            
            wn_tag = get_wordnet_pos(tag)
            lemma = lr.lemmatize(w, pos=wn_tag)
            gloss_words.append(lemma)

        # Animation Matching
        final_words = []
        for w in gloss_words:
            path = w + ".mp4"
            if finders.find(path):
                final_words.append(w)
            else:
                for char in w:
                    final_words.append(char)

        # --- REAL TIME AUDIO SUPPORT ---
        # If the request comes from JavaScript (AJAX), return JSON data
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'words': final_words, 'text': original_text})

        # --- NORMAL PAGE LOAD ---
        # FIXED: Removed the duplicate return statement. 
        # This single return handles the standard page load.
        return render(request, 'animation.html', {
            'words': final_words, 
            'text': original_text,
            'translated': clean_en 
        })
    else:
        return render(request, 'animation.html')


# ─── Shared NLP Pipeline ────────────────────────────────────────────────────

def _process_text_to_words(original_text):
    """Run the full NLP pipeline on a piece of text and return (final_words, clean_en)."""
    # Translation
    try:
        text_en = GoogleTranslator(source='auto', target='en').translate(original_text)
    except Exception:
        text_en = original_text

    clean_en = clean_text(text_en)
    words = word_tokenize(clean_en)
    tagged = nltk.pos_tag(words)

    lr = WordNetLemmatizer()
    gloss_words = []
    for w, tag in tagged:
        if w in ['a', 'an', 'the', 'is', 'am', 'are', 'was', 'were', 'to', 'do', 'does', 'did', 'done']:
            continue
        if w == 'i':
            gloss_words.append('me')
            continue
        wn_tag = get_wordnet_pos(tag)
        lemma = lr.lemmatize(w, pos=wn_tag)
        gloss_words.append(lemma)

    final_words = []
    for w in gloss_words:
        if finders.find(w + ".mp4"):
            final_words.append(w)
        else:
            for char in w:
                final_words.append(char)

    return final_words, clean_en


# ─── Process Plain Text ──────────────────────────────────────────────────────

@login_required(login_url="login")
def process_text_view(request):
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        if not text:
            return JsonResponse({'error': 'No text provided.'}, status=400)
        final_words, clean_en = _process_text_to_words(text)
        return JsonResponse({'words': final_words, 'text': text, 'translated': clean_en})
    return JsonResponse({'error': 'POST required.'}, status=405)


# ─── Upload Document ─────────────────────────────────────────────────────────

@login_required(login_url="login")
def upload_document_view(request):
    if request.method == 'POST':
        doc = request.FILES.get('document')
        if not doc:
            return JsonResponse({'error': 'No file uploaded.'}, status=400)

        filename = doc.name.lower()
        extracted = ''

        try:
            if filename.endswith('.txt'):
                extracted = doc.read().decode('utf-8', errors='ignore')

            elif filename.endswith('.pdf'):
                try:
                    from pypdf import PdfReader
                except ImportError:
                    from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(doc.read()))
                extracted = ' '.join(
                    page.extract_text() or '' for page in reader.pages
                )

            elif filename.endswith('.docx'):
                from docx import Document
                doc_obj = Document(io.BytesIO(doc.read()))
                extracted = ' '.join(p.text for p in doc_obj.paragraphs)

            else:
                return JsonResponse(
                    {'error': 'Unsupported file type. Please upload .txt, .pdf, or .docx'},
                    status=400
                )
        except Exception as e:
            return JsonResponse({'error': f'Failed to read file: {str(e)}'}, status=500)

        extracted = extracted.strip()
        if not extracted:
            return JsonResponse({'error': 'No text could be extracted from the document.'}, status=400)

        # Limit to first 500 words to keep the animation queue manageable
        words_raw = extracted.split()
        if len(words_raw) > 500:
            extracted = ' '.join(words_raw[:500])

        final_words, clean_en = _process_text_to_words(extracted)
        return JsonResponse({'words': final_words, 'text': extracted[:200] + ('...' if len(extracted) > 200 else ''), 'translated': clean_en})

    return JsonResponse({'error': 'POST required.'}, status=405)

# Auth Views
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('animation')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            else:
                return redirect('animation')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect("home")