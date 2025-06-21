
from flask import Flask, request, jsonify, render_template, send_file
from flask_caching import Cache
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import nltk
import inflect
import random
from nltk.corpus import wordnet as wn
import PyPDF2
from werkzeug.utils import secure_filename
from html import escape
import torch
import logging
import re
import io
import json
from fpdf import FPDF

# Conditional Wikipedia import
try:
    import wikipedia
except ImportError:
    wikipedia = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['CACHE_TYPE'] = 'SimpleCache'
cache = Cache(app)

# Download required NLTK data
nltk_resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
for resource in nltk_resources:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialize models with proper tokenizer configuration
def initialize_models():
    """Initialize NLP models with proper configuration"""
    models = {}
    
    try:
        logger.info("Initializing question generation model...")
        # Explicitly set legacy=False to avoid tokenizer warnings
        qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-e2e-qg")

        qg_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-e2e-qg")
        models['qg_pipeline'] = pipeline(
            "text2text-generation",
            model=qg_model,
            tokenizer=qg_tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        logger.error(f"Error initializing question generation: {e}")
        logger.info("Falling back to t5-small...")
        try:
            qg_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            qg_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            models['qg_pipeline'] = pipeline(
                "text2text-generation",
                model=qg_model,
                tokenizer=qg_tokenizer
            )
        except Exception:
            # Ultimate fallback
            models['qg_pipeline'] = pipeline("text2text-generation", model="t5-small")

    try:
        logger.info("Initializing question answering pipeline...")
        models['qa_pipeline'] = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        logger.error(f"Error initializing QA pipeline: {e}")
        models['qa_pipeline'] = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    try:
        logger.info("Initializing sentence transformer...")
        models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Error initializing sentence transformer: {e}")
        models['sentence_model'] = None

    return models

# Initialize models at startup
nlp_models = initialize_models()

# Helper functions
def clean_question(question):
    """Clean and format generated questions"""
    if not question:
        return None
        
    # Remove model-specific artifacts
    question = question.replace('<sep>', '').replace('generate question: ', '').strip()
    
    # Remove trailing punctuation
    question = question.rstrip('?.!')
    
    # Ensure it meets minimum length
    if len(question) < 10:
        return None
        
    # Capitalize and add question mark
    question = question[0].upper() + question[1:]
    if not question.endswith('?'):
        question += '?'
        
    return question

def is_question_unique(new_question, existing_questions):
    """Check if a question is semantically unique"""
    if not existing_questions or not nlp_models['sentence_model']:
        return True
        
    try:
        new_embedding = nlp_models['sentence_model'].encode(new_question)
        existing_embeddings = nlp_models['sentence_model'].encode(list(existing_questions))
        similarities = util.cos_sim(new_embedding, existing_embeddings)[0]
        return all(sim < 0.75 for sim in similarities)
    except Exception as e:
        logger.warning(f"Similarity check failed: {e}")
        return new_question not in existing_questions

def is_scientist_related_question(question_text):
    """Check if question is about a scientist/person"""
    scientist_keywords = [
        "who is", "who was", "who discovered", "who invented", "who developed",
        "who created", "who founded", "who pioneered", "who is known for",
        "which scientist", "which person", "which researcher", "whom"
    ]
    question_lower = question_text.lower()
    return any(keyword in question_lower for keyword in scientist_keywords)

def generate_distractors(answer, context, num_distractors=3):
    """Generate distractors for MCQ questions"""
    try:
        if not answer or not context:
            raise ValueError("Missing answer or context")
        answer_text = str(answer).strip().lower()
        distractors = set()
        if len(answer_text) < 1:
            raise ValueError("Answer text too short")
        if answer_text.replace('.', '', 1).isdigit():
            return handle_numeric_answer(answer_text, num_distractors)
            
        return handle_textual_answer(answer_text, context, num_distractors)
    except Exception as e:
        logger.error(f"Distractor generation error: {str(e)}")
        base = answer_text if isinstance(answer, str) else "option"
        return [f"{base.capitalize()} {i+1}" for i in range(num_distractors)]
p = inflect.engine()
def is_word_number(s):
    try:
        return p.number_to_words(p.words_to_number(s)) == s.lower()
    except:
        return False
def handle_numeric_answer(answer_text, num_distractors):
    distractors = set()
    try:
        if is_word_number(answer_text):
            value = p.words_to_number(answer_text)
            format_type = 'word'
        else:
            value = float(answer_text) if '.' in answer_text else int(answer_text)
            format_type = 'digit'
    except:
        return []
    magnitude = abs(value)
    if magnitude == 0:
        perturbations = [1, -1, 0.5, -0.5, 2, -2]
    elif magnitude < 1:
        scale = max(0.1, magnitude/10)
        perturbations = [scale, -scale, scale*2, -scale*2]
    elif magnitude < 10:
        perturbations = [1, -1, 2, -2, 0.5, -0.5]
    elif magnitude < 100:
        perturbations = [5, -5, 10, -10, 2, -2]
    else:
        scale = max(1, magnitude // 10)
        perturbations = [scale, -scale, scale*2, -scale*2]
    for p_val in perturbations:
        if len(distractors) >= num_distractors:
            break
        new_val = value + p_val
        if format_type == 'digit':
            distractor = str(int(new_val) if new_val == int(new_val) else round(new_val, 2))
        else:
            distractor = p.number_to_words(int(new_val))
        if distractor.lower() != answer_text.lower():
            distractors.add(distractor)
    return list(distractors)[:num_distractors]
def handle_textual_answer(answer_text, context, num_distractors):
    distractors = set()
    try:
        synonyms = set()
        antonyms = set()
        hypernyms = set()
        hyponyms = set()
        for syn in wn.synsets(answer_text):
            for lemma in syn.lemmas():
                clean_name = lemma.name().replace('_', ' ').lower()
                if clean_name != answer_text:
                    synonyms.add(clean_name)
                for antonym in lemma.antonyms():
                    clean_ant = antonym.name().replace('_', ' ').lower()
                    if clean_ant != answer_text:
                        antonyms.add(clean_ant)
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    clean_hyper = lemma.name().replace('_', ' ').lower()
                    if clean_hyper != answer_text:
                        hypernyms.add(clean_hyper)
            for hypo in syn.hyponyms():
                for lemma in hypo.lemmas():
                    clean_hypo = lemma.name().replace('_', ' ').lower()
                    if clean_hypo != answer_text:
                        hyponyms.add(clean_hypo)
        distractors.update(list(antonyms)[:1])
        if len(distractors) < num_distractors:
            distractors.update(list(hypernyms.union(hyponyms))[:num_distractors - len(distractors)])
        if len(distractors) < num_distractors:
            distractors.update(list(synonyms)[:num_distractors - len(distractors)])
    except Exception as e:
        logger.warning(f"WordNet processing failed: {str(e)}")
    if len(distractors) < num_distractors:
        try:
            sentences = nltk.sent_tokenize(context)
            words = nltk.word_tokenize(context.lower())
            pos_tags = nltk.pos_tag(words)
            candidate_terms = [
                word for word, pos in pos_tags 
                if pos in ['NN', 'NNS', 'NNP', 'NNPS'] 
                and word != answer_text
                and len(word) > 2
            ]
            term_freq = nltk.FreqDist(candidate_terms)
            top_terms = [term for term, _ in term_freq.most_common(10)]
            for term in top_terms:
                if len(distractors) >= num_distractors:
                    break
                if term not in distractors:
                    distractors.add(term)
        except Exception as e:
            logger.warning(f"Context processing failed: {str(e)}")
        if len(distractors) < num_distractors:
            general_fallbacks = [
                'scientist', 'engineer', 'researcher', 'device', 'machine', 'tool',
                'solution', 'method', 'concept', 'innovation', 'application',
                'experiment', 'reaction', 'model', 'component', 'system', 'strategy',
                'environment', 'organism', 'material', 'resource', 'mechanism'
            ]
            random.shuffle(general_fallbacks)
            for word in general_fallbacks:
                if len(distractors) >= num_distractors:
                    break
                if word.lower() != answer_text and word not in distractors:
                    distractors.add(word)
    final_distractors = []
    for d in list(distractors)[:num_distractors]:
        if d and isinstance(d, str):
            final_distractors.append(d.capitalize() if answer_text[0].isupper() else d)
    return final_distractors
@cache.memoize(timeout=3600)
def get_wikipedia_summary(topic, sentences):
    if not wikipedia:
        return topic
    try:
        return wikipedia.summary(topic, sentences=sentences, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=sentences)
    except Exception as e:
        logger.error(f"Error fetching Wikipedia summary: {e}")
        return topic
@app.route('/')
def serve_index():
    """Serve the index.html file."""
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        filename = secure_filename(file.filename)
        
        if filename.endswith('.pdf'):
            try:
                pdf = PyPDF2.PdfReader(file)
                context = ''.join(page.extract_text() or '' for page in pdf.pages)
                return jsonify({'context': context[:10000]})
            except Exception as e:
                logger.error(f"PDF processing error: {e}")
                return jsonify({'error': 'Error processing PDF file'}), 400
                
        elif filename.endswith('.txt'):
            try:
                context = file.read().decode('utf-8')
                return jsonify({'context': context[:10000]})
            except Exception as e:
                logger.error(f"Text file processing error: {e}")
                return jsonify({'error': 'Error processing text file'}), 400
        
        # ✅ FIXED line below
        return jsonify({'error': 'Unsupported file format. Please upload a .pdf or .txt file.'}), 400

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'error': 'Unexpected error during file upload'}), 500

     
@app.route('/generate', methods=['POST'])
    
def generate_pdf_content(questions, topic):
    """Generate PDF content for download"""
    # Implementation remains similar to original
    # ...

# Flask routes



@cache.memoize(timeout=3600)
def get_wikipedia_summary(topic, sentences):
    if not wikipedia:
        return f"Summary for {topic} (Wikipedia module not available)"
    
    try:
        return wikipedia.summary(topic, sentences=sentences, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=sentences)
    except wikipedia.PageError:
        return f"Summary for {topic} (page not found)"
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
        return f"Summary for {topic} (error fetching)"

@app.route('/generate', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        topic = escape(data.get('topic', '').strip()[:500])
        if not topic:
            return jsonify({'error': 'Topic cannot be empty'}), 400
            
        qtype = data.get('qtype', 'MCQ')
        num_questions = min(max(int(data.get('num_questions', 5)), 1), 20)
        difficulty = data.get('difficulty', 'medium')
        context = data.get('context', '').strip()
        
        # Use Wikipedia if no custom context provided
        if not context and wikipedia:
            summary_sentences = {'easy': 5, 'medium': 10, 'hard': 20}.get(difficulty, 10)
            context = get_wikipedia_summary(topic, summary_sentences)
        
        # Validate context
        if not context or len(context) < 100:
            return jsonify({'error': 'Context is too short or missing. Please provide more content.'}), 400
            
        response_questions = []
        max_attempts = num_questions * 3
        attempts = 0
        generated_questions = set()
        
        # Generate questions until we have enough valid ones
        while len(response_questions) < num_questions and attempts < max_attempts:
            attempts += 1
            try:
                # Generate questions using explicit prompt
                input_text = f"generate questions: {context[:3000]}"  # Limit context size
                
                # Use max_new_tokens exclusively to avoid conflict
                question_output = nlp_models['qg_pipeline'](
                    input_text,
                    max_new_tokens=128,
                    do_sample=True,
                    top_k=50,
                    temperature=0.7,
                    num_return_sequences=1
                )
                
                # Process generated questions
                raw_questions = question_output[0].get('generated_text', '').strip()
                question_list = [clean_question(q) for q in raw_questions.split('<sep>')]
                question_list = [q for q in question_list if q]  # Remove empty
                
                for question_text in question_list:
                    if len(response_questions) >= num_questions:
                        break
                        
                    # Skip duplicates
                    if (question_text in generated_questions or 
                        not is_question_unique(question_text, generated_questions)):
                        continue
                    
                    # Get answer for this question
                    try:
                        answer_result = nlp_models['qa_pipeline'](
                            question=question_text, 
                            context=context,
                            max_answer_len=50
                        )
                    except Exception as e:
                        logger.warning(f"QA failed for: {question_text} - {e}")
                        continue
                        
                    answer = answer_result.get('answer', '').strip()
                    score = answer_result.get('score', 0.0)
                    
                    # Validate answer
                    if (not answer or score < 0.3 or 
                        len(answer) > 100 or len(answer) < 1):
                        continue
                    
                    # Add to generated set to prevent duplicates
                    generated_questions.add(question_text)
                    
                    # Create question object based on type
                    if qtype == 'MCQ':
                        # Generate distractors
                        if is_scientist_related_question(question_text):
                            distractors = generate_scientist_distractors(answer, 3)
                        elif "how many" in question_text.lower():
                            distractors = handle_numeric_answer(answer, 3)
                        else:
                            distractors = generate_distractors(answer, context, 3)
                            
                        options = distractors + [answer]
                        random.shuffle(options)
                        options = options[:4]  # Ensure exactly 4 options
                        
                        question_entry = {
                            'type': 'MCQ',
                            'question': question_text,
                            'options': options,
                            'answer': answer
                        }
                    else:  # Short Answer
                        question_entry = {
                            'type': 'Short Answer',
                            'question': question_text,
                            'answer': answer
                        }
                        
                    response_questions.append(question_entry)
                    
            except Exception as e:
                logger.error(f"Error generating questions (attempt {attempts}): {e}")
        
        if not response_questions:
            return jsonify({
                'error': 'Could not generate valid questions. Try providing more detailed context.',
                'context_sample': context[:300] + '...' if len(context) > 300 else context
            }), 400
            
        return jsonify({
            'questions': response_questions,
            'context': context[:500] + "..." if len(context) > 500 else context
        })
        
    except Exception as e:
        logger.error(f"Server error in /generate: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        topic = request.form.get('topic', 'Generated Quiz')
        questions = json.loads(request.form.get('questions', '[]'))
        
        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.cell(200, 10, txt=topic, ln=True, align='C')
        pdf.ln(10)
        
        # Add questions
        for i, q in enumerate(questions, 1):
            pdf.multi_cell(0, 10, f"{i}. {q['question']}", align='L')
            if q['type'] == 'MCQ':
                for j, opt in enumerate(q.get('options', [])):
                    pdf.multi_cell(0, 10, f"   {chr(97+j)}. {opt}", align='L')
            pdf.ln(5)
        
        # Send as response
        pdf_output = pdf.output(dest='S').encode('latin1')
        return send_file(
            io.BytesIO(pdf_output),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{topic.replace(' ', '_')}_quiz.pdf"
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return jsonify({'error': 'Failed to generate PDF'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)