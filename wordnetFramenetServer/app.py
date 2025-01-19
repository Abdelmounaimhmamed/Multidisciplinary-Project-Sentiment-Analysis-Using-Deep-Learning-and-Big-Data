# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import framenet as fn
import ssl
import json

app = Flask(__name__)
CORS(app)

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('framenet_v17')

def get_wordnet_info(word, pos=None):
    """Get WordNet information for a word."""
    try:
        synsets = wn.synsets(word, pos=pos)
        if not synsets:
            return None
        
        synset = synsets[0]
        return {
            'definition': synset.definition(),
            'synonyms': [lemma.name() for lemma in synset.lemmas()],
            'hypernyms': [hyper.name().split('.')[0] for hyper in synset.hypernyms()],
            'hyponyms': [hypo.name().split('.')[0] for hypo in synset.hyponyms()]
        }
    except Exception as e:
        print(f"Error in WordNet analysis: {str(e)}")
        return None

def get_framenet_info(word):
    """Get FrameNet information for a word."""
    try:
        frames = fn.frames_by_lemma(word)
        if not frames:
            return None
        
        frame = frames[0]
        return {
            'frame_name': frame.name,
            'frame_definition': frame.definition,
            'frame_elements': [fe.name for fe in frame.FE.values()]
        }
    except Exception as e:
        print(f"Error in FrameNet analysis: {str(e)}")
        return None

def analyze_tweet(tweet):
    """Analyze a single tweet using WordNet and FrameNet."""
    try:
        tokens = word_tokenize(tweet)
        pos_tags = pos_tag(tokens)
        
        analysis = {
            'original_tweet': tweet,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'wordnet_analysis': {},
            'framenet_analysis': {}
        }
        
        pos_map = {'JJ': 'a', 'VB': 'v', 'NN': 'n', 'RB': 'r'}
        
        for word, pos in pos_tags:
            if word.isalnum() and len(word) > 2:
                wn_pos = pos_map.get(pos[:2])
                
                wn_info = get_wordnet_info(word.lower(), wn_pos)
                if wn_info:
                    analysis['wordnet_analysis'][word] = wn_info
                
                fn_info = get_framenet_info(word.lower())
                if fn_info:
                    analysis['framenet_analysis'][word] = fn_info
        
        return analysis
    except Exception as e:
        print(f"Error analyzing tweet: {str(e)}")
        return None

@app.route('/tweets', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        tweet = data.get('tweet')
        
        if not tweet:
            return jsonify({'error': 'No tweet provided'}), 400
        
        analysis = analyze_tweet(tweet)
        if analysis:
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Analysis failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)