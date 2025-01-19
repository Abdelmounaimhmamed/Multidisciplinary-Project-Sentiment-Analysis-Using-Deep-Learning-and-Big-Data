##Hna hadchi nadiiiiiiiiiiii
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import framenet as fn
import os
import ssl
import json 
import csv



nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')


        

def download_nltk_data():
    """Download required NLTK data packages with error handling."""
    try:
        # Handle SSL certificate verification issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Create nltk_data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)

        # Download required NLTK data
        required_packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'wordnet',
            'framenet_v17'
        ]
        
        for package in required_packages:
            try:
                nltk.download(package, quiet=True)
                print(f"Successfully downloaded {package}")
            except Exception as e:
                print(f"Error downloading {package}: {str(e)}")
                
    except Exception as e:
        print(f"Error during NLTK setup: {str(e)}")
        raise

class TweetSemanticAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with a CSV file containing tweets and labels."""
        # Ensure NLTK data is downloaded before processing
        download_nltk_data()
        
        try:
            self.df = pd.read_csv(csv_file)
            print(f"Successfully loaded CSV file with {len(self.df)} rows")
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            raise
        
    def save_results_to_json(self, results, output_file):
        """Sauvegarder les résultats dans un fichier JSON."""
        try:
            with open(output_file, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, ensure_ascii=False, indent=4)
            print(f"Results successfully saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to JSON: {str(e)}")
            
        
    def get_wordnet_info(self, word, pos=None):
        """Get WordNet information for a word."""
        try:
            synsets = wn.synsets(word, pos=pos)
            if not synsets:
                return None
            
            # Get the most common synset
            synset = synsets[0]
            
            return {
                'definition': synset.definition(),
                'synonyms': [lemma.name() for lemma in synset.lemmas()],
                'hypernyms': [hyper.name().split('.')[0] for hyper in synset.hypernyms()],
                'hyponyms': [hypo.name().split('.')[0] for hypo in synset.hyponyms()]
            }
        except Exception as e:
            print(f"Error in WordNet analysis for word '{word}': {str(e)}")
            return None
    
    def get_framenet_info(self, word):
        """Get FrameNet information for a word."""
        try:
            frames = fn.frames_by_lemma(word)
            if not frames:
                return None
            
            # Get the first frame (most relevant)
            frame = frames[0]
            
            return {
                'frame_name': frame.name,
                'frame_definition': frame.definition,
                'frame_elements': [fe.name for fe in frame.FE.values()]
            }
        except Exception as e:
            print(f"Error in FrameNet analysis for word '{word}': {str(e)}")
            return None
    
    def analyze_tweet(self, tweet):
        """Analyze a single tweet using WordNet and FrameNet."""
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(tweet)
            pos_tags = pos_tag(tokens)
            
            analysis = {
                'original_tweet': tweet,
                'tokens': tokens,
                'pos_tags': pos_tags,
                'wordnet_analysis': {},
                'framenet_analysis': {}
            }
            
            # Convert POS tags to WordNet format
            pos_map = {'JJ': 'a', 'VB': 'v', 'NN': 'n', 'RB': 'r'}
            
            # Analyze each word
            for word, pos in pos_tags:
                # Skip punctuation and common words
                if word.isalnum() and len(word) > 2:
                    # Get WordNet POS
                    wn_pos = pos_map.get(pos[:2])
                    
                    # WordNet analysis
                    wn_info = self.get_wordnet_info(word.lower(), wn_pos)
                    if wn_info:
                        analysis['wordnet_analysis'][word] = wn_info
                    
                    # FrameNet analysis
                    fn_info = self.get_framenet_info(word.lower())
                    if fn_info:
                        analysis['framenet_analysis'][word] = fn_info
            
            return analysis
        except Exception as e:
            print(f"Error analyzing tweet '{tweet}': {str(e)}")
            return None
    
    def analyze_dataset(self):
        """Analyze all tweets in the dataset."""
        analyses = []
        for _, row in self.df.iterrows():
            try:
                analysis = self.analyze_tweet(row['tweet'])
                if analysis:  # Only add successful analyses
                    analysis['label'] = row['label']
                    analyses.append(analysis)
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                continue
        return analyses
    
    
    def save_summary_to_csv(self, summary, filename):
        """Save the semantic summary to a CSV file."""
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Frame', 'Count', 'Sentiment'])
                
                # Save positive frames
                for frame, count in summary['positive_common_frames']:
                    writer.writerow([frame, count, 'positive'])
                
                # Save negative frames
                for frame, count in summary['negative_common_frames']:
                    writer.writerow([frame, count, 'negative'])
                    
            print(f"Summary successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving summary to CSV: {str(e)}")

    def get_semantic_summary(self, analyses):
        """Generate a summary of semantic patterns in the dataset."""
        try:
            positive_frames = {}
            negative_frames = {}
            
            for analysis in analyses:
                frames = analysis['framenet_analysis']
                label = analysis['label']
                
                target_dict = positive_frames if label == 'positive' else negative_frames
                
                for word, frame_info in frames.items():
                    frame_name = frame_info['frame_name']
                    target_dict[frame_name] = target_dict.get(frame_name, 0) + 1
            
            return {
                'positive_common_frames': sorted(positive_frames.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)[:5],
                'negative_common_frames': sorted(negative_frames.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)[:5]
            }
        except Exception as e:
            print(f"Error generating semantic summary: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    try:
        # Initialize analyzer
        analyzer = TweetSemanticAnalyzer('C:\\Users\\RPC\\Desktop\\CVtech\\BI\\tweets.csv')
        
        # Analyze all tweets
        print("Starting dataset analysis...")
        analyses = analyzer.analyze_dataset()
        
        if analyses:
            
            analyzer.save_results_to_json(analyses, 'C:\\Users\\RPC\\Desktop\\CVtech\\BI\\tweet_analysis_results.json')

            
            # Get semantic summary
            summary = analyzer.get_semantic_summary(analyses)
            
            # Print example analysis for first tweet
            print("\nExample analysis for first tweet:")
            print(f"Tweet: {analyses[0]['original_tweet']}")
            print("\nWordNet Analysis:")
            for word, info in analyses[0]['wordnet_analysis'].items():
                print(f"\n{word}:")
                print(f"Definition: {info['definition']}")
                print(f"Synonyms: {', '.join(info['synonyms'])}")
            
            print("\nFrameNet Analysis:")
            for word, info in analyses[0]['framenet_analysis'].items():
                print(f"\n{word}:")
                print(f"Frame: {info['frame_name']}")
                print(f"Definition: {info['frame_definition']}")
            
            if summary:
                analyzer.save_summary_to_csv(summary, 'C:\\Users\\RPC\\Desktop\\CVtech\\BI\\semantic_summary.csv')
                print("\nCommon semantic frames in dataset:")
                print("\nPositive tweets:")
                for frame, count in summary['positive_common_frames']:
                    print(f"{frame}: {count} occurrences")
                print("\nNegative tweets:")
                for frame, count in summary['negative_common_frames']:
                    print(f"{frame}: {count} occurrences")
        else:
            print("No analyses were generated. Please check the error messages above.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")