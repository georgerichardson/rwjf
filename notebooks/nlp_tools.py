from sklearn.base import TransformerMixin
import gensim
from gensim.models.phrases import Phraser, Phrases


class SpacyTokenizer(TransformerMixin):
    """SpacyTokenizer
    
    Tokenizes a series of documents that have been parsed using spaCy.
    """
    
    def __init__(self, lower=True, remove_pos=['SYM','PUNCT'],
                 remove_stop=False, remove_numbers=False,
                 remove_brackets=False, ascii_only=False,
                 min_length=0):
        """__init__
    
        Args:
            lower (bool):
            remove_pos (bool):
            remove_stop (bool):
            remove_numbers (bool):
            remove_brackets (bool):
            ascii_only (bool):
            min_length:
    
        Returns:
        r
        """
        self.lower = lower
        self.remove_pos = remove_pos
        self.remove_stop = remove_stop
        self.remove_numbers = remove_numbers
        self.remove_brackets = remove_brackets
        self.ascii_only = ascii_only
        self.min_length = min_length
        
    def fit(self, docs, *args):
        return self
    
    def transform(self, docs, *args):
        lower = self.lower
        remove_pos = self.remove_pos
        remove_stop = self.remove_stop
        remove_numbers = self.remove_numbers
        remove_brackets = self.remove_brackets
        ascii_only = self.ascii_only
        min_length = self.min_length
        tokenized = []
        for d in docs:
            tokens = []
            for token in d:
                if token.pos_ in remove_pos:
                    continue
                if token.is_stop & remove_stop:
                    continue
                if token.like_num & remove_numbers:
                    continue
                if token.is_bracket & remove_brackets:
                    continue
                if ascii_only & (token.is_ascii == False):
                    continue
                if min_length & (len(token) <= min_length):
                    continue
                if lower:
                    tokens.append(token.lower_)
                else:
                    tokens.append(token.text)
            tokenized.append(tokens)
        return tokenized


class SpacyLemmatizer(TransformerMixin):
    """SpacyLemmatizer

    Lemmatizes a series of documents that have been parsed using spaCy.
    """
    
    def __init__(self, remove=['SYM', 'PUNCT']):
        self.remove = remove
        
    def fit(self, docs, *args):
        return self
    
    def transform(self, docs, *args):
        remove = self.remove
        lemmatized = []
        for d in docs:
            lemmas = []
            for token in d:
                if token.pos_ in remove:
                    continue
                else:
                    lemmas.append(token.lemma_)
            lemmatized.append(lemmas)
        return lemmatized


class GensimNGrammer(TransformerMixin):
    def __init__(self, n=3, **phrase_kwargs):
        self.n = n
        self.phrase_kwargs = phrase_kwargs
        
    def fit(self, texts, *args):
        n = self.n
        if n > 1:
            for _ in range(n - 1):
                ngrams = Phrases(texts)
                ngrammer = Phraser(ngrams)
                texts = ngrammer[texts]
            self.ngrams = ngrams
            self.ngrammer = ngrammer
        return self
        
    def ngram(self, texts):
        phrase_kwargs = self.phrase_kwargs
        n = self.n
        if n > 1:
            for _ in range(n - 1):
                ngrams = Phrases(texts, **phrase_kwargs)
                ngrammer = Phraser(ngrams)
                texts = ngrammer[texts]
            self.ngrams = ngrams
            self.ngrammer = ngrammer
        return list(texts)
        
    def transform(self, texts, *args):
        return self.ngram(texts)


class GensimTokenizer(TransformerMixin):
    def __init__(self, **tokenize_kwargs):
        self.tokenize_kwargs = tokenize_kwargs
        
    def tokenize(self, texts):
        tokens = []
        tokenize_kwargs = self.tokenize_kwargs
        for text in texts:
            
            # tokenize each message; simply lowercase & match alphabetic chars, for now
            # yield gensim.utils.tokenize(text, **tokenize_kwargs)
            tokens.append(list(gensim.utils.tokenize(text, **tokenize_kwargs)))
        return tokens
                
    def fit(self, texts, *args):
        return self

    def transform(self, texts, *args):
        return self.tokenize(texts) 


class GensimLemmatizer(TransformerMixin):
    """Lemmatizes a series of documents using Gensim's lemmatizing utility.
    
    Note: this requires the Python library 'pattern', which must be manually
    installed from the GitHub repo for Python 3.
    """
    def __init__(self, **lemmatize_kwargs):
        self.lemmatize_kwargs = lemmatize_kwargs
        
    def lemmatize(self, texts):
        lemmatize_kwargs = self.lemmatize_kwargs
        lemmas = []
        for text in texts:
            # tokenize each message; simply lowercase &
	    # match alphabetic chars, for now
            # yield gensim.utils.lemmatize(text, **lemmatize_kwargs)
            text = gensim.utils.lemmatize(text, **lemmatize_kwargs)
            lemmas.append(list(text))
                
    def fit(self, texts, *args):
        return self

    def transform(self, texts, *args):
        return self.lemmatize(texts)

class CleanText(TransformerMixin):
    """Encodes text and strips specified characters.
    
    Parameters
    ----------
    encoding: str
        Type of string encoding to use. (Default is 'latin1')
    remove_chars: list
        Group of characters that will be removed from the texts.
        (Default is ['\n', '\r'])
    log_every: int, optional
        Number of documents to be processed in a group to be logged.
        
    Returns
    -------
    text: iterator
        Generator that encodes and cleans text
    """
    
    def __init__(self, texts=None, encoding='utf-8',
		 remove_chars=['\n', '\r', '\t'],
                 clean_whitespace=True,
                 log_every=None):
        self.encoding = encoding
        self.remove_chars = remove_chars
        self.clean_whitespace = clean_whitespace
        self.log_every = log_every
        self.texts = texts
    
    def fit(self, texts, *args):
        return self
    
    def clean_chars(self, text):
        remove_chars = self.remove_chars 
        for char in remove_chars:
            text = text.replace(char, '')
        return text
    
    def encode(self, text, encoding):
        text = gensim.utils.to_unicode(text, encoding).strip()
        return text

    def remove_whitespace(self, text):
        text = ' '.join(text.split())
        return text
        
    def process(self, texts):
        clean_whitespace = self.clean_whitespace
        encoding = self.encoding
        log_every = self.log_every
        processed = 0
        cleaned = []
        for text in texts:
            # text = self.encode(text, encoding)
            text = self.clean_chars(text)
            if clean_whitespace:
                text = self.remove_whitespace(text)
            if log_every and processed % log_every == 0:
                    logging.info("Cleaned {} docs".format(processed))
            # yield text
            processed += 1
            cleaned.append(text)
        return cleaned
    
    def transform(self, texts, *args):
        return self.process(texts)
    
class BaseSearchGensim():
    
    def __init__(self, model, params):
        self.model = model
        self.params = params
    
    def explode_params(self, params):
        flat = [[(k, v) for v in vs] for k, vs in params.items()]
        param_combinations = [dict(items) for items in itertools.product(*flat)]
        return param_combinations

class CoherenceSearchGensim(BaseSearchGensim):
    
    def __init__(self, model, params, corpus, dictionary, texts):
        self.corpus = corpus
        self.dictionary = dictionary
        self.texts = texts
        super().__init__(model, params)
    
    def optimise_coherence(self):
        params = self.params
        model = self.model
        corpus = self.corpus
        dictionary = self.dictionary
        texts = self.texts
        
        self.params_tried = []
        self.models_tried = []
        self.coherence_values = []

        self.best_model = None
        
        param_combos = self.explode_params(params)

        for param_group in param_combos:
            model_temp = self.model(corpus=corpus, id2word=dictionary, **param_group)
            self.models_tried.append(model_temp)
            coherencemodel = CoherenceModel(model=model_temp, texts=texts, dictionary=dictionary, coherence='c_v')
            self.coherence_values.append(coherencemodel.get_coherence())
            if self.best_model is None:
                self.best_model = model_temp
            else:
                if self.coherence_values[-1] > coherencemodel.get_coherence():
                    self.best_model = model_temp
