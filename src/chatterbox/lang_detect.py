import re
from typing import Tuple, List, Dict

SUPPORTED_LANGUAGES = {
    'ar', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'he', 'hi', 
    'it', 'ja', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 
    'sw', 'tr', 'zh'
}

# Stopwords for Latin script languages
STOPWORDS = {
    'en': {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she'},
    'tr': {'ve', 'bir', 'için', 'bu', 'da', 'çok', 'ile', 'daha', 'en', 'ne', 'kadar', 'var', 'gibi', 'de', 'ama', 'o', 'ben', 'iyi', 'onu', 'sen', 'bana', 'bunu', 'her', 'yok', 'benim', 'hayır', 'evet', 'diye', 'zaman', 'neden'},
    'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'die', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass', 'sie'},
    'fr': {'le', 'la', 'les', 'de', 'et', 'en', 'à', 'un', 'une', 'que', 'qui', 'pour', 'dans', 'des', 'du', 'il', 'est', 'pas', 'sur', 'ce', 'par', 'je', 'ne', 'au', 'avec', 'se', 'son', 'sa', 'ou', 'y'},
    'es': {'el', 'la', 'los', 'las', 'de', 'y', 'en', 'a', 'un', 'una', 'que', 'por', 'para', 'con', 'no', 'su', 'se', 'es', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque'},
    'it': {'il', 'la', 'i', 'le', 'di', 'e', 'in', 'a', 'un', 'una', 'che', 'per', 'con', 'non', 'un', 'una', 'su', 'è', 'al', 'lo', 'come', 'ma', 'più', 'se', 'o', 'da', 'ci', 'mi', 'ha', 'questo'},
    'pt': {'o', 'a', 'os', 'as', 'de', 'e', 'em', 'um', 'uma', 'que', 'por', 'para', 'com', 'não', 'seu', 'sua', 'é', 'ao', 'do', 'da', 'como', 'mas', 'mais', 'se', 'ou', 'já', 'na', 'no', 'me', 'nos'},
    'nl': {'de', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zou', 'of', 'wat', 'mijn'},
    'sv': {'och', 'det', 'att', 'i', 'en', 'jag', 'hon', 'som', 'han', 'på', 'den', 'med', 'var', 'sig', 'för', 'så', 'till', 'är', 'men', 'ett', 'om', 'hade', 'de', 'av', 'icke', 'mig', 'du', 'har', 'vi', 'nu'},
    'da': {'og', 'i', 'jeg', 'det', 'at', 'en', 'den', 'til', 'er', 'som', 'på', 'de', 'med', 'han', 'af', 'for', 'ikke', 'der', 'var', 'mig', 'sig', 'men', 'et', 'har', 'om', 'vi', 'min', 'havde', 'ham', 'hun'},
    'no': {'og', 'i', 'jeg', 'det', 'at', 'en', 'et', 'den', 'til', 'er', 'som', 'på', 'de', 'med', 'han', 'av', 'ikke', 'for', 'der', 'var', 'meg', 'seg', 'men', 'har', 'om', 'vi', 'min', 'hadde', 'ham', 'hun'},
    'fi': {'ja', 'on', 'ei', 'niin', 'että', 'se', 'hän', 'mutta', 'ovat', 'kuin', 'myös', 'kun', 'tai', 'jos', 'olla', 'voi', 'vain', 'sen', 'mitä', 'hyvä', 'nyt', 'yli', 'jotta', 'me', 'mä', 'sä', 'ne', 'tämä', 'tuo', 'nuo'},
    'pl': {'i', 'w', 'na', 'z', 'do', 'nie', 'że', 'o', 'to', 'jak', 'a', 'co', 'jest', 'się', 'od', 'za', 'po', 'ale', 'tym', 'czy', 'tak', 'by', 'już', 'dla', 'tu', 'mi', 'ty', 'że', 'mu', 'ich'},
    'ms': {'dan', 'yang', 'itu', 'ini', 'di', 'pada', 'untuk', 'dengan', 'ada', 'dari', 'tidak', 'ke', 'oleh', 'dalam', 'akan', 'juga', 'telah', 'kepada', 'atau', 'mereka', 'sebagai', 'ia', 'lebih', 'boleh', 'satu', 'kita', 'saya', 'orang', 'banyak', 'kami'},
    'sw': {'na', 'ya', 'wa', 'kwa', 'katika', 'ni', 'za', 'kuwa', 'kama', 'ambayo', 'au', 'la', 'hii', 'pia', 'kama', 'ili', 'tu', 'hilo', 'vya', 'kwamba', 'sana', 'yake', 'lakini', 'hiyo', 'mimi', 'kila', 'sisi', 'yao', 'huyu', 'hata'}
}

class LanguageDetector:
    """
    A class for automatic language detection of text input using 
    script analysis and word frequencies (stopwords), with no external dependencies.
    """
    
    def __init__(self):
        self.supported_langs = SUPPORTED_LANGUAGES
        
    def is_supported(self, lang: str) -> bool:
        """
        Checks if a language is in the supported languages list.
        
        Args:
            lang: ISO 639-1 language code string
            
        Returns:
            True if language is supported, False otherwise
        """
        return lang in self.supported_langs
        
    def detect_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Detects the language of the text and returns a confidence score.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple containing the ISO 639-1 language code and confidence (0.0 to 1.0)
        """
        if not text or not isinstance(text, str):
            return 'en', 1.0
            
        text = text.strip()
        if len(text) < 10:
            return 'en', 0.5  # Default for very short text
            
        # 1. Check Scripts via Unicode ranges
        # Hiragana/Katakana for Japanese
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja', 0.9
            
        # Hangul for Korean
        if re.search(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]', text):
            return 'ko', 0.9
            
        # Hanzi for Chinese (simplified & traditional)
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh', 0.9
            
        # Arabic script (Arabic, Farsi, Urdu etc. - we map to ar)
        if re.search(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]', text):
            return 'ar', 0.9
            
        # Hebrew script
        if re.search(r'[\u0590-\u05ff]', text):
            return 'he', 0.9
            
        # Cyrillic (Russian)
        if re.search(r'[\u0400-\u04ff]', text):
            return 'ru', 0.9
            
        # Greek
        if re.search(r'[\u0370-\u03ff]', text):
            return 'el', 0.9
            
        # Devanagari (Hindi)
        if re.search(r'[\u0900-\u097f]', text):
            return 'hi', 0.9
            
        # 2. For Latin script, fallback to Stopwords analysis
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 'en', 0.5
            
        scores = {lang: 0 for lang in STOPWORDS.keys()}
        total_matched = 0
        
        for word in words:
            for lang, stops in STOPWORDS.items():
                if word in stops:
                    scores[lang] += 1
                    total_matched += 1
                    
        if total_matched == 0:
            return 'en', 0.5
            
        # Find the language with highest score
        best_lang = max(scores, key=scores.get)
        max_score = scores[best_lang]
        
        confidence = min(0.99, max_score / len(words) * 2.0) 
        if max_score == 0:
            return 'en', 0.5
            
        return best_lang, float(confidence)

    def detect(self, text: str) -> str:
        """
        Detects the language of the text.
        
        Args:
            text: Input text string
            
        Returns:
            ISO 639-1 language code (str)
        """
        lang, _ = self.detect_with_confidence(text)
        return lang

def detect_language(text: str) -> str:
    """
    Convenience module-level function to detect language.
    
    Args:
        text: Input text string
        
    Returns:
        ISO 639-1 language code (str)
    """
    detector = LanguageDetector()
    return detector.detect(text)
