import re
from typing import Dict, List, Tuple

class TextNormalizer:
    """Base class for text normalization before TTS synthesis."""
    
    def normalize(self, text: str) -> str:
        """
        Normalize the given text.
        
        Args:
            text (str): The raw input text.
            
        Returns:
            str: The normalized speakable text.
        """
        # Basic cleanup: normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class TurkishTextNormalizer(TextNormalizer):
    """Text normalizer for Turkish language."""
    
    def __init__(self):
        self.units = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
        self.tens = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
        self.scales = ["", "bin", "milyon", "milyar", "trilyon"]
        
        self.abbreviations = {
            r'\bDr\.': 'Doktor',
            r'\bProf\.': 'Profesör',
            r'\bDoç\.': 'Doçent',
            r'\bAv\.': 'Avukat',
            r'\bMüh\.': 'Mühendis',
            r'\bvb\.': 've benzeri',
            r'\bvs\.': 've saire',
            r'\byr\.': 'yardımcı',
            r'\bHz\.': 'Hazreti',
        }
        
        self.ordinals = {
            "1": "birinci", "2": "ikinci", "3": "üçüncü", "4": "dördüncü", 
            "5": "beşinci", "6": "altıncı", "7": "yedinci", "8": "sekizinci", 
            "9": "dokuzuncu", "10": "onuncu", "20": "yirminci", "30": "otuzuncu",
            "40": "kırkıncı", "50": "ellinci", "60": "altmışıncı", "70": "yetmişinci",
            "80": "sekseninci", "90": "doksanıncı", "100": "yüzüncü"
        }
        
        self.months = {
            "01": "ocak", "02": "şubat", "03": "mart", "04": "nisan",
            "05": "mayıs", "06": "haziran", "07": "temmuz", "08": "ağustos",
            "09": "eylül", "10": "ekim", "11": "kasım", "12": "aralık",
            "1": "ocak", "2": "şubat", "3": "mart", "4": "nisan",
            "5": "mayıs", "6": "haziran", "7": "temmuz", "8": "ağustos",
            "9": "eylül"
        }

    def num2words(self, n: int) -> str:
        if n == 0:
            return "sıfır"
        
        if n < 0:
            return "eksi " + self.num2words(-n)
            
        words = []
        scale_idx = 0
        
        while n > 0:
            chunk = n % 1000
            n = n // 1000
            
            if chunk > 0:
                chunk_words = []
                hundreds = chunk // 100
                remainder = chunk % 100
                
                if hundreds == 1:
                    chunk_words.append("yüz")
                elif hundreds > 1:
                    chunk_words.append(self.units[hundreds] + " yüz")
                    
                t = remainder // 10
                u = remainder % 10
                
                if t > 0:
                    chunk_words.append(self.tens[t])
                if u > 0:
                    chunk_words.append(self.units[u])
                    
                chunk_str = " ".join(chunk_words)
                
                if scale_idx == 1 and chunk == 1:
                    words.insert(0, "bin")
                else:
                    if scale_idx > 0:
                        chunk_str += " " + self.scales[scale_idx]
                    words.insert(0, chunk_str)
                    
            scale_idx += 1
            
        return " ".join(words).strip()

    def _replace_numbers(self, match) -> str:
        num_str = match.group(0)
        # Handle decimal
        if ',' in num_str or '.' in num_str:
            num_str = num_str.replace(',', '.')
            if '.' in num_str:
                parts = num_str.split('.')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    return f"{self.num2words(int(parts[0]))} virgül {self.num2words(int(parts[1]))}"
        
        if num_str.isdigit():
            return self.num2words(int(num_str))
        return num_str

    def _replace_ordinals(self, match) -> str:
        num = match.group(1)
        if num in self.ordinals:
            return self.ordinals[num]
        
        # Fallback for complex ordinals, just pronounce number + 'inci' roughly
        return self.num2words(int(num)) + " inci"

    def _replace_currency(self, match) -> str:
        curr = match.group(1)
        amt = match.group(2)
        
        amt_words = self._replace_numbers(re.match(r'[\d.,]+', amt))
        
        if curr == '₺' or curr.lower() == 'tl':
            return amt_words + " lira"
        elif curr == '$':
            return amt_words + " dolar"
        elif curr == '€':
            return amt_words + " euro"
        elif curr == '£':
            return amt_words + " sterlin"
            
        return amt_words

    def _replace_dates(self, match) -> str:
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)
        
        day_word = self.num2words(int(day))
        month_word = self.months.get(month, month)
        year_word = self.num2words(int(year))
        
        return f"{day_word} {month_word} {year_word}"

    def _replace_time(self, match) -> str:
        hour = match.group(1)
        minute = match.group(2)
        
        hour_word = self.num2words(int(hour))
        if int(minute) == 0:
            return hour_word
        minute_word = self.num2words(int(minute))
        
        return f"{hour_word} {minute_word}"

    def normalize(self, text: str) -> str:
        # Time
        text = re.sub(r'\b(\d{1,2}):(\d{2})\b', self._replace_time, text)
        
        # Dates (DD.MM.YYYY or DD/MM/YYYY)
        text = re.sub(r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b', self._replace_dates, text)
        
        # Currency
        text = re.sub(r'([₺$€£])\s*([\d.,]+)', self._replace_currency, text)
        text = re.sub(r'([\d.,]+)\s*([₺$€£])', lambda m: self._replace_currency(re.match(r'([₺$€£])([\d.,]+)', m.group(2) + m.group(1))), text)
        
        # Emails
        text = text.replace('@', ' et ')
        
        # URLs (simple approximation)
        text = re.sub(r'\b(www\.)?([a-zA-Z0-9-]+)\.([a-zA-Z]{2,})\b', lambda m: f"{'dabılyu dabılyu dabılyu nokta ' if m.group(1) else ''}{m.group(2)} nokta {m.group(3)}", text)
        
        # Abbreviations
        for abbr, full in self.abbreviations.items():
            text = re.sub(abbr, full, text)
            
        # Ordinals
        text = re.sub(r'\b(\d+)\.', self._replace_ordinals, text)
        
        # Decimals & Numbers
        text = re.sub(r'\b\d+[,.]\d+\b', self._replace_numbers, text)
        text = re.sub(r'\b\d+\b', self._replace_numbers, text)
        
        # Cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class EnglishTextNormalizer(TextNormalizer):
    """Text normalizer for English language."""
    
    def __init__(self):
        self.units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", 
                      "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.scales = ["", "thousand", "million", "billion", "trillion"]
        
        self.abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bProf\.': 'Professor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bSt\.': 'Saint',
            r'\betc\.': 'et cetera',
            r'\bvs\.': 'versus',
        }
        
        self.ordinals = {
            "1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth",
            "6": "sixth", "7": "seventh", "8": "eighth", "9": "ninth", "10": "tenth",
            "11": "eleventh", "12": "twelfth", "13": "thirteenth", "20": "twentieth",
            "30": "thirtieth"
        }
        
        self.months = {
            "01": "January", "02": "February", "03": "March", "04": "April",
            "05": "May", "06": "June", "07": "July", "08": "August",
            "09": "September", "10": "October", "11": "November", "12": "December",
            "1": "January", "2": "February", "3": "March", "4": "April",
            "5": "May", "6": "June", "7": "July", "8": "August",
            "9": "September"
        }

    def num2words(self, n: int) -> str:
        if n == 0:
            return "zero"
        
        if n < 0:
            return "minus " + self.num2words(-n)
            
        words = []
        scale_idx = 0
        
        while n > 0:
            chunk = n % 1000
            n = n // 1000
            
            if chunk > 0:
                chunk_words = []
                hundreds = chunk // 100
                remainder = chunk % 100
                
                if hundreds > 0:
                    chunk_words.append(self.units[hundreds] + " hundred")
                    
                if remainder > 0:
                    if remainder < 20:
                        chunk_words.append(self.units[remainder])
                    else:
                        t = remainder // 10
                        u = remainder % 10
                        if u == 0:
                            chunk_words.append(self.tens[t])
                        else:
                            chunk_words.append(self.tens[t] + " " + self.units[u])
                            
                chunk_str = " ".join(chunk_words)
                
                if scale_idx > 0:
                    chunk_str += " " + self.scales[scale_idx]
                words.insert(0, chunk_str)
                
            scale_idx += 1
            
        return " ".join(words).strip()
        
    def year2words(self, n: int) -> str:
        if 1100 <= n <= 9999:
            thousands = n // 100
            remainder = n % 100
            if remainder == 0:
                return self.num2words(thousands) + " hundred"
            elif remainder < 10:
                return self.num2words(thousands) + " oh " + self.num2words(remainder)
            else:
                return self.num2words(thousands) + " " + self.num2words(remainder)
        return self.num2words(n)

    def _replace_numbers(self, match) -> str:
        num_str = match.group(0)
        # Handle decimal
        if '.' in num_str:
            parts = num_str.split('.')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                decimals = " ".join([self.num2words(int(d)) for d in parts[1]])
                return f"{self.num2words(int(parts[0]))} point {decimals}"
                
        if num_str.isdigit():
            val = int(num_str)
            if 1900 <= val <= 2099:
                return self.year2words(val)
            return self.num2words(val)
        return num_str

    def _replace_currency(self, match) -> str:
        curr = match.group(1)
        amt = match.group(2)
        
        amt_words = self._replace_numbers(re.match(r'[\d.,]+', amt))
        
        if curr == '$':
            return amt_words + " dollars"
        elif curr == '€':
            return amt_words + " euros"
        elif curr == '£':
            return amt_words + " pounds"
            
        return amt_words

    def _replace_dates(self, match) -> str:
        month = match.group(1)
        day = match.group(2)
        year = match.group(3)
        
        month_word = self.months.get(month, month)
        
        day_val = day.lstrip('0')
        day_word = self.ordinals.get(day_val, self.num2words(int(day_val)) + "th")
        
        year_word = self.year2words(int(year))
        
        return f"{month_word} {day_word}, {year_word}"

    def normalize(self, text: str) -> str:
        # Dates (MM/DD/YYYY)
        text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', self._replace_dates, text)
        
        # Currency
        text = re.sub(r'([$€£])\s*([\d.,]+)', self._replace_currency, text)
        
        # Emails
        text = text.replace('@', ' at ')
        
        # URLs (simple approximation)
        text = re.sub(r'\b(www\.)?([a-zA-Z0-9-]+)\.([a-zA-Z]{2,})\b', lambda m: f"{'w w w dot ' if m.group(1) else ''}{m.group(2)} dot {m.group(3)}", text)
        
        # Abbreviations
        for abbr, full in self.abbreviations.items():
            text = re.sub(abbr, full, text)
            
        # Decimals & Numbers
        text = re.sub(r'\b\d+\.\d+\b', self._replace_numbers, text)
        text = re.sub(r'\b\d+\b', self._replace_numbers, text)
        
        # Cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def get_normalizer(lang: str) -> TextNormalizer:
    """
    Factory function to get the appropriate text normalizer for the given language.
    
    Args:
        lang (str): The language code (e.g., 'en', 'tr').
        
    Returns:
        TextNormalizer: The text normalizer instance for the specified language.
    """
    lang = lang.lower()
    if lang == 'tr':
        return TurkishTextNormalizer()
    elif lang == 'en':
        return EnglishTextNormalizer()
    else:
        return TextNormalizer()
