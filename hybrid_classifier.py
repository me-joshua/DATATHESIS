import re
from dataclasses import dataclass
from typing import Optional
from collections import Counter

@dataclass
class ClassificationResult:
    label: str    
    method: str   
    explanation: str = ""

class RuleBasedClassifier:
    # --- MATH SIGNALS ---
    MATH_NUMERALS = re.compile(r'[௧-௲0-9]')
    MATH_KW = ['எத்தனை', 'மொத்தம்', 'பாதி', 'இரண்டு', 'மூன்று', 'எண்ணிக்கை', 'എത്ര', 'ആകെ', 'പകുതി', 'രണ്ട്', 'മൂന്ന്', 'എണ്ണം']

    # --- CULTURAL SIGNALS ---
    CULT_STRONG = ['பொங்கல்', 'தீபாவளி', 'முருகன்', 'சிவன்', 'பிள்ளையார்', 'கோயில்', 'ഓണം', 'വിഷു', 'ശിവൻ', 'മുരുകൻ', 'ക്ഷേത്രം']
    CULT_WEAK = ['தேங்காய்', 'தென்னை', 'வாழை', 'நெல்', 'விளக்கு', 'மஞ்சள்', 'மழை', 'தெങ്ങ്', 'തേങ്ങ', 'വാഴ', 'നെല്ല്', 'മഴ']

    # --- WORDPLAY SIGNALS ---
    WORD_KW = ['இஷ்டம்', 'நஷ்டം', 'ஏதான', 'ആരാണ്', 'ഇഷ്ടം', 'നഷ്ടം', 'ഏതാണ്', 'ആരാണ്']
    
    # NEW: Common starters to IGNORE for phonetic/alliteration checks
    # These words repeat naturally and shouldn't count as "Wordplay"
    STOP_WORDS_START = ['ഇത്', 'എനിക്ക്', 'ഞാൻ', 'ഒരു', 'അത്', 'இது', 'எனக்கு', 'நான்', 'ஒரு', 'அது']

    def __init__(self):
        self.math_p = re.compile('|'.join(re.escape(k) for k in self.MATH_KW))
        self.cult_s_p = re.compile('|'.join(re.escape(k) for k in self.CULT_STRONG))
        self.cult_w_p = re.compile('|'.join(re.escape(k) for k in self.CULT_WEAK))
        self.word_p = re.compile('|'.join(re.escape(k) for k in self.WORD_KW))

    def _detect_phonetic_wordplay(self, text: str) -> float:
        """Smarter Alliteration Detection: Ignores common filler words."""
        words = text.split()
        if len(words) < 4: return 0.0
        
        # Filter out common starters before checking first-letter repetition
        filtered_starts = [w[0] for w in words if w not in self.STOP_WORDS_START and len(w) > 0]
        
        if not filtered_starts: return 0.0
        
        counts = Counter(filtered_starts)
        most_common_count = counts.most_common(1)[0][1]
        
        # Requires at least 3 non-stop-word repetitions to even get a 0.4
        if most_common_count >= 3:
            return 0.4 
        return 0.0

    def classify(self, text: str) -> Optional[ClassificationResult]:
        scores = {"Mathematical": 0.0, "Cultural": 0.0, "Wordplay": 0.0}

        # 1. Math: Needs Numeral + Keyword to hit 0.7
        has_num = bool(self.MATH_NUMERALS.search(text))
        math_hits = len(self.math_p.findall(text))
        scores["Mathematical"] = (0.4 if has_num else 0.0) + min(math_hits * 0.3, 0.5)

        # 2. Cultural: Strong keywords hit 0.7 immediately
        cult_s = len(self.cult_s_p.findall(text))
        cult_w = len(self.cult_w_p.findall(text))
        scores["Cultural"] = min(cult_s * 0.7 + cult_w * 0.4, 0.95)

        # 3. Wordplay: Keywords (0.5) + Phonetic Boost (0.4)
        word_hits = len(self.word_p.findall(text))
        phonetic_boost = self._detect_phonetic_wordplay(text)
        scores["Wordplay"] = min(word_hits * 0.5 + phonetic_boost, 0.9)

        # 4. COMPETITIVE PRIORITY LOGIC
        # If Cultural and Wordplay are close, CULTURAL wins (Metaphor > Puns)
        if scores["Cultural"] >= 0.4 and scores["Wordplay"] >= 0.4:
            scores["Wordplay"] -= 0.2

        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]

        # Final Threshold and Tie-Breaker
        sorted_vals = sorted(scores.values(), reverse=True)
        if sorted_vals[0] == sorted_vals[1] or top_score < 0.5:
            return None 

        return ClassificationResult(
            label=top_label, 
            method="Layer 1", 
            explanation=f"Score: {top_score:.2f}"
        )