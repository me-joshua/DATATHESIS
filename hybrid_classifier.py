import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClassificationResult:
    label: str    
    method: str   
    explanation: str = ""

class RuleBasedClassifier:
    # --- MATH SIGNALS ---
    MATH_NUMERALS = re.compile(r'[௧-௲0-9]')
    TAM_MATH_KW = ['கூட்டல்', 'கழித்தல்', 'பெருக்கல்', 'வகுத்தல்', 'எத்தனை', 'மொத்தம்', 'பாதி', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது', 'பத்து', 'நூறு']
    MAL_MATH_KW = ['കൂട്ടൽ', 'കുറയ്ക്കൽ', 'ഗുണനം', 'ഹരണം', 'എത്ര', 'ആകെ', 'പകുതി', 'ഒന്ന്', 'രണ്ട്', 'മൂന്ന്', 'നാല്', 'അഞ്ച്', 'ആറ്', 'ഏഴ്', 'എട്ട്', 'ഒൻപത്', 'പത്ത്', 'നൂറ്']

    # --- CULTURAL SIGNALS ---
    TAM_CULTURAL_KW = ['பொங்கல்', 'தீபாவளி', 'நவராத்திரி', 'முருகன்', 'கணேசன்', 'அம்மன்', 'சிவன்', 'விஷ்ணு', 'லட்சுமி', 'சரஸ்வதி', 'கோயில்', 'பூஜை', 'விளக்கு', 'மஞ்சள்', 'மழை', 'ஆறு']
    MAL_CULTURAL_KW = ['ഓണം', 'വിഷു', 'നവരാത്രി', 'മുരുകൻ', 'ഗണപതി', 'ഭഗവതി', 'ശിവൻ', 'വിഷ്ണു', 'ലക്ഷ്മി', 'സരസ്വതി', 'അമ്പലം', 'ക്ഷേത്രം', 'പൂജ', 'നിലവിളക്ക്', 'മഴ', 'പുഴ']

    # --- WORDPLAY SIGNALS ---
    # These often include specific question formats or pun indicators
    WORDPLAY_KW = ['இஷ்டம்', 'நஷ்டம்', 'ஏതാണ്', 'ஆരാണ്', 'ഇഷ്ടം', 'നഷ്ടം', 'ഏതാണ്', 'ആരാണ്', 'അപ്പം', 'കറി']

    def __init__(self):
        # Compile patterns for all categories
        self.math_kw_p = re.compile('|'.join(re.escape(k) for k in self.TAM_MATH_KW + self.MAL_MATH_KW))
        self.cult_kw_p = re.compile('|'.join(re.escape(k) for k in self.TAM_CULTURAL_KW + self.MAL_CULTURAL_KW))
        self.word_kw_p = re.compile('|'.join(re.escape(k) for k in self.WORDPLAY_KW))

    def classify(self, text: str) -> Optional[ClassificationResult]:
        scores = {
            "Mathematical": 0.0,
            "Cultural": 0.0,
            "Wordplay": 0.0
        }

        # 1. Score Mathematical
        has_numeral = bool(self.MATH_NUMERALS.search(text))
        math_hits = len(self.math_kw_p.findall(text))
        # A numeral gives 0.5 (not enough to trigger 0.6 alone). Keywords push it over.
        scores["Mathematical"] = (0.5 if has_numeral else 0.0) + min(math_hits * 0.3, 0.4)

        # 2. Score Cultural
        cult_hits = len(self.cult_kw_p.findall(text))
        # Cultural keywords are very strong identifiers
        scores["Cultural"] = min(cult_hits * 0.4, 0.9)

        # 3. Score Wordplay
        word_hits = len(self.word_kw_p.findall(text))
        scores["Wordplay"] = min(word_hits * 0.5, 0.9)

        # 4. Final Decision Logic
        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]

        # THRESHOLD CHECK:
        # If the score is high (>= 0.6), Layer 1 handles it.
        # If the score is low (e.g., 0.5), it returns None so Layer 2 (RAG) can judge.
        if top_score >= 0.6:
            return ClassificationResult(
                label=top_label, 
                method="Layer 1 (Weighted Keywords)", 
                explanation=f"Strong {top_label} signal detected (Score: {top_score:.2f})"
            )

        return None