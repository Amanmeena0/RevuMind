"""
Readability Metrics Utility
===========================
Provides helper functions to calculate Flesch Reading Ease and 
Flesch-Kincaid Grade Level of review texts.
"""

import re

def count_syllables_in_word(word: str) -> int:
    """
    Estimates the number of syllables in a word using basic English heuristics.
    """
    word = word.lower().strip()
    if not word:
        return 0
        
    # Remove punctuation
    word = re.sub(r'[^\w]', '', word)
    if not word:
        return 0
        
    # Simple exceptions dictionary
    exceptions = {
        "the": 1, "me": 1, "you": 1, "she": 1, "he": 1, "we": 1, "they": 1,
        "are": 1, "our": 1, "their": 1, "here": 1, "there": 1, "were": 1,
        "some": 1, "come": 1, "done": 1, "gone": 1, "one": 1, "two": 1
    }
    if word in exceptions:
        return exceptions[word]
        
    # Rule 1: Count vowels and contiguous vowel clusters
    vowels = "aeiouy"
    count = 0
    in_vowel_group = False
    
    for char in word:
        if char in vowels:
            if not in_vowel_group:
                count += 1
                in_vowel_group = True
        else:
            in_vowel_group = False
            
    # Rule 2: Exclude silent 'e' at the end of a word (e.g. 'game', 'line')
    if word.endswith('e') and count > 1:
        # Check if the 'e' is preceded by an 'l' + another consonant (like 'little', 'handle' -> keep 'e')
        if not (len(word) > 2 and word[-2] == 'l' and word[-3] not in vowels):
            count -= 1
            
    # Rule 3: Add extra syllable for ending 'es' and 'ed' if they form a sound
    # (very rough heuristic: check common suffixes)
    if word.endswith('ed') and not word.endswith('ted') and not word.endswith('ded') and count > 1:
        pass # standard 'ed' is silent e (like 'liked' -> 1 syl), so don't add count
        
    # Ensure every word has at least one syllable
    return max(1, count)

def calculate_readability(text: str) -> dict:
    """
    Calculates readability features for a given text.
    Returns a dictionary with:
    - flesch_reading_ease: Float (typically 0.0 to 100.0)
    - flesch_kincaid_grade: Float (grade level)
    - word_count: Int
    - sentence_count: Int
    - char_count: Int
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "word_count": 0,
            "sentence_count": 0,
            "char_count": 0
        }
        
    char_count = len(text)
    
    # Split sentences using common punctuation
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences)
    if sentence_count == 0:
        sentence_count = 1
        
    # Get all words
    words = [w.strip() for w in re.split(r'\s+', text) if w.strip()]
    word_count = len(words)
    if word_count == 0:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "word_count": 0,
            "sentence_count": sentence_count,
            "char_count": char_count
        }
        
    total_syllables = sum(count_syllables_in_word(w) for w in words)
    
    # Ratios
    avg_sentence_len = word_count / sentence_count
    avg_syllables_per_word = total_syllables / word_count
    
    # Flesch Reading Ease formula
    # FRE = 206.835 - (1.015 * ASL) - (84.6 * ASW)
    fre = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables_per_word)
    fre = max(0.0, min(100.0, fre)) # Clamp to standard bounds
    
    # Flesch-Kincaid Grade Level formula
    # FKGL = (0.39 * ASL) + (11.8 * ASW) - 15.59
    fkgl = (0.39 * avg_sentence_len) + (11.8 * avg_syllables_per_word) - 15.59
    fkgl = max(0.0, fkgl) # Clamp lower bound to 0
    
    return {
        "flesch_reading_ease": float(round(fre, 2)),
        "flesch_kincaid_grade": float(round(fkgl, 2)),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": char_count
    }

if __name__ == "__main__":
    test_text = "This product is absolutely amazing! The battery life is exceptional and lasted me two full days. I would highly recommend it."
    metrics = calculate_readability(test_text)
    print("Test text metrics:", metrics)
