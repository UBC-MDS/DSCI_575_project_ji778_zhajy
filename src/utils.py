import string

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now",
}


def tokenize_text(text: str) -> list[str]:
    """Lowercase text, remove punctuation, and drop stopwords."""
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()

    tokens = [token for token in tokens if token not in STOPWORDS]

    return tokens