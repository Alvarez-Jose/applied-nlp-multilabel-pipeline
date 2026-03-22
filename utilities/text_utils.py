'''
Text cleaning and extraction utilities for news articles.
'''
import re
from typing import Optional, List

def clean_article_text(text: str) -> str:
    '''
    clean article text by removing boilerplate and normalizing whitespace
    '''
    if not text:
        return ""

    # Strip Unicode replacement characters (U+FFFD) left by encoding errors in the scraper
    text = text.replace("\ufffd", "").replace("\x00", "")

    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # common boilerplate patterns for new sites
    boilerplate_patterns = [
        # WAFA-specific patterns
        r'Follow WAFA on.*',
        r'Share this article.*',
        r'Read also:.*',
        r'To receive WAFA.*',
        r'Copyright ©.*',
        r'All rights reserved.*',
        
        # General news patterns
        r'Share this.*',
        r'Follow us on.*',
        r'Read more:.*',
        r'Source:.*',
        r'Published on.*',
        r'Last updated:.*',
        r'Last modified:.*',
        r'Photo:.*',
        r'Also read:.*',
        r'Related:.*',
        r'Tags:.*',
        r'Category:.*',
        r'Comments:.*',
        r'Sign up.*',
        r'Subscribe.*',
        r'Newsletter.*',
        r'Click here.*',
        r'Watch:.*',
        r'Video:.*',
        r'Audio:.*',
        r'Advertisement.*',
        r'Sponsored.*',
    ]
    
    # Remove each boilerplate pattern
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Trim overall whitespace
    text = text.strip()
    
    # Remove any remaining empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text


def extract_title(text: str, max_length: int = 200) -> Optional[str]:
    """
    Extract a title from article text.
    This is a fallback when the article doesn't have a proper title.
    
    Args:
        text: Article text
        max_length: Maximum title length
    
    Returns:
        Extracted title or None
    """
    if not text:
        return None
    
    # Clean the text first
    clean_text = text.strip()
    
    # Try to find the first sentence ending with period
    sentences = re.split(r'[.!?]+', clean_text)
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 20 and len(first_sentence) < max_length:
            return first_sentence
    
    # Try to find the first line (if text has line breaks)
    lines = clean_text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 20 and len(line) < max_length:
            return line
    
    # Fallback: first 200 characters that look like a title
    if len(clean_text) > 20:
        # Find a reasonable breaking point (end of first "sentence")
        first_period = clean_text.find('.')
        if first_period > 20:
            return clean_text[:min(first_period + 1, max_length)].strip()
        else:
            return clean_text[:max_length].strip()
    
    return None


def extract_location(text: str) -> Optional[str]:
    """
    Extract location mentions from text.
    Focus on Palestinian locations.
    
    Args:
        text: Article text
    
    Returns:
        Comma-separated location mentions or None
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Common Palestinian locations (expand this list as needed)
    palestinian_locations = [
        # Governorates
        'hebron', 'bethlehem', 'jerusalem', 'ramallah', 'jenin',
        'nablus', 'tulkarm', 'qalqilya', 'salfit', 'tubas', 'jericho',
        
        # Regions
        'west bank', 'gaza', 'gaza strip', 'east jerusalem',
        
        # Major cities/towns
        'al-khalil', 'al-khalīl', 'al khalil',  # Hebron in Arabic
        'beit jala', 'beit sahour', 'beit lahia',
        'rafah', 'khan yunis', 'deir al-balah',
        'al-bireh', 'abu dis', 'al-azzariya',
        'yatta', 'dura', 'halhul', 'bani naim',
        'surif', 'beit ummar', 'sa\'ir',
        
        # Specific areas/neighborhoods
        'old city', 'h2 area', 'h1 area',
        'settlements', 'outposts', 'checkpoints',
        'ibrahimi mosque', 'al-aqsa', 'temple mount',
        
        # Refugee camps
        'arroub camp', 'fawwar camp', 'deheisheh camp',
        'aqabat jabr camp', 'jenin camp', 'balata camp',
        'askar camp', 'tulkarm camp',
    ]
    
    found_locations = []
    
    # Check for each location
    for location in palestinian_locations:
        if location.lower() in text_lower:
            # Capitalize first letter of each word for consistency
            formatted = ' '.join(word.capitalize() for word in location.split())
            found_locations.append(formatted)
    
    # Remove duplicates while preserving order
    unique_locations = []
    for loc in found_locations:
        if loc not in unique_locations:
            unique_locations.append(loc)
    
    # Return comma-separated list (max 3 locations)
    if unique_locations:
        return ', '.join(unique_locations[:3])
    
    return None


def extract_actors(text: str) -> Optional[str]:
    """
    Extract actor mentions from text (e.g., IDF, settlers, Palestinians).
    
    Args:
        text: Article text
    
    Returns:
        Comma-separated actor mentions or None
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Common actors in Palestinian context
    actors = [
        # Israeli forces
        'idf', 'israeli forces', 'israeli army', 'israeli soldiers',
        'israeli police', 'border police', 'israeli security forces',
        'israeli authorities',
        
        # Settlers
        'settlers', 'israeli settlers', 'jewish settlers',
        
        # Palestinians
        'palestinians', 'palestinian', 'palestinian residents',
        'palestinian youth', 'palestinian protesters',
        'palestinian farmers', 'palestinian workers',
        
        # Palestinian authorities
        'palestinian authority', 'pa forces', 'palestinian police',
        
        # International/other
        'activists', 'demonstrators', 'protesters',
        'international activists', 'human rights workers',
        'un staff', 'ocha staff', 'ngo workers',
        
        # Militant groups (less common in WAFA but included for completeness)
        'hamas', 'fatah', 'islamic jihad', 'pflp', 'dflp',
    ]
    
    found_actors = []
    
    # Check for each actor (case-insensitive)
    for actor in actors:
        if actor.lower() in text_lower:
            # Capitalize appropriately
            if actor == 'idf':
                formatted = 'IDF'
            elif actor == 'pa':
                formatted = 'PA'
            elif actor in ['hamas', 'fatah', 'pflp', 'dflp']:
                formatted = actor.upper()
            else:
                formatted = actor.title()
            
            found_actors.append(formatted)
    
    # Remove duplicates while preserving order
    unique_actors = []
    for actor in found_actors:
        if actor not in unique_actors:
            unique_actors.append(actor)
    
    # Return comma-separated list (max 3 actors)
    if unique_actors:
        return ', '.join(unique_actors[:3])
    
    return None


def extract_paragraphs(text: str, min_length: int = 50) -> List[str]:
    """
    Split text into meaningful paragraphs.
    
    Args:
        text: Article text
        min_length: Minimum paragraph length
    
    Returns:
        List of paragraphs
    """
    if not text:
        return []
    
    # Split by double newlines (common paragraph separation)
    paragraphs = text.split('\n\n')
    
    # Filter short paragraphs and clean them
    meaningful_paragraphs = []
    for p in paragraphs:
        cleaned = p.strip()
        if len(cleaned) >= min_length:
            meaningful_paragraphs.append(cleaned)
    
    return meaningful_paragraphs


def find_names_in_text(text: str) -> List[str]:
    """
    Find potential names in text (simple heuristic).
    This helps with actor identification.
    
    Args:
        text: Article text
    
    Returns:
        List of potential names
    """
    if not text:
        return []
    
    # Simple pattern for names (capitalized words, at least 2 letters)
    # This is very basic and will have false positives
    words = text.split()
    potential_names = []
    
    for word in words:
        # Remove punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        
        # Check if it looks like a name (starts with capital, reasonable length)
        if (clean_word and clean_word[0].isupper() and 
            len(clean_word) > 2 and len(clean_word) < 20 and
            clean_word.isalpha()):
            
            # Common false positives to exclude
            common_words = {
                'The', 'And', 'But', 'For', 'Nor', 'Yet', 'So',
                'With', 'From', 'That', 'This', 'These', 'Those',
                'Which', 'Who', 'Whom', 'Whose', 'Where', 'When',
                'Why', 'How', 'What', 'Will', 'Would', 'Could',
                'Should', 'Must', 'Might', 'May', 'Can', 'Shall',
                'Israel', 'Palestinian', 'Israeli', 'Palestine',
                'West', 'Bank', 'Gaza', 'Jerusalem', 'Hebron',
                'Bethlehem', 'Ramallah', 'Jenin', 'Nablus',
            }
            
            if clean_word not in common_words:
                potential_names.append(clean_word)
    
    return list(set(potential_names))[:5]  # Return unique, max 5


def estimate_word_count(text: str) -> int:
    """
    Estimate word count in text.
    
    Args:
        text: Article text
    
    Returns:
        Estimated word count
    """
    if not text:
        return 0
    
    words = text.split()
    return len(words)


def is_text_english(text: str, threshold: float = 0.7) -> bool:
    """
    Check if text appears to be English.
    Simple heuristic based on common English words.
    
    Args:
        text: Text to check
        threshold: Proportion of English words needed
    
    Returns:
        True if text appears to be English
    """
    if not text:
        return False
    
    # Common English words
    common_english = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
        'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
        'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
        'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
        'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
        'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
    }
    
    words = text.lower().split()
    if not words:
        return False
    
    # Count English words
    english_count = sum(1 for word in words if word in common_english)
    
    # Calculate proportion
    proportion = english_count / len(words)
    
    return proportion >= threshold


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Text potentially containing HTML
    
    Returns:
        Clean text without HTML tags
    """
    if not text:
        return ""
    
    # Simple HTML tag removal
    clean = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove multiple spaces
    clean = re.sub(r'\s+', ' ', clean)
    
    return clean.strip()
    