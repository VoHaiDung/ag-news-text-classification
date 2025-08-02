import re
from typing import List, Optional
import unicodedata

class TextCleaner:
    """Text preprocessing and cleaning"""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_numbers: bool = False):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.number_pattern = re.compile(r'\d+')
        
    def clean_text(self, text: str) -> str:
        """Clean a single text"""
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
            
        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
            
        # Lowercase
        if self.lowercase:
            text = text.lower()
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts"""
        return [self.clean_text(text) for text in texts]
