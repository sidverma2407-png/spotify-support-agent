import re

class TemplateGenerator:
    def __init__(self):
        # Fallback safe templates if we can't extract good info from historical
        self.intent_templates = {
            'account_login': "Hi there! We can help with your account access. Could you DM us your account's email address or username? We'll take a look under the hood.",
            'billing_subscription': "Hey! Help's here for your subscription. Can you DM us the email address tied to your account? We'll check backstage.",
            'playback_app_issue': "Hi there! Let's get this sorted. Could you let us know the exact device, operating system, and Spotify version you're using?",
            'content_availability': "Hey! Content availability can vary depending on licensing agreements. You can find more info on how this works here: https://support.spotify.com/article/content-missing/",
            'ads_recommendations': "Hi! Thanks for the feedback. We're always testing new features and ways to improve recommendations. We'll make sure to pass this along to the right team.",
            'praise_gratitude': "Thanks for the shoutout! We're thrilled you're enjoying the music. Let us know if you ever need anything else. \ud83c\udfb6"
        }

    def clean_historical_response(self, response):
        """
        Cleans a historical response by removing agent initials and specific user mentions.
        """
        # Remove agent signatures like " /AB", "- XY", "^XYZ"
        response = re.sub(r'\s+/[A-Z]{2,3}$', '', response)
        response = re.sub(r'\s+-[A-Z]{2,3}$', '', response)
        response = re.sub(r'\s+\^[A-Z]{2,3}$', '', response)
        
        # Replace specific @ mentions
        response = re.sub(r'@\d+', '', response).strip()
        
        # Replace "Hi [Name]!" with a generic greeting
        response = re.sub(r'^(Hi|Hey|Hello)\s+[A-Za-z]+[!,]?\s*', r'\1 there! ', response)
        
        return response

    def extract_standard_link(self, response):
        """Extracts standard Spotify support links if present."""
        links = re.findall(r'https?://(?:support\.spotify\.com|spoti\.fi)/\S+', response)
        return links[0] if links else None

    def adapt_response(self, intent, retrieved_example):
        """
        Deterministically adapt a historical response.
        If it's safe to adapt, we clean it. If it contains potentially unsafe 
        specifics that are hard to clean, we use a template.
        """
        historical_response = retrieved_example['support_response']
        cleaned = self.clean_historical_response(historical_response)
        
        # If the response still contains potentially specific order numbers or weird URLs, use template
        # (Very basic heuristic for deterministic safety)
        if re.search(r'\b\d{6,}\b', cleaned): # Long numbers might be order IDs
            return self.intent_templates.get(intent, cleaned)
            
        return cleaned
