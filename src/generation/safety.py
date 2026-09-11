import re

class SafetyChecker:
    def __init__(self):
        self.unsupported_promises = [
            r'\bwe will refund\b', r'\bguarantee\b', r'\bpromise\b', 
            r'\bwe will fix this immediately\b'
        ]
        self.unsupported_timelines = [
            r'\bin 24 hours\b', r'\btomorrow\b', r'\bby next week\b',
            r'\bwithin \d+ hours\b', r'\bwithin \d+ days\b'
        ]
        self.completed_actions = [
            r'\bwe have updated\b', r'\bwe cancelled\b', r'\bwe refunded\b',
            r'\bi have changed\b', r'\bhas been fixed\b'
        ]
        self.excessive_certainty = [
            r'\b100%\b', r'\bdefinitely\b', r'\babsolutely\b', r'\bwithout a doubt\b'
        ]
        
    def check_response(self, response_text):
        """
        Runs automated diagnostics on the generated response.
        Returns a list of safety violation messages (empty if safe).
        """
        violations = []
        lower_resp = response_text.lower()
        
        # Check promises
        for p in self.unsupported_promises:
            if re.search(p, lower_resp):
                violations.append(f"Unsupported promise detected: '{p}'")
                
        # Check timelines
        for t in self.unsupported_timelines:
            if re.search(t, lower_resp):
                violations.append(f"Unsupported timeline detected: '{t}'")
                
        # Check completed actions
        for c in self.completed_actions:
            if re.search(c, lower_resp):
                violations.append(f"Claim of completed action detected: '{c}'")
                
        # Check certainty
        for c in self.excessive_certainty:
            if re.search(c, lower_resp):
                violations.append(f"Excessive certainty detected: '{c}'")
                
        # Check URLs (allow only spotify URLs or twitter DMs)
        urls = re.findall(r'https?://[^\s]+', lower_resp)
        for url in urls:
            if not (url.startswith('https://support.spotify.com') or 
                    url.startswith('http://support.spotify.com') or
                    url.startswith('https://spoti.fi') or 
                    url.startswith('http://spoti.fi') or
                    url.startswith('https://twitter.com')):
                violations.append(f"Unsupported URL detected: '{url}'")
                
        return violations
