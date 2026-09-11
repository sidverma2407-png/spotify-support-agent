import re

class EscalationPolicy:
    def __init__(self, retrieval_threshold=0.65, confidence_threshold=0.60):
        self.retrieval_threshold = retrieval_threshold
        self.confidence_threshold = confidence_threshold
        
        self.escalation_intents = {
            'other_unclear': 'Intent is unclear',
            'agent_handoff': 'Customer explicitly requests human agent or handoff is typical',
            'cancellation_refund': 'Requires account-specific financial action'
        }
        
        self.sensitive_keywords = [
            r'\bpassword\b', r'\bhacked\b', r'\bstolen\b', r'\bbank\b', 
            r'\bchargeback\b', r'\bfraud\b', r'\bsuicide\b', r'\bpolice\b'
        ]
        
    def should_escalate(self, customer_message, predicted_intent, retrieved_examples, classifier_confidence=None):
        """
        Evaluates whether a message should be escalated based on defined policies.
        Returns (bool, str) -> (should_escalate, reason)
        """
        # 1. Intent-based escalation
        if predicted_intent in self.escalation_intents:
            return True, self.escalation_intents[predicted_intent]
            
        # 2. Confidence threshold
        if classifier_confidence is not None and classifier_confidence < self.confidence_threshold:
            return True, f"Classifier confidence ({classifier_confidence:.2f}) below threshold ({self.confidence_threshold})"
            
        # 3. Retrieval evidence threshold
        if not retrieved_examples:
            return True, "No historical examples retrieved"
            
        top_score = retrieved_examples[0]['score']
        if top_score < self.retrieval_threshold:
            return True, f"Top retrieval score ({top_score:.2f}) below threshold ({self.retrieval_threshold})"
            
        # 4. Retrieval conflict (Top examples strongly disagree on intent)
        top_intents = [ex['intent'] for ex in retrieved_examples[:3]]
        if len(set(top_intents)) == 3: # All 3 top examples have different intents
            return True, "Retrieval evidence strongly disagrees on intent/topic"
            
        # 5. Sensitive/Security keywords
        lower_msg = customer_message.lower()
        for kw in self.sensitive_keywords:
            if re.search(kw, lower_msg):
                return True, f"Sensitive/security keyword detected ({kw})"
                
        # Safe to auto-handle
        return False, "Sufficient evidence and no escalation triggers"
