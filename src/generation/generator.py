from src.generation.policy import EscalationPolicy
from src.generation.templates import TemplateGenerator
from src.generation.safety import SafetyChecker

class ResponseGenerator:
    def __init__(self, retrieval_threshold=0.65, confidence_threshold=0.60):
        self.policy = EscalationPolicy(retrieval_threshold, confidence_threshold)
        self.templates = TemplateGenerator()
        self.safety = SafetyChecker()
        
    def generate_response(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_examples: list,
        classifier_confidence: float = None
    ) -> dict:
        """
        Generates a grounded response or escalates.
        """
        # 1. Check Escalation Policy
        should_escalate, reason = self.policy.should_escalate(
            customer_message, 
            predicted_intent, 
            retrieved_examples, 
            classifier_confidence
        )
        
        if should_escalate:
            return {
                "response": "This request has been escalated to a human agent.",
                "action": "escalate",
                "intent": predicted_intent,
                "confidence": classifier_confidence,
                "evidence": [],
                "reason": reason,
                "safety_violations": []
            }
            
        # 2. Generation Strategy (Deterministic Adaptation)
        top_example = retrieved_examples[0]
        generated_text = self.templates.adapt_response(predicted_intent, top_example)
        
        # 3. Safety Diagnostics
        violations = self.safety.check_response(generated_text)
        
        # If it violates safety during generation, escalate as a fallback
        if violations:
            return {
                "response": "This request has been escalated to a human agent.",
                "action": "escalate",
                "intent": predicted_intent,
                "confidence": classifier_confidence,
                "evidence": [{"conversation_id": top_example["conversation_id"], "similarity": top_example["score"]}],
                "reason": f"Safety/Grounding violation detected during generation: {'; '.join(violations)}",
                "safety_violations": violations
            }
            
        # 4. Success Auto-handle
        return {
            "response": generated_text,
            "action": "auto_handle",
            "intent": predicted_intent,
            "confidence": classifier_confidence,
            "evidence": [{"conversation_id": top_example["conversation_id"], "similarity": top_example["score"]}],
            "reason": "Sufficient historical evidence with matching intent.",
            "safety_violations": []
        }
