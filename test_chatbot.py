from chatbot import chat
from deepeval.test_case import Turn,ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric,KnowledgeRetentionMetric,ConversationCompletenessMetric
from deepeval import evaluate


turns = []
history = []

for  user_msg in [
    "Hi, I placed an order last week, the order ID is ORD-1042.",
    "Is it ging to arrive on time?",
    "What was the ETA you just mentioned?",
    "Can I upgrade to express shipping?"
]:
        reply,history, _ = chat(user_msg,history)
        turns.append(Turn(role="user",content=user_msg))
        turns.append(Turn(role="assistant", content=reply))

test_case = ConversationalTestCase(turns= turns)
turnRelevancyMetric = TurnRelevancyMetric(threshold = 0.5,model ="claude-haiku-4-5")
retentionMetric = KnowledgeRetentionMetric(threshold = 0.5, model = "claude-haiku-4-5")
completenessMetric = ConversationCompletenessMetric(threshold = 0.5,model = "claude-haiku-4-5")

evaluate(test_cases = [test_case],metrics=[turnRelevancyMetric,retentionMetric,completenessMetric])
