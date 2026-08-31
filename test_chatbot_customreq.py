from chatbot import chat
from deepeval.test_case import Turn,ConversationalTestCase
from deepeval import evaluate
from deepeval.metrics import ConversationalGEval
from deepeval.models import AnthropicModel
from deepeval.test_case import MultiTurnParams



eval_model = AnthropicModel(model="claude-haiku-4-5", temperature=0)

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

correctness = ConversationalGEval(
    name ="Correctness",
    criteria=(
        "Did the chatbot fully resolve the customer's issue?"
        "It should use tools when needed and provide accurate answers."
    ),
    model = eval_model,
    threshold = 0.8,
    evaluation_params=[
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT
    ])

evaluate(test_cases = [test_case],metrics=[correctness])