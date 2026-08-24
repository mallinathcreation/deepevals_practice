from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from agent_instrumented import support_agent
from deepeval.metrics import TaskCompletionMetric
from deepeval.models import AnthropicModel

actual_output= support_agent("Where is my order ORD-1042?")


test_case = LLMTestCase(input="Where is my order ORD-1042?",
            actual_output = actual_output)

eval_model = AnthropicModel(model="claude-haiku-4-5",temperature=0)

evaluate(test_cases=[test_case],metrics=[TaskCompletionMetric(threshold=0.7,model=eval_model)])