from deepeval.metrics import TaskCompletionMetric
from deepeval.models import AnthropicModel
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.dataset import Golden, EvaluationDataset
from agent_instrumented import support_agent as _support_agent
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace
from deepeval.contextvars import get_current_golden
from deepeval.evaluate import AsyncConfig


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_output=golden.expected_output)
    return _support_agent(user_input)


eval_model = AnthropicModel(model="claude-haiku-4-5", temperature=0)

task_completion = TaskCompletionMetric(threshold=0.7, model=eval_model)
tool_correctness = ToolCorrectnessMetric()

dataSet = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?", expected_tools=[ToolCall(name="get_order_status")]),
    Golden(input="What is the refund policy for electronics?", expected_tools=[ToolCall(name="get_refund_policy")])
])

for golden in dataSet.evals_iterator(metrics=[task_completion, tool_correctness]
                                     ,async_config=AsyncConfig(run_async=False)):
    support_agent(golden.input)
