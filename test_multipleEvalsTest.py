from deepeval.metrics import PromptAlignmentMetric
from deepeval.models import AnthropicModel
from deepeval.dataset import Golden, EvaluationDataset
from agent_instrumented import support_agent as _support_agent
from deepeval.tracing import observe
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import StepEfficiencyMetric
from deepeval.metrics import AnswerRelevancyMetric


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)


eval_model = AnthropicModel(model="claude-haiku-4-5", temperature=0)

prompt_alignment=PromptAlignmentMetric(prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Keep replies short and helpful."],threshold=0.7,model=eval_model)

step_efficiency =StepEfficiencyMetric(threshold=0.5,model=eval_model)

answer_relevancy = AnswerRelevancyMetric(threshold=0.7,model=eval_model)

dataSet = EvaluationDataset(goldens=
                  [   Golden(input ="Where is my order ORD-1042?"),
                      Golden(input ="What is the refund policy for electronics?"),
                      Golden(input ="I want to return my order ORD-1042. What should I do?")
                  ])

for golden in dataSet.evals_iterator(metrics=[prompt_alignment, step_efficiency, answer_relevancy]
                                     ,async_config=AsyncConfig(run_async=False)):
    support_agent(golden.input)