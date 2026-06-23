import logging

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from src.hrp_agent_text2sql.errors import CannotAnswerError
from src.hrp_agent_text2sql.gigamodels import model
from src.hrp_agent_text2sql.promts.main import COLUMN_META
from src.hrp_agent_text2sql.promts.tasks import (
    promt_check_self_info,
    promt_verifier,
)
from src.hrp_agent_text2sql.schemas.agent_context import ContextSchema
from src.hrp_agent_text2sql.schemas.agent_state import GraphState
from src.hrp_agent_text2sql.schemas.structured_output import (
    FeatureCheckerOutput,
    SelfCheckerOutput,
)
from src.hrp_agent_text2sql.utils.node_log import log_node_execution
from src.hrp_agent_text2sql.utils.parser_processing import (
    EscapedPydanticOutputParser,
    try_invoke,
)

logger = logging.getLogger(__name__)


@log_node_execution("feature_checker_node") 
def feature_checker_node(state: GraphState) -> Command:
    """
    Checks several times using promt whether there are enough attributes in the storefront to answer the question
    if the answer is NO at least once, the check fails.
    """
    for _ in range(ContextSchema.feature_checker_retry):
        promt = ChatPromptTemplate.from_messages([
            ("human", promt_verifier)
        ])
        parser = EscapedPydanticOutputParser(pydantic_object=FeatureCheckerOutput)
        final_promt = promt.partial(output_format=parser.get_format_instructions(),
                                    COLUMN_META=COLUMN_META,
                                    question=state["question"])
        chain = final_promt | model | parser
        respond = try_invoke(chain)
        if respond.verdict == 0:
            raise CannotAnswerError(f"There are not enough attributes in the storefront to answer the question.\n{respond.justification}")
        
    return Command(goto="self_info_checker_node", update={"explain4selfinfo": '',
                                                          "final_sql": '',
                                                          "output_table": []})
    
@log_node_execution("self_info_checker_node") 
def self_info_checker_node(state: GraphState) -> Command:
    """
    Checks using promt whether the question is about self. If yes, then a clarifying part is added to the question.
    The result of the check is returned to GraphState.
    """
    question_final = state["question"]
    promt = ChatPromptTemplate.from_messages([
        ("human", promt_check_self_info)
    ])
    parser = EscapedPydanticOutputParser(pydantic_object=SelfCheckerOutput)
    final_promt = promt.partial(output_format=parser.get_format_instructions(),
                                question=state["question"])
    chain = final_promt | model | parser
    respond = try_invoke(chain)

    if respond.verdict == 1:
        user_data = state.get("user_data")
        self_add_part = """Меня зовут {surname} {name} {patr}, мой табельный номер {i_pernr}. """.\
            format(surname=user_data["per_fio"]["surname"], 
                    name=user_data["per_fio"]["name"],
                    patr=user_data["per_fio"]["patronimics"],
                    i_pernr=user_data["employee_id"])
        question_final = self_add_part + question_final

    return Command(goto="find_names_node", update={"question_final": question_final,
                                                   "explain4selfinfo": respond.analysis})
