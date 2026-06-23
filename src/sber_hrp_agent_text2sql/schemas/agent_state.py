from typing import TypedDict, Annotated, Dict, List, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.hrp_agent_text2sql.schemas.structured_output import CatFeaturesOutput, ColumnsPrunningOutput, ComplexityFlagOutput


class ExecutionRecord(TypedDict):
    execution_id: str 
    duration_sec: Optional[float]
    executed: bool
    error: Optional[str]


class NodeExecutionLog(TypedDict):
    node_name: str
    total_executions: int
    total_duration_sec: float
    executions: List[ExecutionRecord]


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_data: Dict
    question: str 
    question_final: str
    output_table: list
    output_sql: str
    final_sql: str
    sql_meta: dict
    explain4selfinfo: str 
    final_try: int
    ch_error: str
    final_success_status: bool 
    output_oss_di: dict
    explanation_of_sql: str 
    add_oss_promt: str
    promt_to_inference: str
    add_lang_features_promt: str
    add_find_teams_promt: str
    add_skills_promt: str
    found_names: List[Dict]
    add_fio_info: str
    cat_features: CatFeaturesOutput
    skills_map: dict
    rls_error: str
    can_execute: bool
    execution_logs: Dict[str, NodeExecutionLog]
    execution_counter: Dict[str, int] 
    table_description_prunned: str
    new_question: str
    prunner_oss_columns_exceptions: List[str]
    prunner_ruk_columns_exceptions: List[str]
    prunner_res: ColumnsPrunningOutput
    complexity_of_query: ComplexityFlagOutput
    plan: Dict[str, Dict[str, str]]
    tables2use: Dict[str, str]
    plan2use_promt: str
    decompose_results: List[str]
    tb_desc_promt: str
