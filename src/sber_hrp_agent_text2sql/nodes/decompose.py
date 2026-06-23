import logging

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from src.hrp_agent_text2sql.gigamodels import model
from src.hrp_agent_text2sql.promts.decompose import (
    promt_compl_checker,
    promt_planner_v4c,
)
from src.hrp_agent_text2sql.promts.main import prompt_main_decompose
from src.hrp_agent_text2sql.schemas.agent_context import ContextSchema
from src.hrp_agent_text2sql.schemas.agent_state import GraphState
from src.hrp_agent_text2sql.schemas.structured_output import (
    ComplexityFlagOutput,
    DecomposeSQLCreatorOutput,
    SimplePlanOutput
)
from src.hrp_agent_text2sql.utils.node_log import log_node_execution
from src.hrp_agent_text2sql.utils.parser_processing import (
    EscapedPydanticOutputParser,
    try_invoke,
)

logger = logging.getLogger(__name__)

@log_node_execution("get_complexity_node")
def get_complexity_node(state: GraphState) -> Command:
    """
    Checks the question for complexity and the need for decomposition into several simple questions.
    """
    user_query = state.get("question_final")
    promt = ChatPromptTemplate.from_messages([
        ("system", promt_compl_checker)
    ])
    parser = EscapedPydanticOutputParser(pydantic_object=ComplexityFlagOutput)
    final_promt = promt.partial(frmt=parser.get_format_instructions(),
                                qu=user_query,
                                attr=state["table_description_prunned"])
    chain = final_promt | model | parser
    respond = try_invoke(chain)
    return Command(goto="get_decompose_node" if respond.verdict else "create_sql_query_node", \
                   update={"complexity_of_query": respond})

@log_node_execution("get_decompose_node")
def get_decompose_node(state: GraphState) -> Command:
    """
    Creates a plan of several simple tasks to answer a complex question.
    """
    user_query = state.get("question_final")
    promt = ChatPromptTemplate.from_messages([
        ("system", promt_planner_v4c)
    ])
    parser = EscapedPydanticOutputParser(pydantic_object=SimplePlanOutput)
    final_promt = promt.partial(frmt=parser.get_format_instructions(),
                                qu=user_query,
                                attr_pool=state["table_description_prunned"])
    chain = final_promt | model | parser
    respond = try_invoke(chain)

    plan2use = respond.output
    plan2use_promt = [f"{key[-1]}. {plan2use[key]['step_desc']} - Результат записан в таблицу: {plan2use[key]['tab_name']}"
                for key in plan2use]
    plan2use_promt = "\n".join(plan2use_promt)
    # initial list of tables available for use for SQL generation
    start_tab = [{"name": ContextSchema.TABNAME, "columns": state["table_description_prunned"], "step_desc": ""}]
    return Command(goto="create_decompose_sql_query_node", 
                   update={"plan": plan2use,
                           "plan2use_promt": plan2use_promt,
                           "tables2use": start_tab,
                           "decompose_results": []})

@log_node_execution("create_decompose_sql_query_node")
def create_decompose_sql_query_node(state: GraphState) -> Command:
    """
    Generates single step SQL in terms of answering a complex question.
    """
    tb_desc = state.get("tables2use") # available tables
    plan_stage = len(tb_desc) # current stage of the plan
    plan2use_promt = state.get("plan2use_promt") # promt of the entire plan for answering the question

    step_desc_frmt = "Служит ответом на промежуточный вопрос по пункту плана: {step_desc}"
    tb_desc_promt = [f"{i + 1}) Таблица {sample['name']}. " + \
               (step_desc_frmt.format(step_desc=sample['step_desc']) if sample['step_desc'] else "") + \
               f" С атрибутным составом:\n{sample['columns']}" for i, sample in enumerate(tb_desc)]
    tb_desc_promt = "\n".join(tb_desc_promt)

    promt_to_inference = prompt_main_decompose.format(question=state["question_final"],
                                        plan=plan2use_promt,
                                        plan_stage=plan_stage,
                                        ACTUAL_DATA=ContextSchema.ACTUAL_DATA,
                                        TABLE_DESCRIPTIONS=tb_desc_promt,
                                        add_promt=state["add_oss_promt"],
                                        add_fio_info=state["add_fio_info"],
                                        add_find_teams_promt=state["add_find_teams_promt"],
                                        add_lang_features_promt=state["add_lang_features_promt"],
                                        add_skills_promt=state["add_skills_promt"])
    promt_to_inference_ = promt_to_inference.replace('{', '{{').replace('}', '}}')

    task_promt = f"""
    Ты эксперт по SQL и анализу данных. 
    """ 
    sql_creator_parser = EscapedPydanticOutputParser(pydantic_object=DecomposeSQLCreatorOutput)     
    promt = ChatPromptTemplate.from_messages([
            ("system", task_promt + "\nWrap the output in ```json``` tags\n{format_instructions}."), 
            ("human", promt_to_inference_)
        ]
    ).partial(format_instructions=sql_creator_parser.get_format_instructions())
    chain = promt | model | sql_creator_parser
    creater_output = try_invoke(chain)

    new_tab_desc = creater_output.sql_meta
    new_tab_desc = ",\n".join([f"Колонка {i} содержит {new_tab_desc[i]}" for i in new_tab_desc])
    plan2use = state.get("plan")

    # add a new table to the list of available ones
    tb_desc.append({"name": plan2use[f"step_{plan_stage}"]["tab_name"],
                    "columns": new_tab_desc,
                    "step_desc": plan2use[f"step_{plan_stage}"]["step_desc"]})
    
    # we add to the list of intermediate results the generated SQL responding to one of the stages of the plan
    decompose_results = state.get("decompose_results")
    decompose_results.append(creater_output.sql)
    
    target_node = "create_decompose_sql_query_node" if plan_stage < len(plan2use) else "create_cumulative_sql_node"
    return Command(goto=target_node, update={"tables2use": tb_desc,
                                             "tb_desc_promt": tb_desc_promt,
                                                             "output_sql": creater_output.sql,
                                                             "sql_meta": creater_output.sql_meta,
                                                             "decompose_results": decompose_results})
