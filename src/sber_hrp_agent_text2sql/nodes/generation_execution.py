import logging

import pandas as pd
from sqlglot.errors import TokenError, ParseError
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.types import Command

from src.hrp_agent_text2sql.gigamodels import model
from src.hrp_agent_text2sql.promts.decompose import promt_cte_fin
from src.hrp_agent_text2sql.promts.main import prompt_main, prompt_main_decompose_ch_error
from src.hrp_agent_text2sql.promts.tasks import promt_describer_solution
from src.hrp_agent_text2sql.schemas.agent_context import ContextSchema
from src.hrp_agent_text2sql.schemas.agent_state import GraphState
from src.hrp_agent_text2sql.schemas.structured_output import (
    CumulativeSQLOutput,
    SQLCreatorOutput,
    SQLRecreatorOutput,
)
from src.hrp_agent_text2sql.utils.node_log import log_node_execution
from src.hrp_agent_text2sql.utils.parser_processing import (
    EscapedPydanticOutputParser,
    try_invoke,
)
from src.hrp_agent_text2sql.utils.sql import can_return_result

logger = logging.getLogger(__name__)

@log_node_execution("create_cumulative_sql_node")
def create_cumulative_sql_node(state: GraphState) -> Command:
    """
    Generates final SQL consisting of CTEs responding to intermediate stages of the plan.
    """
    plan2use_promt = state.get("plan2use_promt")
    tb_desc_promt = state.get("tb_desc_promt")

    # we create a prompt that will be sent in case of an error from ClickHouse
    promt_to_inference_ch_error = prompt_main_decompose_ch_error.format(question=state["question_final"],
                                        plan=plan2use_promt,
                                        TABLE_DESCRIPTIONS=tb_desc_promt,
                                        ACTUAL_DATA=ContextSchema.ACTUAL_DATA,
                                        add_promt=state["add_oss_promt"],
                                        add_fio_info=state["add_fio_info"],
                                        add_find_teams_promt=state["add_find_teams_promt"],
                                        add_lang_features_promt=state["add_lang_features_promt"],
                                        add_skills_promt=state["add_skills_promt"])
    promt_to_inference_ch_error = promt_to_inference_ch_error.replace('{', '{{').replace('}', '}}')

    user_query = state.get("question_final")
    plan2use = state.get("plan")
    decompose_results = state.get("decompose_results")

    scratch = f"Основоной вопрос: {user_query}\n"
    for i in range(len(plan2use)):
        scratch = scratch + f"{i + 1} этап плана аналитики, {plan2use[f'step_{i + 1}']['step_desc']}\n " + \
            f"Запись в таблицу(CTE) с именем {plan2use[f'step_{i + 1}']['tab_name']}\n"
        scratch = scratch + "*"*80 + '\n'
        scratch = scratch + decompose_results[i] + '\n'
        scratch = scratch + "="*80 + "\n"

    promt = ChatPromptTemplate.from_messages([
        ("system", promt_cte_fin)
    ])
    parser = EscapedPydanticOutputParser(pydantic_object=CumulativeSQLOutput)
    final_promt = promt.partial(frmt=parser.get_format_instructions(),
                                scratch=scratch)
    chain = final_promt | model | parser
    respond = try_invoke(chain)

    return Command(goto="preprocess_sql_query_node", 
                    update={"output_sql": respond.sql,
                            "sql_meta": respond.sql_meta,
                            "analysis": respond.analysis,
                            "promt_to_inference": promt_to_inference_ch_error})

@log_node_execution("create_sql_query_node")
def create_sql_query_node(state: GraphState) -> Command:
    """
    Creates a final promt and sends it to llm.
    The Promt and the resulting SQL query are added to GraphState.
    """
    promt_to_inference = prompt_main.format(question=state["question_final"], 
                                        TABNAME=ContextSchema.TABNAME,
                                        ACTUAL_DATA=ContextSchema.ACTUAL_DATA,
                                        TABLE_DESCRIPTION=state["table_description_prunned"],
                                        add_promt=state["add_oss_promt"],
                                        add_fio_info=state["add_fio_info"],
                                        add_find_teams_promt=state["add_find_teams_promt"],
                                        add_lang_features_promt=state["add_lang_features_promt"],
                                        add_skills_promt=state["add_skills_promt"])
    promt_to_inference_ = promt_to_inference.replace('{', '{{').replace('}', '}}')

    task_promt = f"""
    Ты эксперт по SQL и анализу данных. 
    """ 
    sql_creator_parser = EscapedPydanticOutputParser(pydantic_object=SQLCreatorOutput)     
    promt = ChatPromptTemplate.from_messages([
            ("system", task_promt + "\nWrap the output in ```json``` tags\n{format_instructions}."), 
            ("human", promt_to_inference_)
        ]
    ).partial(format_instructions=sql_creator_parser.get_format_instructions())
    chain = promt | model | sql_creator_parser
    creater_output = try_invoke(chain)

    return Command(goto="preprocess_sql_query_node", update={"promt_to_inference": promt_to_inference,
                                                        "output_sql": creater_output.sql,
                                                        "sql_meta": creater_output.sql_meta,
                                                        "new_question": creater_output.question})

@log_node_execution("preprocess_sql_query_node")
def preprocess_sql_query_node(state: GraphState) -> Command:
    """
    Preprocessing an SQL query received from llm. The final SQL is added to GraphState.
    """ 
    final_sql = state["output_sql"].replace(";", "")
    is_private_network = True
    can_execute = True
    rls_error = ""

    return Command(goto="sql_query_to_click_node",
                update={"final_sql": final_sql,
                        "can_execute": can_execute and can_return_result(final_sql, is_private_network),
                        "rls_error": rls_error})

@log_node_execution("sql_query_to_click_node")
def sql_query_to_click_node(state: GraphState) -> Command:
    """
    A request in click, with a request without errors, saves the result in GraphState, with success_status True, otherwise False. 
    Counts the number of query attempts (SQL generation).
    """
    final_try = state["final_try"] + 1

    if not state["can_execute"]:
        return Command(goto="create_sql_query_node" if final_try <= ContextSchema.sql_query_retry else END,\
                        update={"message": "Запрос не может быть выполнен",
                                "final_try": final_try,
                                "output_table": [],
                                "final_success_status": False,
                                "rls_error": state["rls_error"]})
    try:
        sql = state["final_sql"]
        data_ = ContextSchema.click_client.query(f"""
            {sql}
        """)
        if isinstance(data_, (dict, list)):
            output_table = pd.DataFrame(data_)
        else:
            output_table = pd.DataFrame(data_.result_rows, columns=data_.column_names)
        promt_to_inference = state["promt_to_inference"].replace('{', '{{').replace('}', '}}')
        return Command(goto="explain_sql_query_node", update={"messages": [{"role": "user", "content": promt_to_inference},\
                                                                      {"role": "ai", "content": sql}],
                                                         "final_try": final_try,
                                                         "output_table": output_table.to_dict(orient='records'),
                                                         "final_success_status": True})
    except Exception as e:
        output_table = []
        ch_error = f"Описание ошибки: {str(e)}"
        return Command(goto="recreate_sql_ch_error_node" if final_try <= ContextSchema.sql_query_retry else END,\
                update={"final_try": final_try,
                        "output_table": output_table,
                        "ch_error": ch_error,
                        "final_success_status": False})

@log_node_execution("recreate_sql_ch_error_node")
def recreate_sql_ch_error_node(state: GraphState) -> Command:
    """
    Rewrites the SQL query, processing an error from ClickHouse.
    """
    promt_to_inference = state["promt_to_inference"].replace('{', '{{').replace('}', '}}')
    task_promt = f"""
    Ты эксперт по SQL и анализу данных. 
    На основе истории диалога и полученной ошибки, исправь изначальный SQL запрос.       
    Напиши новый корректный SQL, который будет соответствовать всем изначальным требованиям и логике.
    """    
    sql_recreator_parser = EscapedPydanticOutputParser(pydantic_object=SQLRecreatorOutput)      
    promt = ChatPromptTemplate.from_messages([
            ("system", task_promt + "\nWrap the output in ```json``` tags\n{format_instructions}."), 
            ("human", promt_to_inference),
            ("ai", state["final_sql"]),
            ("human", state["ch_error"])
        ]
    ).partial(format_instructions=sql_recreator_parser.get_format_instructions())
    chain = promt | model | sql_recreator_parser
    recreater_output = try_invoke(chain)

    return Command(goto="preprocess_sql_query_node", update={"sql_recreator_ch_error": recreater_output,
                                                        "output_sql": recreater_output.fixed_sql,
                                                        "sql_meta": recreater_output.sql_meta})

@log_node_execution("explain_sql_query_node")
def explain_sql_query_node(state: GraphState) -> Command:
    """
    Description/explanation of the SQL query generated and added to GraphState.
    """
    data_desc = state.get("tb_desc_promt", state["table_description_prunned"])
    frm_explain = promt_describer_solution.format(message=state["question_final"], sql_query=state["final_sql"], 
                                                                    TABLE_DESCRIPTION=data_desc)
    explanation = model.invoke(frm_explain).content

    return Command(goto=END, update={"explanation_of_sql": explanation})
