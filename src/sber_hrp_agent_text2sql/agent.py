from typing import Dict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from src.hrp_agent_text2sql.schemas.agent_state import GraphState
from src.hrp_agent_text2sql.utils.node_log import NodeRegistry

from src.hrp_agent_text2sql.nodes.preparations_inspections import (feature_checker_node,
                                                                        self_info_checker_node)

from src.hrp_agent_text2sql.nodes.attributes_processing import (find_names_node,
                                                                     get_cat_features_node,
                                                                     rukovod_methodology_node,
                                                                     add_lang_features_node,
                                                                     check_found_skills_node,
                                                                     get_oss_agile_structure_node,
                                                                     column_prunner_node)

from src.hrp_agent_text2sql.nodes.decompose import (get_complexity_node,
                                                         get_decompose_node,
                                                         create_decompose_sql_query_node)

from src.hrp_agent_text2sql.nodes.generation_execution import (create_cumulative_sql_node,
                                                                    create_sql_query_node,
                                                                    preprocess_sql_query_node,
                                                                    sql_query_to_click_node,
                                                                    recreate_sql_ch_error_node,
                                                                    explain_sql_query_node)


class AnAgent:
    def __init__(self):
        pass

    def create_graph(self):
        checkpointer = InMemorySaver() 
        workflow = StateGraph(GraphState)

        workflow.add_node("feature_checker_node", feature_checker_node)
        workflow.add_node("self_info_checker_node", self_info_checker_node)
        workflow.add_node("find_names_node", find_names_node)
        workflow.add_node("get_cat_features_node", get_cat_features_node)
        workflow.add_node("rukovod_methodology_node", rukovod_methodology_node)
        workflow.add_node("add_lang_features_node", add_lang_features_node)
        workflow.add_node("check_found_skills_node", check_found_skills_node)
        workflow.add_node("get_oss_agile_structure_node", get_oss_agile_structure_node)
        workflow.add_node("column_prunner_node", column_prunner_node)
        workflow.add_node("get_complexity_node", get_complexity_node)
        workflow.add_node("get_decompose_node", get_decompose_node)
        workflow.add_node("create_decompose_sql_query_node", create_decompose_sql_query_node)
        workflow.add_node("create_cumulative_sql_node", create_cumulative_sql_node)
        workflow.add_node("create_sql_query_node", create_sql_query_node)
        workflow.add_node("preprocess_sql_query_node", preprocess_sql_query_node)
        workflow.add_node("sql_query_to_click_node", sql_query_to_click_node)
        workflow.add_node("recreate_sql_ch_error_node", recreate_sql_ch_error_node)
        workflow.add_node("explain_sql_query_node", explain_sql_query_node)

        workflow.set_entry_point("feature_checker_node")

        app = workflow.compile()
        return app

    def get_initial_state(self, question_from_user: str, user_data: Dict) -> dict:
        """Получить начальное состояние"""
        state = {
            "question": question_from_user,
            "user_data": user_data,
            "final_try": 0,
            "add_oss_promt": "",
            "add_lang_features_promt": "",
            "add_skills_promt": "",
            "explanation_of_sql": "",
            "final_success_status": False,
            "ch_error": "",
            "output_oss_di": {
                "question": "",
                "extractedNER": [],
                "findInOss": []
            },
            "execution_logs": {},
            "prunner_oss_columns_exceptions": [],
            "prunner_ruk_columns_exceptions": []
        }
        NodeRegistry.initialize_logs(state)
        return state