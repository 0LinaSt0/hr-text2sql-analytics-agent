import copy
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from src.hrp_agent_text2sql.config import (
    CUT_OSS_WITH_SIMILARITY,
    FIND_COMMAND_WITH_OSS,
)
from src.hrp_agent_text2sql.gigamodels import model
from src.hrp_agent_text2sql.promts.add import (add_lang_features_promt, 
                                                    add_find_boss_promt)
from src.hrp_agent_text2sql.promts.main import TABLE_DESCRIPTION
from src.hrp_agent_text2sql.promts.tasks import (
    promt_oss_agile_extractor,
    skills_task_promt,
    task_promt_cat_features,
    task_promt_columns,
)
from src.hrp_agent_text2sql.schemas.agent_context import ContextSchema
from src.hrp_agent_text2sql.schemas.agent_state import GraphState
from src.hrp_agent_text2sql.schemas.db_info import COLUMNS_DESCRIPTION
from src.hrp_agent_text2sql.schemas.structured_output import (
    CatFeaturesOutput,
    ColumnsPrunningOutput,
    OssAgileStructure,
    RelevantSkills,
)
from src.hrp_agent_text2sql.utils.node_log import log_node_execution
from src.hrp_agent_text2sql.utils.oss import (
    check_found_struct,
    get_oss_sql_full_data,
    make_oss_promt2add,
    search_structure,
)
from src.hrp_agent_text2sql.utils.parser_processing import (
    EscapedPydanticOutputParser,
    try_invoke,
)
from src.hrp_agent_text2sql.utils.person import (
    find_nearest_by_oss,
    format_filter4ruk_wihIPERNR,
    format_filter4ruk_wihtOSS,
    get_fio_i_pernr_from_input,
    get_i_pernr_from_base,
)
from langgraph.graph import END

logger = logging.getLogger(__name__)


@log_node_execution("find_names_node") 
def find_names_node(state: GraphState) -> Command:
    """
    Collects the full names specified in the user's request and links them to the closest ones in the linear and agile structure of the TBn.
    """
    user_input = state.get("question_final")
    mydata = state.get("user_data") # dict with the user's timesheet and full name 
    persons_info = state.get("found_names", [])
    fios_and_tbns = get_fio_i_pernr_from_input(user_input) # list of dictionaries with keys: full name, TBn found in the question
    for sample in fios_and_tbns:
        value_fio = get_i_pernr_from_base(sample)
        if len(value_fio) > 1: # found several with the same name 
            nearest = find_nearest_by_oss(mydata, value_fio, cut=1) 
        elif len(value_fio) == 0: # search returned no results
            nearest = dict()
        else:
            nearest = copy.copy(value_fio) # found only one employee

        if len(nearest) > 0: # check for search success
            persons_info.append({**sample, "found_names": nearest.to_dict("records")}) # add a dictionary with information about the employee from the question
    tofrmt = "Для идентификации сотрудника {fio} используй его табельный номер - {tbn}, в фильтрах(WHERE) при составлении sql-запроса."
    add_fio_info = [tofrmt.format(fio=smp["fio"], tbn=str(smp["found_names"][0]["employee_id"]))
                        for smp in persons_info if smp["fio"] != "не найдено"]
    add_fio_info = "\n  ".join(add_fio_info) # Promt for employees
    return Command(goto="get_cat_features_node", update={"found_names": persons_info,
                                                    "add_fio_info": add_fio_info})

@log_node_execution("get_cat_features_node") 
def get_cat_features_node(state) -> Command:
    """
    Using promt, it extracts categorical features: skills and determines whether a methodology is needed by commands and languages.
    The result is returned to GraphState.
    """
    cat_fetures_parser = EscapedPydanticOutputParser(pydantic_object=CatFeaturesOutput)
    user_input = state.get("question_final")
    promt = ChatPromptTemplate.from_messages(
            [("system", task_promt_cat_features)]
        ).partial(output_format=cat_fetures_parser.get_format_instructions(),
                  user_input=user_input
                  )
    chain = promt | model | cat_fetures_parser
    cat_features = try_invoke(chain)
    return Command(goto="rukovod_methodology_node" if (cat_features.team_info == 1 or cat_features.boss_info == 1)\
                    else "add_lang_features_node",
                    update={"cat_features": cat_features,
                            "add_find_teams_promt": "",
                            "add_skills_promt": "",
                            "add_lang_features_promt": ""})

@log_node_execution("rukovod_methodology_node") 
def rukovod_methodology_node(state: GraphState) -> Command:
    """
    Adds a methodology for working with managers (team definition) to GraphState. 
    Deletes industrial information for employees (add_fio_info) Full name TBn.
    """
    add_fio_info_ = ""
    cat_features = state.get("cat_features")
    mydata = state.get("user_data")
    user_id = mydata["employee_id"]
    infoByRuks = state.get("found_names") # list of dictionaries with information on employees from the question
    add_find_teams_promt_ = []
    for sample in infoByRuks:
        found_info = sample["found_names"]
        if len(found_info) == 0: # searching for an employee in the database did not produce results
            continue
        found_info = found_info[0]
        if FIND_COMMAND_WITH_OSS: # False (for now we only work with TBn)
            addfilter = format_filter4ruk_wihtOSS(found_info, sample["fio"])
        else:
            addfilter = format_filter4ruk_wihIPERNR(found_info, sample["fio"])
        add_find_teams_promt_.append(addfilter)

    # formation of industrial projects with methodology
    add_find_teams_promt_ = "  \n".join(add_find_teams_promt_)
    add_find_boss_promt_ = add_find_boss_promt.format(TABNAME=ContextSchema.TABNAME, 
                                                     user_id=user_id, 
                                                     ACTUAL_DATA=ContextSchema.ACTUAL_DATA)
    # determine the necessary methodology
    if (cat_features.team_info == 1 and cat_features.boss_info == 1):
        add_find_teams_promt_ = add_find_teams_promt_ + add_find_boss_promt_
        add_fio_info_ = ""
    elif (cat_features.team_info == 0 and cat_features.boss_info == 1):
        add_find_teams_promt_ = add_find_boss_promt_
        add_fio_info_= state.get("add_fio_info", "")
    # add required columns to the prunner
    prunner_ruk_columns_exceptions = ["lid_tribe_i_pernr", "lid_1_lvl_i_pernr", "lid_2_lvl_i_pernr",
    "lid_3_lvl_i_pernr", "lid_cluster_i_pernr", "it_lid_cluster_i_pernr", "cur_tribe_i_pernr", "po_i_pernr"] 
    return Command(goto="add_lang_features_node", update={"prunner_ruk_columns_exceptions": prunner_ruk_columns_exceptions,
                                                     "add_find_teams_promt": add_find_teams_promt_,
                                                     "add_fio_info": add_fio_info_})

@log_node_execution("add_lang_features_node") 
def add_lang_features_node(state: GraphState) -> Command:
    """
    Adds an additional method (methodology) to GraphState, which explains how to work with languages in the database, if necessary.
    Declares an empty skill-based promt in the graph state; if skills are extracted from the question, we go to the node for creating the methodology 
    on working with skills.
    """
    add_skills_promt_ = ''
    add_lang_features_promt_ = ''
    if state["cat_features"].languages == 1:
        add_lang_features_promt_ = add_lang_features_promt.format(TABNAME=ContextSchema.TABNAME)
    return Command(goto="check_found_skills_node" if state["cat_features"].skills else "get_oss_agile_structure_node", \
                          update={"add_lang_features_promt": add_lang_features_promt_, 
                                  "add_skills_promt": add_skills_promt_})

@log_node_execution("check_found_skills_node") 
def check_found_skills_node(state: GraphState) -> Command:
    """
    Finds similar skills from the table in the vector database, checks their validity, 
    adds validated skills to the industry and a rule on how to work with them.
    """
    skills_map = {}
    for skill in state["cat_features"].skills:
        results = ContextSchema.skills_retriever.invoke(skill, cut=10, need_keys=["skillName"])
        skills_in_data = []
        for res in results:
            skills_in_data.append(res['skillName'])
        skills_map[skill] = skills_in_data

    for key, values in skills_map.items():
        task_promt = skills_task_promt.format(key=key, values=values)
        skills_parser = EscapedPydanticOutputParser(pydantic_object=RelevantSkills)
        promt = ChatPromptTemplate.from_messages([
                ("system", task_promt + "\nWrap the output in ```json``` tags\n{format_instructions}.")
            ]
        ).partial(format_instructions=skills_parser.get_format_instructions())
        chain = promt | model | skills_parser
        output = try_invoke(chain)
        skills_map[key] = output.relevant_skills

    metodology_promt_part = ""
    for skill, find_skills in skills_map.items():
        filters = "%' OR x ILIKE '%".join(find_skills)
        if filters != 'НЕТ':
            metodology_promt_part += \
                f"Для навыка '{skill}' используй БЕЗ ИЗМЕНЕНИЙ фильтр: arrayExists(x -> x ILIKE '%{filters}%', all_skills)\n  "

    add_skills_promt_ = metodology_promt_part
    return Command(goto="get_oss_agile_structure_node", update={"skills_map": skills_map, 
                                                               "add_skills_promt": add_skills_promt_})

@log_node_execution("get_oss_agile_structure_node")
def get_oss_agile_structure_node(state: GraphState) -> Command:
    """
    Extracts structures using promt: linear, agile.
    Finds the nearest ones in the database using a retriever, and after checking adds them to GraphState.
    """
    task_promt_structures = promt_oss_agile_extractor.format(qu=state["question"])

    parser = EscapedPydanticOutputParser(pydantic_object=OssAgileStructure)
    promt = ChatPromptTemplate.from_messages([("system", task_promt_structures +\
                                                "\nWrap the output in ```json``` tags\n{format_instructions}.")])\
                                             .partial(format_instructions=parser.get_format_instructions())
    chain = promt | model | parser
    output_structures = try_invoke(chain)
    val = output_structures.structures
    findInOss = []
    add_oss_promt = ""
    cleaned_found_oss = []
    prunner_oss_columns_exceptions = []
    if len(val) > 0:
        if val[0] != "НЕТ":
            user_data = state.get("user_data")
            retriever = ContextSchema.oss_retriever
            findInOss = [search_structure(fnd, retriever, ["structure_name", "structure_type", "structure_code"])
                            for fnd in val]
            step1_add_promt = check_found_struct(val, findInOss)
            cleaned_found_oss = get_oss_sql_full_data(step1_add_promt, user_data, CUT_OSS_WITH_SIMILARITY)
            add_oss_promt = make_oss_promt2add(cleaned_found_oss)
            add_oss_promt = "\n  ".join(add_oss_promt)
            for key in cleaned_found_oss:
                cleaned_found_oss[key] = cleaned_found_oss[key].to_dict("records")
                lvl = cleaned_found_oss[key][0]['level_name']
                if re.findall(r'[0-9]+', lvl): # linear structure
                    prunner_oss_columns_exceptions.extend(['unit_id_tree', lvl])
                else: # agile structure
                    prunner_oss_columns_exceptions.extend([lvl.split("_")[0] + "_code", lvl])
    # add required columns to the prunner
    prunner_oss_columns_exceptions = list(set(prunner_oss_columns_exceptions)) 
    return Command(goto="column_prunner_node",
                update={"add_oss_promt": add_oss_promt,
                        "prunner_oss_columns_exceptions": prunner_oss_columns_exceptions,
                        "output_oss_di": {"question": state["question"],
                                        "extractedNER": val,
                                        "findInOss": findInOss,
                                        "cleanedFioundOss": cleaned_found_oss}})

@log_node_execution("column_prunner_node")
def column_prunner_node(state: GraphState) -> Command:
    """
    Extracts from the list of attributes only those necessary to answer the question.
    """
    columns_parser = EscapedPydanticOutputParser(pydantic_object=ColumnsPrunningOutput)
    # add required columns (if not, then add 'employee_id' to the required ones)
    columns_exceptions = state["prunner_oss_columns_exceptions"] + state["prunner_ruk_columns_exceptions"]
    columns_exceptions_add_promt = "employee_id"
    if len(columns_exceptions) > 0:
        columns_exceptions_add_promt = ", ".join(columns_exceptions)

    promt = ChatPromptTemplate.from_messages(
            [("system", task_promt_columns)]
        ).partial(output_format=columns_parser.get_format_instructions(),
                    question=state["question_final"],
                    TABLE_DESCRIPTION=TABLE_DESCRIPTION,
                    columns_exceptions=columns_exceptions_add_promt
                    )
    
    chain = promt | model | columns_parser
    prunner_res = try_invoke(chain)

    TABLE_DESCRIPTION_PRUNNED = ",\n".join(
    f"Колонка {column} с типом {value.get('column_type')} содержит {value.get('description')}"
    for column, value in COLUMNS_DESCRIPTION.items() if column in prunner_res.relevant_columns)

    return Command(goto="get_complexity_node",
            update={"prunner_res": prunner_res,
                    "table_description_prunned": TABLE_DESCRIPTION_PRUNNED})