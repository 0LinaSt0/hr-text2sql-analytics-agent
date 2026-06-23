import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Union, List

from src.hrp_agent_text2sql.promts.tasks import promt_find_fio_i_pernr, task_full_name_promt
from src.hrp_agent_text2sql.gigamodels import model
from src.hrp_agent_text2sql.utils.parser_processing import EscapedPydanticOutputParser, try_invoke
from src.hrp_agent_text2sql.schemas.agent_context import client, ContextSchema
from src.hrp_agent_text2sql.schemas.structured_output import FIOIDFromInputOutput, FullFioOutput

def find_nearest_by_oss(my_data: Dict, found_df: pd.DataFrame, cut: int = 1) -> pd.DataFrame:
    """
    Search for the closest (by line and agile) of the found employees to the user.
    
    Args:
        my_data: TBn and full name of the user.
        found_df: TBn and structure of employees found in the database by full name.
        cut: Number of employees to return (by default, the closest in the line and agile).
    
    Returns:
        DataFrame, TBn and the structure of the employee closest in structure.
    """
    i_pernr = my_data["employee_id"]
    query = client.query(f"""
        SELECT distinct employee_id, unit_id_tree, tribe_code, cluster_code, team_code
        FROM {ContextSchema.TABNAME}
        where 1 = 1
            and employee_id = {i_pernr}
        """)   
    my_df = pd.DataFrame(query.result_rows, columns=query.column_names) # user linear structure
    assert len(my_df) > 0, "User not found!"
    assert len(found_df) > 0, "The required employee was not found!"
    # user linear structures
    myval = my_df.to_dict("records")
    myset = set(myval[0]["unit_id_tree"])
    # Agile user structure
    myagile = set([str(myval[0]['tribe_code']), str(myval[0]['cluster_code']), str(myval[0]['team_code'])])
    myagile.discard('0') 

    foundval =  found_df.to_dict("records")
    for sample in foundval:
        # employee line structures
        set_found = set(sample.get("unit_id_tree"))
        # Agile employee structure
        agile_found =  set([str(sample.get("tribe_code")), str(sample.get("cluster_code")), str(sample.get("team_code"))])
        agile_found.discard('0') 
        # Intersections
        sample["intersection"] = len(set_found & myset) + 2*(len(agile_found & myagile))
    output = pd.DataFrame(foundval).sort_values(["intersection"], ascending=False) 
    return output[:cut]

def get_fullName(lst: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Additional verification to bring your full name into an expanded, full format.

    Args:
        lst: List of dictionaries with full name and TBn.
    
    Returns:
        List from dictionaries, by key full name, expanded, full name format.
    """
    ln = len(lst)
    fio2insert_di = {i + 1: lst[i]['fio'] for i in range(ln) if lst[i]['fio'] != 'not found'}
    fio2insert = [f"{i + 1}: {lst[i]['fio']}" for i in range(ln) if lst[i]['fio'] != 'not found']
    fio2insert_str = "\n".join(fio2insert)
    promt = ChatPromptTemplate.from_messages([
        ("system", task_full_name_promt)
    ])
    
    parser = EscapedPydanticOutputParser(pydantic_object=FullFioOutput)
    final_promt = promt.partial(output_format=parser.get_format_instructions(),
                                fio=fio2insert_str)
    chain = final_promt | model | parser
    respond = try_invoke(chain)
    respond = respond.output
    respond = {fio2insert_di[i]: respond[i] for i in respond}
    return respond

def get_fio_i_pernr_from_input(user_input: str) -> Dict[str, str]:
    """
    Parsing for full name and TBn.
    
    Args:
        user_input: Question from a user.
    
    Returns:
        List from dictionaries with full name and TBn.
    """
    promt = ChatPromptTemplate.from_messages([
        ("system", promt_find_fio_i_pernr) 
    ])
    parser = EscapedPydanticOutputParser(pydantic_object=FIOIDFromInputOutput)
    final_promt = promt.partial(output_format=parser.get_format_instructions(),
                                user_input=user_input)
    chain = final_promt | model | parser
    respond = try_invoke(chain)
    respond = respond.output
    if respond:
        fullNames = get_fullName(respond)
        for i in respond:
            i["fio"] = fullNames.get(i["fio"], "not found")
    return respond

def get_i_pernr_from_base(dict_sample_of_person: Dict[str, str]) -> pd.DataFrame:
    """
    Search for employees found in the user's question in the database. Priority by TBn, if not, search by full name (may return 
    several values with the same full name).
    
    Args:
        dict_sample_of_person: A dictionary with the full name and personal identification number of the employee found in the question.
    
    Returns:
        DataFrame, information found on the employee (TBn and the structure in which he works).
    """
    if (dict_sample_of_person["fio"] == "not found") \
            and (dict_sample_of_person["tabel_num"] == "not found"):
        return pd.DataFrame()
    if dict_sample_of_person["tabel_num"] != "not found":
        val = dict_sample_of_person["tabel_num"]
        filter_format = f"employee_id = {val}" 
    else:
        val = dict_sample_of_person["fio"]
        val = val.split()
        params = "[person_surname, person_name, person_patronimics]"
        filter_format = f"length(arrayDistinct(arrayIntersect({params}, {val}))) == length({val})"

    seekInBase = client.query(f"""
        SELECT distinct employee_id,
            unit_id_tree,
            tribe_code, 
            cluster_code,  
            team_code,
            tribe_name,      
            team_name
        FROM {ContextSchema.TABNAME}
        where 1 = 1
            and {filter_format}
    """)   
    seekInBaseDf = pd.DataFrame(seekInBase.result_rows, columns=seekInBase.column_names)
    return seekInBaseDf

def format_filter4ruk_wihtOSS(ruk_dict: Dict, nameOfRuk: Union[str, None]) -> str:
    columns = ["unit_id_tree", 
               "tribe_code", 
               "cluster_code", 
               "team_code"]
    ruk_dict_filtered = {key: ruk_dict[key] for key in columns if (ruk_dict[key] != 0)}
    prefixval = "Для сотрудника, указанного в запросе пользователя - "
    if not nameOfRuk is None:
         prefixval = prefixval + f"ФИО: {nameOfRuk}, "
    prefixval = prefixval + f"его табельный номер: {ruk_dict['employee_id']}\n"
    prefixval = prefixval + """  Для определения членов его команды ТОЧНО И БЕЗ ИЗМЕНЕНИЙ вставь следующий блок в условие WHERE.
    ФИЛЬТР (СТРОГО НЕ МЕНЯТЬ, НЕ РАСШИРЯТЬ, НЕ ПЕРЕФОРМУЛИРОВАТЬ):"""
    oss = f"and (unit_id_tree = {ruk_dict_filtered['unit_id_tree']}"
    agile = []
    for key in ["tribe_code", "cluster_code", "team_code"]:
        if key in ruk_dict_filtered:
            val = f"{key} = {ruk_dict_filtered[key]}"
            agile.append(val)
    if len(agile) > 0:
        agile = "or (" + "\n and ".join(agile) + ")"
    else:
        agile = ""
    prefixval = prefixval + "\n" + oss + "\n" + agile + ")"
    return prefixval

def format_filter4ruk_wihIPERNR(ruk_dict: Dict, nameOfRuk: Union[str, None]) -> str:
    """
    Creates a methodological project, with a dynamic example, as determined by the employee’s team.
    
    Args:
        ruk_dict: TBn and structure.
        nameOfRuk: Full name.
    
    Returns:
        Promt with methodology.
    """
    columns = ["lid_tribe_i_pernr",
               "lid_1_lvl_i_pernr",
               "lid_2_lvl_i_pernr",
               "lid_3_lvl_i_pernr",
               "lid_cluster_i_pernr",
               "it_lid_cluster_i_pernr",
               "cur_tribe_i_pernr",
               "po_i_pernr"]
    prefixval = "Для сотрудника, указанного в запросе пользователя - "
    if not nameOfRuk is None:
         prefixval = prefixval + f"ФИО: {nameOfRuk}, "
    prefixval = prefixval + f"его табельный номер: {ruk_dict['employee_id']}\n"
    prefixval = prefixval + """  Для определения членов его команды ТОЧНО И БЕЗ ИЗМЕНЕНИЙ вставь следующий блок в условие WHERE.
    ФИЛЬТР (СТРОГО НЕ МЕНЯТЬ, НЕ РАСШИРЯТЬ, НЕ ПЕРЕФОРМУЛИРОВАТЬ):"""
    rukovod = []
    for sm in columns:
        val = f"{sm} = {ruk_dict['employee_id']}"
        rukovod.append(val)
    rukovod_str = "\n or ".join(rukovod)
    prefixval = prefixval + "\n" + rukovod_str
    return prefixval
