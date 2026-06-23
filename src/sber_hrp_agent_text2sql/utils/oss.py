import re
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict

from src.hrp_agent_text2sql.gigamodels import model
from src.hrp_agent_text2sql.promts.tasks import promt_check_equality 
from src.hrp_agent_text2sql.utils.parser_processing import EscapedPydanticOutputParser
from src.hrp_agent_text2sql.schemas.agent_context import CustomRetriver, client, ContextSchema
from src.hrp_agent_text2sql.utils.person import find_nearest_by_oss


class OssOutputIndexFormat(BaseModel):
    output_index_true: int = Field("Index of the selected structure or -1 if none of the below fit", ge=-1)

    
def search_structure(query: str, ensemble_retriever: CustomRetriver, listOfCols: List[str]) -> List[Dict]:
    """
    Searches for a structure by name and returns the result with complete information
    
    Args:
        query: Structure from the question.
        ensemble_retriever: Retriever.
        listOfCols: Metadata about the structure.
    
    Returns:
        A list of metadata dictionaries about the structure, the 10 closest ones found in the vector database.
    """
    docs = ensemble_retriever.invoke(query, need_keys=listOfCols)
    
    results = []
    for doc in docs[:10]:
        result = {
            'stucture_name': doc['structure_name'],
            'structure_type': doc['structure_type'],
            'structure_code': doc['structure_code'],
        }
        results.append(result)
    
    return results

def check_found_struct(extractedNER: List, findInOss: List[List[dict]]) -> dict:
    """
    The function of checking extracted entities for compliance with the divisions found in the vector database.
    Highlights all matching values.
    
    Args:
        extractedNER: List of structures extracted from the question.
        findInOss: Structures found by the retriever.
    
    Returns:
        Dictionary of validated structures.
    """
    zipped = list(zip(extractedNER, findInOss))
    aggregate2str = lambda li, key: "\n".join([f"Index {i}: {j[key]}" for i, j in enumerate(li)])

    parser = EscapedPydanticOutputParser(pydantic_object=OssOutputIndexFormat)
    promt = ChatPromptTemplate.from_messages([("system", promt_check_equality)])
    final_promt = promt.partial(output_format=parser.get_format_instructions())
    chain = final_promt | model | parser
    chosens = {ner: {"variants": variants,
                    "found_index": chain.invoke({"fnd": ner,
                                    "listOf": aggregate2str(variants, "stucture_name")}).output_index_true}
                for ner, variants in zipped}
    # I will determine the index of the unit that is suitable in the opinion of the gig
    chosens = {ner: [chosens[ner]["variants"][chosens[ner]["found_index"]]["stucture_name"],
                        chosens[ner]["variants"]
                    ] for ner in chosens 
                    if chosens[ner]["found_index"] != -1}
    # because search by name, we look for duplicates by name, we will choose from all
    chosens = {ner: [smp for smp in chosens[ner][1] if smp["stucture_name"] == chosens[ner][0]] for ner in chosens}
    return chosens

def get_oss_sql_full_data(input_dict: Dict[str, List], user_data: Dict, cut: int = 1) -> dict:
    """
    Forms a pool of full paths of found and verified OSH/agile entities
    
    Args:
        input_dict: List of structures extracted from the question.
        findInOss: Structures found by the retriever.
    
    Returns:
        Dictionary of validated structures.
    """
    sql_format = """
        SELECT distinct unit_id_tree, lvl_01_org_name,
            lvl_02_org_name, lvl_03_org_name, lvl_04_org_name,
            lvl_05_org_name, tribe_name, cluster_name, team_name,
            tribe_code, cluster_code, team_code
        FROM {tabname}
        where {level_name} = '{oss_agile_name}'
    """
    fin_pool_of_oss = dict()
    for key in input_dict:
        prom = pd.DataFrame()
        for sample in input_dict[key]:
            postoss = client.query(sql_format.format(level_name=sample["structure_type"],
                                                    oss_agile_name=sample["stucture_name"],
                                                    tabname=ContextSchema.TABNAME))
            data_oss = pd.DataFrame(postoss.result_rows, columns=postoss.column_names)
            data_oss["level_name"] = sample["structure_type"]
            prom = pd.concat([prom, data_oss], axis=0)
        prom.reset_index(drop=True, inplace=True)
        fin_pool_of_oss[key] = prom
    # Optional part with clipping, you can remove it to evaluate the full functionality
    fin_pool_of_oss = {key: find_nearest_by_oss(user_data, fin_pool_of_oss[key], cut) for key in fin_pool_of_oss}
    return fin_pool_of_oss

def make_oss_promt2add(final_clear_oss_data: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Generates a list of prompts for each structure from the question.
    
    Args:
        final_clear_oss_data: Dictionary of validated structures.
    
    Returns:
        List of promts.
    """
    output_promts = dict()
    for key in final_clear_oss_data:
        sample = final_clear_oss_data[key].to_dict("records")[0]
        lvl = sample["level_name"]
        regres = re.compile("[0-9]+").findall(lvl)
        if len(regres) > 0:
            regres = int(regres[0])
            col_code = sample["unit_id_tree"][regres - 1]
            filter_form = f"unit_id_tree[{regres}] = '{col_code}'"
        else:
            col_code = lvl.split("_")[0] + "_code"
            filter_form = f"{col_code} = {sample[col_code]}"
        output_promts[key] = filter_form
    frmt = """Для определения подразделения '{key}', БЕЗ ИЗМЕНЕНИЙ вставь следующий фильтр в условие WHERE.
    ФИЛЬТР (СТРОГО НЕ МЕНЯТЬ, НЕ РАСШИРЯТЬ, НЕ ПЕРЕФОРУЛИРОВАТЬ): {filt}""" 
    output = [frmt.format(key=key, filt=output_promts[key]) for key in output_promts]
    return output
