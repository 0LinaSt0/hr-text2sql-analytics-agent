import clickhouse_connect
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from src.hrp_agent_text2sql.gigamodels import model_emb

client = clickhouse_connect.get_client(host='YOUR_HOST',
                                    port=8443,
                                    username = 'YOU_LOGIN',
                                    password = '************',
                                    verify = 'False')

vector_bases = {"oss": json.load(open(r"src/hrp_agent_text2sql/custom_retrievers/oss_ebase_embs.json")), 
                "skills": json.load(open(r"src/hrp_agent_text2sql/custom_retrievers/skills_ebase_embs.json"))}


class CustomRetriver:
    def __init__(self, base_, model_embed): 
        self.base = base_
        self.embeddings = np.array([i["embeddings"] for i in self.base], dtype="float32")
        self.model_embed = model_embed

    def invoke(self, text, cut=10, need_keys=["stucture_name", "structure_type"]):
        embed_query = np.array(self.model_embed.embed_query(text))
        embed_query = np.expand_dims(embed_query, axis=0)
        csn = cosine_similarity(embed_query, self.embeddings)
        csn = csn.squeeze()
        argsorted = np.argsort(csn)[-cut: ]
        output = [self.base[i] for i in argsorted]
        output = output[::-1]
        output = [{key: sample[key] for key in need_keys} for sample in output]
        return output


def load_structure_retriever(loaded_from=vector_bases["oss"], embedding_model=model_emb) -> CustomRetriver:
    """
    Loads the retriever using OSHS
    """
    retriever = CustomRetriver(base_=loaded_from, model_embed=embedding_model)
    return retriever

def load_skills_retriver(loaded_from=vector_bases["skills"], embedding_model=model_emb) -> CustomRetriver:
    """
    Loads retriever by skill
    """
    retriever = CustomRetriver(base_=loaded_from, model_embed=embedding_model)
    return retriever


@dataclass
class ContextSchema:
    click_client = client
    feature_checker_retry = 2
    sql_query_retry = 1
    TABNAME = "anagent.history_v0"
    ACTUAL_DATA = (pd.to_datetime("now")).__str__()[:10] #(pd.to_datetime("now") - pd.offsets.MonthEnd(1)).__str__()[:10] 
    skills_retriever = load_skills_retriver()
    oss_retriever  = load_structure_retriever()
    user_access_token = None
    user_constraints = None
    employee_id = None
