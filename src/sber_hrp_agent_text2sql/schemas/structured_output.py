from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import Dict, List, Literal


class FeatureCheckerOutput(BaseModel):
    """
    Check result.
    """
    justification: Optional[str] = Field(description="Обоснование.")
    verdict: int = Field(desc="Флаг означающий, возможно ли ответить на вопрос.", ge=0, le=1)


class SelfCheckerOutput(BaseModel):
    """
    Check result.
    """
    analysis: str = Field(description="Reasoning and analysis.")
    verdict: int = Field(desc="Flag indicating whether we are talking about something of mine.", ge=0, le=1)


class SQLCreatorOutput(BaseModel):
    """
    Paraphrased user question, SQL query and description of output fields.
    An example of the correct format for sql_meta:
    {"user_id": "Unique user identifier", "total_spent": "Total amount of money spent"}
    """
    question: str = Field(description="Paraphrased user question")
    sql: str = Field(description="SQL QUERY")
    sql_meta: Dict[str, str] = Field(description="List with field descriptions in the format: 'field_name' -'description'")

class CumulativeSQLOutput(BaseModel):
    """
    Analysis, SQL query and description of output fields.
    An example of the correct format for sql_meta:
    {"user_id": "Unique user identifier", "total_spent": "Total amount of money spent"}
    """
    analysis: str = Field(description="Analysis, details and rules")
    sql: str = Field(description="Combined SQL QUERY")
    sql_meta: Dict[str, str] = Field(description="""List with a description of the fields of the final SQL QUERY in the format:
                                        'field_name' -'description'""")

class DecomposeSQLCreatorOutput(BaseModel):
    """
    Analysis, SQL query and description of output fields.
    An example of the correct format for sql_meta:
    {"user_id": "Unique user identifier", "total_spent": "Total amount of money spent"}
    """
    analysis: str = Field(description="Analysis, details and rules")
    sql: str = Field(description="SQL QUERY")
    sql_meta: Dict[str, str] = Field(description="List with field descriptions in the format: 'field_name' -'description'")

class SQLRecreatorOutput(BaseModel):
    """
    Analysis of the fix and a new SQL query describing the output fields.
    An example of the correct format for sql_meta:
    {"user_id": "Unique user identifier", "total_spent": "Total amount of money spent"}
    """
    analysis: str = Field(description="What was corrected and why: reasoning and analysis.")
    fixed_sql: str = Field(description="CORRECTED SQL QUERY")
    sql_meta: Dict[str, str] = Field(description="""List with a description of the fields of the CORRECTED SQL QUERY in the format:
                                        'field_name' -'description'""")
    

class CatFeaturesOutput(BaseModel):
    """
    Extracted categories from user question
    """
    info_expl: str = Field(description="""
                           Explanation of the rating given for the flag for mentioning the leader, for the flag for mentioning natural languages, and
                           skills found in the question.
                           """)
    languages: Literal[0, 1] = Field(description="Is the query talking about natural languages [0, 1]")
    skills: Optional[List[str]] = Field(description="skills, abilities, areas of knowledge, competencies")
    team_info: Literal[0, 1] = Field(description="mention of a specific manager, as part of the search for a team or department [0, 1]")
    boss_info: Literal[0, 1] = Field(description="mention of searching or obtaining information about the manager of a specific employee [0, 1]")


class RelevantSkills(BaseModel):
    """
    Results of checking the relevance of the found skills
    """
    relevant_skills: List[str] = Field(description="Relevant skills, abilities, areas of knowledge, competencies or 'NO'")


class OssAgileStructure(BaseModel): 
    """
    All names of structures extracted from the sentence
    """
    analysis: str = Field(description="What and why extracted: reasoning and analysis.")
    structures: List[str] = Field(description="""Names of specific structures, blocks, divisions, tribes, clusters, centers,
                                   divisions and company names or 'NO' if nothing is found.""")

class FIOIDFromInputOutput(BaseModel):
    output: List[Optional[Dict[Literal["fio", "tabel_num"], str]]] = Field(description="""
        The corresponding full name (full name) and personnel numbers (TN -from 3 to 8 digits) extracted from the user's request
                                                                 """)


class FullFioOutput(BaseModel):
    output: Dict[int, str] = Field(desc=f"""
        Dictionary with full name index and edited full name in EXPANDED, FULL format,
            in the nominative case and singular.
    """)


class ColumnsPrunningOutput(BaseModel):
    """ 
    Explanation and list of columns.
    """
    analysis: str = Field(description="What data, attributes, columns are needed for a detailed answer to the question -reasoning and analysis.")
    relevant_columns: List[str] = Field(description="Column titles.")


class ComplexityFlagOutput(BaseModel):
    explanation: str = Field(desc="Explanation of the reasons for the request complexity flag.")
    verdict: int = Field(desc="Flag indicating that the request specified by the client is 'complex'.", ge=0, le=1)


class SimplePlanOutput(BaseModel):
    output: Dict[str, Dict[Literal["step_desc", "tab_name"], str]] = Field(desc=f"""
        A dictionary that contains a set of steps, of type step_<num>, as keys and values
        a dictionary describing the current step of the plan -step_desc and the name of the table in which the result will be written -tab_name.""")
