import json
import re
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.utils.pydantic import TBaseModel
from langchain_core.exceptions import OutputParserException


class EscapedPydanticOutputParser(PydanticOutputParser):
    """We correctly process all special characters with slashes so that there are no problems with conversion to json"""

    @staticmethod
    def escape_special_chars(text):
        # Find all string values ​​and escape special characters
        pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'

        def escape_match(match):
            string_content = match.group(1)
            # Escaping special characters
            escaped = string_content.encode('unicode_escape').decode('utf-8')
            # Removing unnecessary escaping for already escaped characters
            escaped = escaped.replace('\\\\', '\\')
            return f'"{escaped}"'

        return re.sub(pattern, escape_match, text)

    def parse(self, text: str) -> TBaseModel:
        text = self.escape_special_chars(text)
        return super().parse(text)
    

def try_invoke(chain, retries=1):
    """
    Simple chain call with retries in case of parsing errors.
    
    Args:
        chain: LangChain chain.
        retries: Number of additional retries (default 1).
    
    Returns:
        The result of the chain execution.
    """
    last_error = None
    
    for attempt in range(retries + 1):
        try:
            return chain.invoke({})
        except (OutputParserException, json.JSONDecodeError, Exception) as e:
            last_error = e
            
            # Checking whether it needs to be repeated
            error_msg = str(e).lower()
            is_parsing_error = any(
                keyword in error_msg 
                for keyword in ['json', 'parser', 'output', 'format', 'decode', 'invalid']
            )
            
            if not is_parsing_error:
                raise # Not a parsing error -exit immediately
            
            if attempt == retries:
                raise last_error  # Running out of attempts
