class BaseText2SqlError(Exception):
    stage = "unknown"


class SqlGenerationError(BaseText2SqlError):
    stage = "sqlGeneration"


class CannotAnswerError(BaseText2SqlError):
    stage = "sqlGeneration"


class ExternalNetworkError(BaseText2SqlError):
    stage = "sqlGeneration",


class RLSError(BaseText2SqlError):
    stage = "rlsGeneration"
