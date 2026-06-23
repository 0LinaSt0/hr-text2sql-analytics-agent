import re
import logging
from collections import defaultdict
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from src.hrp_agent_text2sql.schemas.db_info import COLUMNS_DESCRIPTION, AGILE_FIELDS_MAP
from src.hrp_agent_text2sql.errors import RLSError, SqlGenerationError, ExternalNetworkError

logger = logging.getLogger(__name__)

_OPR_MAPPING = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.LT: "<",
    exp.GTE: ">=",
    exp.LTE: "<=",
    exp.ILike: "=",
    exp.Like: ""
}

SPACE_RE = re.compile(r"\s+")


def unwrap(expr: exp.Expression) -> exp.Expression:
    try:
        while isinstance(expr, (exp.Alias, exp.Round, exp.Mul, exp.Div, exp.Cast)):
            expr = expr.this
    except RecursionError:
        logger.error(f"Error processing expression {str(expr)}")
    return expr


def non_aggregates(expression: exp.Expression) -> bool:
    """Function to check presence of a column in an expression"""
    return isinstance(expression, (exp.Column, exp.Star, exp.Literal))


def aggregates(expression: exp.Expression) -> bool:
    """Function to check presence of aggregates in an expression"""
    return isinstance(expression, (exp.AggFunc,))


def _has_in_sql(sql: str, check_func: callable) -> bool | None:
    """Iterate over SELECT sql expressions with the check function"""
    expression = sqlglot.parse_one(sql, read="clickhouse")
    if not isinstance(expression, exp.Select):
        return None
    for expr in expression.expressions:
        expr = unwrap(expr)
        if check_func(expr) is True:
            return True
    return False


def get_where_columns(sql: str, operators: list[str] = None, filter_literal: bool = True) -> list[str]:
    """Get list of columns from WHERE clause
    :param operators: - list of operators to filter columns by, all by default
    :param filter_literal: - only take columns used with operator for non-columns"""
    operators = operators or []
    expression = sqlglot.parse_one(sql, read="clickhouse")
    where_expr = expression.find(exp.Where)
    if not where_expr:
        return []

    def collect_columns(node):
        columns = set()
        if isinstance(node, exp.Column):
            column_name = node.name
            columns.add(column_name)
        elif hasattr(node, 'args'):
            if not operators or (
                    not isinstance(node, exp.Predicate)
                    or _OPR_MAPPING.get(type(node)) in operators
            ):
                for arg in node.args.values():
                    if isinstance(arg, list):
                        for item in arg:
                            columns.update(collect_columns(item))
                    else:
                        if filter_literal and isinstance(arg, exp.EQ):
                            left_ = arg.left.this.this if isinstance(arg.left.this, exp.Identifier) else arg.left.this
                            right_ = arg.right.this.this if isinstance(arg.right.this, exp.Identifier) else arg.right.this
                            if left_ in COLUMNS_DESCRIPTION.keys() and right_ in COLUMNS_DESCRIPTION.keys():
                                continue
                        columns.update(collect_columns(arg))
        return columns
    return list(collect_columns(where_expr.this))


def get_group_by_columns(sql: str) -> list[str]:
    """Get list of columns from GROUP BY"""
    expression = sqlglot.parse_one(sql, read="clickhouse")
    group_by_expr = expression.find(exp.Group)
    if group_by_expr is None:
        return []
    grouped = group_by_expr.args.get("expressions", []) if group_by_expr.this is None else group_by_expr.this.columns
    return [column.name for column in grouped] if grouped else []


def has_sql_non_aggregates(sql: str) -> bool | None:
    """In SQL, the returned columns contain fields that are not aggregates"""
    return _has_in_sql(sql, non_aggregates)


def has_sql_aggregates(sql: str) -> bool | None:
    """In SQL, the returned columns have aggregate fields"""
    return _has_in_sql(sql, aggregates)


def validate_query(sql_query: str):
    """Validation - such a request is allowed to be executed in ClickHouse"""
    logger.info("validate_query:\n%s", sql_query)
    try:
        parsed_statements = sqlglot.parse(sql_query, read="clickhouse")
    except SqlglotError as e:
        raise SqlGenerationError(str(e)) from e

    if not parsed_statements or len(parsed_statements) != 1:
        raise SqlGenerationError("Failed to parse request or more than one request was generated")

    statement = parsed_statements[0]
    used_tables = set()
    for table in statement.find_all(exp.Table):
        if not table.name:
            # for table functions: FROM cluster(datalab, db.table), FROM merge(db.table) etc.
            table_function = table.parts[0].name
            raise SqlGenerationError(
                f"It is prohibited to use table functions to read from tables, in particular {table_function}",
            )
        used_tables.add(table.name)

    for query in statement.find_all(exp.Query):
        if "settings" in query.args:
            raise SqlGenerationError(
                "It is prohibited to use the SETTINGS block both in the main request and in subqueries",
            )


def contain_user_identifier_columns(sql_query: str) -> bool:
    """WHERE / GROUP BY contains a uniquely identifying column or group of columns"""
    unique_columns = [column for column, value in COLUMNS_DESCRIPTION.items() if value.get("is_unique")]
    search_columns = get_where_columns(sql_query, ["="])
    search_columns.extend(get_group_by_columns(sql_query))
    if any(column in unique_columns for column in search_columns):
        return True

    unique_group_columns = [
        [column, *value["is_unique_group"]] for column, value in COLUMNS_DESCRIPTION.items() if
        value.get("is_unique_group")
    ]
    for unique_group in unique_group_columns:
        if all(
            column in search_columns
            for column in unique_group
        ):
            return True
    return False


def extract_final_select_columns(sql_query: str) -> list[str]:
    """
    Retrieves column names from the final SELECT query, including CTE. 
    Takes into account aliases (AS), returns names after AS, if any.
    """
    parsed = sqlglot.parse_one(sql_query, dialect="clickhouse")  # dialect можно поменять при необходимости

    def extract_columns_from_select(select: exp.Select) -> list[str]:
        result = []
        for expression in select.expressions:
            if isinstance(expression, exp.Alias):
                # Если есть AS — берём исходное название
                result.append(expression.this.name)
            elif isinstance(expression, exp.Identifier):
                # Простая колонка без AS
                result.append(expression.name)
            elif isinstance(expression, exp.AggFunc):
                result.append(expression.this.name)
            else:
                result.append(str(expression))
        return result

    # Если есть CTE (WITH), ищем последний (основной) SELECT
    with_exp = parsed.args.get("with")
    if with_exp:
        # Основной SELECT после WITH
        main_select = with_exp.parent
        return extract_columns_from_select(main_select)
    else:
        # Простой SELECT без CTE
        return extract_columns_from_select(parsed)


def contain_private_columns(sql_query: str) -> bool:
    """The request contains private columns as a result"""
    private_columns = [column for column, value in COLUMNS_DESCRIPTION.items() if not value.get("public")]
    return any(column in private_columns for column in extract_final_select_columns(sql_query))


def can_return_result(sql_query: str, is_private_network: bool) -> bool:
    """Determines whether the query result can be returned to the user 
    If the user comes from a public network and requests private fields - False 
    If the user came from a private network, you can return the result - True
    """
    if is_private_network:
        return True
    if contain_private_columns(sql_query):
        logger.warning("PWA: cannot return result: access to private columns from public network")
        raise ExternalNetworkError("PWA: cannot return result: access to private columns from public network")
    return True


def need_apply_rls(sql_query: str, employee_id: str, is_private_network: bool) -> bool:
    """Determines whether RLS should be applied, based on the following rules in order: 
    - if the user is from the white list, RLS is not applied; 
    - if the user does not come from a private network, RLS is applied; 
    - if there are no private columns in the request, RLS is not applied; 
    - if the query contains private columns AND the query condition contains a filter that allows you to uniquely identify one row in the database - RLS is applied;"""
    private_columns = [column for column, value in COLUMNS_DESCRIPTION.items() if not value.get("public")]

    if employee_id in {"2132091", "4054", "1534432", "2065650", "1435923"}:
        return False
    if not is_private_network:
        logger.info(f"rewrite needed: Access from public network")
        return True

    if not any(column in private_columns for column in extract_final_select_columns(sql_query)):
        logger.info("No rewrite needed: querying public data only")
        return False

    if contain_user_identifier_columns(sql_query):
        logger.info(f"rewrite needed: Contains user identifier columns")
        return True

    if has_sql_non_aggregates(sql_query) is False:
        logger.info("No rewrite needed: response contains only aggregates")
        return False

    logger.info(f"rewrite needed: Contain private columns without aggregates")
    return True


def build_rls_condition(constraints) -> str:
    if not (constraints.linear or constraints.agile):
        raise RLSError("No constraints found")

    logger.info("building RLS query condition for user constraint: %s", constraints)
    conditions = []

    if constraints.linear:
        conditions.append(f"""
        arrayExists(
            uid -> transform(
                (company, uid),
                [{", ".join(f"('{c.company.lower()}', '{c.unit_id}')" for c in constraints.linear)}],
                [{", ".join(["1"] * len(constraints.linear))}],
                0
            ),
            unit_id_tree
        )
        """)

    group_agile_by_type = defaultdict(list)
    for constraint in constraints.agile:
        group_agile_by_type[constraint.unit_type].append(constraint.unit_id)

    for unit_type, unit_list in group_agile_by_type.items():
        conditions.append(
            f""" {AGILE_FIELDS_MAP[unit_type]} IN ({",".join(f"'{unit}'" for unit in unit_list)}) """,
        )

    return SPACE_RE.sub(" ", " OR ".join(conditions)).strip()


def rewrite_query(sql_query: str, constraints, table_name: str):
    try:
        rls_conditions_str = build_rls_condition(constraints)
    except Exception as e:
        raise RLSError("RLS condition failed with error: %s", e) from e

    return sql_query.replace(
        f"{table_name}",
        f" (SELECT * FROM {table_name} WHERE {rls_conditions_str}) as t ",
    )
