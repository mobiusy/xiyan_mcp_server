import argparse
import logging
import os
import signal
import sys
from typing import Dict, Any

import yaml  # 添加yaml库导入
from mcp.server import FastMCP
from mcp.types import TextContent

from .database_env import DataBaseEnv
from .utils.db_config import DBConfig
from .utils.db_source import HITLSQLDatabase
from .utils.db_util import init_db_conn
from .utils.file_util import extract_sql_from_qwen
from .utils.llm_util import call_openai_sdk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("xiyan_mcp_server")


# Handle SIGINT (Ctrl+C) gracefully
def signal_handler(sig, frame):
    print("Shutting down server gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_yml_config():
    config_path = os.getenv(
        "YML", os.path.join(os.path.dirname(__file__), "config_demo.yml")
    )
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found.")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing configuration file {config_path}: {exc}")
        raise


def get_xiyan_config(db_config):
    dialect = db_config.get("dialect", "mysql")

    if dialect.lower() == "sqlite":
        xiyan_db_config = DBConfig(
            dialect=dialect,
            db_path=db_config.get("db_path"),
        )
    else:
        xiyan_db_config = DBConfig(
            dialect=dialect,
            db_name=db_config["database"],
            user_name=db_config["user"],
            db_pwd=db_config["password"],
            db_host=db_config["host"],
            port=db_config["port"],
        )
    return xiyan_db_config

def build_db_registry(config: dict) -> Dict[str, DBConfig]:
    """Build a registry of project_id -> DBConfig.
    Supports both single 'database' and multi 'databases' in YAML for backward compatibility.
    """
    registry: Dict[str, DBConfig] = {}

    # Multi-project config
    databases_map = config.get("databases")
    if isinstance(databases_map, dict) and databases_map:
        for pid, db_conf in databases_map.items():
            try:
                registry[pid] = get_xiyan_config(db_conf)
            except Exception as e:
                logger.error(f"Invalid database config for project '{pid}': {e}")

    # Backward compatibility: single database
    single_db_conf = config.get("database")
    if single_db_conf:
        try:
            # Use 'default' as the project_id for single-db config
            registry.setdefault("default", get_xiyan_config(single_db_conf))
        except Exception as e:
            logger.error(f"Invalid single database config: {e}")

    if not registry:
        raise ValueError("No database configuration found. Please provide 'database' or 'databases' in YAML.")

    return registry


global_config = get_yml_config()
mcp_config = global_config.get("mcp", {})
model_config = global_config["model"]

# Build registry and default project
DB_REGISTRY: Dict[str, DBConfig] = build_db_registry(global_config)
DEFAULT_PROJECT_ID: str = (
    global_config.get("default_project_id")
    or ("default" if "default" in DB_REGISTRY else next(iter(DB_REGISTRY.keys())))
)

# Default DB config and dialect
default_db_config_obj: DBConfig = DB_REGISTRY[DEFAULT_PROJECT_ID]
dialect = default_db_config_obj.dialect

# Simple caches to avoid re-building engines and schema per request
ENGINE_CACHE: Dict[str, Any] = {}
DB_SOURCE_CACHE: Dict[str, HITLSQLDatabase] = {}


def get_engine(project_id: str):
    """Get or create cached SQLAlchemy Engine for a project."""
    pid = project_id if project_id in DB_REGISTRY else DEFAULT_PROJECT_ID
    if pid in ENGINE_CACHE:
        return ENGINE_CACHE[pid]
    engine = init_db_conn(DB_REGISTRY[pid])
    ENGINE_CACHE[pid] = engine
    return engine


def get_db_source(project_id: str) -> HITLSQLDatabase:
    """Get or create cached HITLSQLDatabase (with built MSchema) for a project."""
    pid = project_id if project_id in DB_REGISTRY else DEFAULT_PROJECT_ID
    if pid in DB_SOURCE_CACHE:
        return DB_SOURCE_CACHE[pid]
    engine = get_engine(pid)
    db_name_opt = DB_REGISTRY[pid].db_name
    db_source = HITLSQLDatabase(engine, db_name=db_name_opt)
    DB_SOURCE_CACHE[pid] = db_source
    return db_source


mcp = FastMCP("xiyan", **mcp_config)


@mcp.resource(
    dialect
    + "://"
    + (
        (default_db_config_obj.db_path or "")
        if dialect.lower() == "sqlite"
        else "/" + (default_db_config_obj.db_name or "")
    )
)
async def read_resource() -> str:
    """List default project's schema as a resource (backward compatible)."""
    db_source = get_db_source(DEFAULT_PROJECT_ID)
    return db_source.mschema.to_mschema()


@mcp.resource(dialect + "://{table_name}")
async def read_resource(table_name) -> str:
    """Read default project's table contents (backward compatible)."""
    try:
        db_source = get_db_source(DEFAULT_PROJECT_ID)
        records, columns = db_source.fetch_with_column_name(
            f"SELECT * FROM {table_name}"
        )
        result = [",".join(map(str, row)) for row in records]
        return "\n".join([",".join(columns)] + result)
    except Exception as e:
        raise RuntimeError(f"Database error: {str(e)}")


# New resources: project-aware schema and table sampling
@mcp.resource("db://{project_id}")
async def read_project_schema(project_id) -> str:
    """List schema of the specified project."""
    db_source = get_db_source(str(project_id))
    return db_source.mschema.to_mschema()


@mcp.resource("db://{project_id}/{table_name}")
async def read_project_table(project_id, table_name) -> str:
    """Read table contents from the specified project."""
    try:
        db_source = get_db_source(str(project_id))
        records, columns = db_source.fetch_with_column_name(
            f"SELECT * FROM {table_name}"
        )
        result = [",".join(map(str, row)) for row in records]
        return "\n".join([",".join(columns)] + result)
    except Exception as e:
        raise RuntimeError(f"Database error: {str(e)}")


def sql_gen_and_execute(db_env: DataBaseEnv, query: str):
    """
    Transfers the input natural language question to sql query (known as Text-to-sql) and executes it on the database.
     Args:
        query: natural language to query the database. e.g. 查询在2024年每个月，卡宴的各经销商销量分别是多少
    """

    # db_env = context_variables.get('db_env', None)
    prompt = f"""你现在是一名{db_env.dialect}数据分析专家，你的任务是根据参考的数据库schema和用户的问题，编写正确的SQL来回答用户的问题，生成的SQL用``sql 和```包围起来。
【数据库schema】
{db_env.mschema_str}

【问题】
{query}
"""
    # logger.info(f"SQL generation prompt: {prompt}")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"用户的问题是: {query}"},
    ]
    param = {
        "model": model_config["name"],
        "messages": messages,
        "key": model_config["key"],
        "url": model_config["url"],
        "api_version": model_config.get("api_version"),
    }

    try:
        response = call_openai_sdk(**param)
        content = response.choices[0].message.content
        sql_query = extract_sql_from_qwen(content)
        status, res = db_env.database.fetch(sql_query)
        if not status:
            for idx in range(3):
                sql_query = sql_fix(
                    db_env.dialect, db_env.mschema_str, query, sql_query, res
                )
                status, res = db_env.database.fetch(sql_query)
                if status:
                    break

        sql_res = db_env.database.fetch_truncated(sql_query, max_rows=100)
        markdown_res = db_env.database.trunc_result_to_markdown(sql_res)
        logger.info(f"SQL query: {sql_query}\nSQL result: {sql_res}")
        return markdown_res.strip()

    except Exception as e:
        return str(e)


def sql_fix(
    dialect: str, mschema: str, query: str, sql_query: str, error_info: str
):
    system_prompt = """现在你是一个{dialect}数据分析专家，需要阅读一个客户的问题，参考的数据库schema，该问题对应的待检查SQL，以及执行该SQL时数据库返回的语法错误，请你仅针对其中的语法错误进行修复，输出修复后的SQL。
注意：
1、仅修复语法错误，不允许改变SQL的逻辑。
2、生成的SQL用```sql 和```包围起来。

【数据库schema】
{schema}
""".format(dialect=dialect, schema=mschema)
    user_prompt = """【问题】
{question}

【待检查SQL】
{sql}

【错误信息】
{sql_res}""".format(question=query, sql=sql_query, sql_res=error_info)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    param = {
        "model": model_config["name"],
        "messages": messages,
        "key": model_config["key"],
        "url": model_config["url"],
    }

    response = call_openai_sdk(**param)
    content = response.choices[0].message.content
    sql_query = extract_sql_from_qwen(content)

    return sql_query


def call_xiyan(query: str, project_id: str | None = None) -> str:
    """Fetch the data from database through a natural language query

    Args:
        query: The query in natual language
    """
    pid = project_id if (project_id and project_id in DB_REGISTRY) else DEFAULT_PROJECT_ID
    logger.info(f"Calling tool with arguments: {query}, project_id: {pid}")
    try:
        db_source = get_db_source(pid)
    except Exception as e:
        return "数据库连接失败" + str(e)
    logger.info("Calling xiyan")
    env = DataBaseEnv(db_source)
    res = sql_gen_and_execute(env, query)

    return str(res)


@mcp.tool()
def get_data(query: str) -> list[TextContent]:
    """Fetch the data from database through a natural language query

    Args:
        query: The query in natural language
    """
    res = call_xiyan(query)
    return [TextContent(type="text", text=res)]


@mcp.tool()
def get_data_by_project(query: str, project_id: str) -> list[TextContent]:
    """Fetch the data from a specified project's database through a natural language query.

    Args:
        query: The query in natural language
        project_id: The target project_id as configured in YAML
    """
    res = call_xiyan(query, project_id)
    return [TextContent(type="text", text=res)]


def main():
    parser = argparse.ArgumentParser(description="Run MCP server.")
    parser.add_argument(
        "transport",
        nargs="?",
        default="stdio",
        choices=["stdio", "streamable-http", "sse"],
        help="Transport type (stdio, streamable-http or sse)",
    )
    parser.add_argument(
        "host", default="localhost", help="host for the http transport"
    )

    parser.add_argument(
        "port", default="8000", help="port for the http transport"
    )
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.settings.port = args.port
        mcp.settings.host = args.host
        logger.info(f"MCP server running at {args.host}/{args.port}")
        mcp.run(transport="streamable-http")
    else:
        logger.info(f"MCP server running by {args.transport}")
        print(f"MCP server running by {args.transport}")
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    print(f"MCP server start")
    main()
