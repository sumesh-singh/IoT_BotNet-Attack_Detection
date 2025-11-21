"""
GraphQL API Entry Point for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Exposes the GraphQL API using Starlette/FastAPI.
"""

from starlette.graphql import GraphQLApp
from .graphql_schema import schema

# Create GraphQL app
graphql_app = GraphQLApp(schema=schema)
