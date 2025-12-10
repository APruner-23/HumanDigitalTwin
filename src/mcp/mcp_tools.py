"""
Langchain Tools for interacting with the MCP server.
These tools allow the LLM to autonomously call the MCP.
"""

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests
import json


# Base URL of the MCP server (can be configured)
_MCP_BASE_URL = "http://localhost:8000"


def set_mcp_base_url(url: str):
    """Sets the base URL of the MCP server."""
    global _MCP_BASE_URL
    _MCP_BASE_URL = url.rstrip('/')


def _make_request(endpoint: str, method: str = "GET", **kwargs):
    """
    Makes an HTTP request to the MCP server.

    Args:
        endpoint: Endpoint to call (without leading slash)
        method: HTTP Method
        **kwargs: Additional parameters for requests

    Returns:
        JSON response from server
    """
    url = f"{_MCP_BASE_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        else:
            raise ValueError(f"HTTP method not supported: {method}")

        response.raise_for_status()
        result = response.json()

        # Log MCP request
        try:
            from ..utils import get_logger
            logger = get_logger()
            logger.log_mcp_request(method, endpoint, kwargs.get('params'), result)
        except:
            pass  # Ignore logging errors

        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status": "failed"}


# Define Pydantic schemas for tool parameters

class IoTRecentDataInput(BaseModel):
    """Schema for get_iot_recent_data."""
    device_id: str = Field(description="IoT device ID (e.g., 'fitbit_1234', 'garmin_5678')")
    limit: int = Field(default=10, description="Maximum number of records to retrieve")


class IoTStatisticsInput(BaseModel):
    """Schema for get_iot_statistics."""
    device_id: str = Field(description="IoT device ID")


@tool(args_schema=IoTRecentDataInput)
def get_iot_recent_data(device_id: str, limit: int = 10) -> str:
    """
    Retrieves the most recent IoT data for a specific device.
    Useful for seeing the latest sensor readings.
    """
    result = _make_request(
        "api/iot/recent",
        params={"device_id": device_id, "limit": limit}
    )
    return json.dumps(result, indent=2)


@tool(args_schema=IoTStatisticsInput)
def get_iot_statistics(device_id: str) -> str:
    """
    Calculates aggregated statistics on a device's IoT data.
    Includes avg, min, max for all numeric parameters (e.g., heart rate, steps, temperature).
    """
    result = _make_request("api/iot/stats", params={"device_id": device_id})
    return json.dumps(result, indent=2)


@tool
def get_user_context() -> str:
    """
    Get a lightweight summary of user context from all sources.
    Returns metadata and available fields, but NOT actual data values.
    Use this to discover what devices and fields are available, then use specific query tools for data.
    """
    result = _make_request("api/context")
    return json.dumps(result, indent=2)


@tool
def list_devices() -> str:
    """
    Lists all IoT devices registered in the system.
    Useful for discovering which devices are available before querying them.
    """
    result = _make_request("api/devices")
    return json.dumps(result, indent=2)


class DataSchemaInput(BaseModel):
    """Schema for get_data_schema."""
    device_id: str = Field(description="IoT device ID")


@tool(args_schema=DataSchemaInput)
def get_data_schema(device_id: str) -> str:
    """
    Returns the data schema for a specific device, showing available fields and their types.
    Use this FIRST to discover what data fields are available before querying specific data.
    This is lightweight and doesn't return actual data, only the structure.
    """
    result = _make_request(f"api/schema/{device_id}")
    return json.dumps(result, indent=2)


class QueryFieldInput(BaseModel):
    """Schema for query_iot_field."""
    device_id: str = Field(description="IoT device ID")
    field_name: str = Field(description="Name of the field to retrieve (e.g., 'heart_rate', 'steps', 'temperature')")
    limit: int = Field(default=10, description="Maximum number of values to retrieve")


@tool(args_schema=QueryFieldInput)
def query_iot_field(device_id: str, field_name: str, limit: int = 10) -> str:
    """
    Query a SPECIFIC field from IoT data, returning only that field's values.
    Much more efficient than getting all data when you only need one field.
    Example: To get only heart rate data, use field_name='heart_rate'
    """
    result = _make_request(
        "api/iot/field",
        params={"device_id": device_id, "field_name": field_name, "limit": limit}
    )
    return json.dumps(result, indent=2)


class LatestValueInput(BaseModel):
    """Schema for get_latest_value."""
    device_id: str = Field(description="IoT device ID")
    field_name: str = Field(description="Name of the field (e.g., 'heart_rate', 'steps')")


@tool(args_schema=LatestValueInput)
def get_latest_value(device_id: str, field_name: str) -> str:
    """
    Get the most recent value for a specific field.
    Most efficient way to check current status of a single metric.
    Example: Get current heart rate with device_id='fitbit_123', field_name='heart_rate'
    """
    result = _make_request(
        "api/iot/latest",
        params={"device_id": device_id, "field_name": field_name}
    )
    return json.dumps(result, indent=2)


class AggregateFieldInput(BaseModel):
    """Schema for aggregate_iot_field."""
    device_id: str = Field(description="IoT device ID")
    field_name: str = Field(description="Name of the field to aggregate")
    operation: str = Field(description="Operation: avg, min, max, sum, count")


@tool(args_schema=AggregateFieldInput)
def aggregate_iot_field(device_id: str, field_name: str, operation: str) -> str:
    """
    Compute aggregations (avg, min, max, sum, count) on a field WITHOUT returning raw data.
    Most efficient for statistical queries. The computation happens on the server.
    Example: Get average heart rate with operation='avg', field_name='heart_rate'
    Available operations: avg, min, max, sum, count
    """
    result = _make_request(
        "api/iot/aggregate",
        params={"device_id": device_id, "field_name": field_name, "operation": operation}
    )
    return json.dumps(result, indent=2)


def get_mcp_tools(mcp_base_url: str = "http://localhost:8000") -> list:
    """
    Returns the list of all MCP tools for Langchain.

    Args:
        mcp_base_url: Base URL of the MCP server

    Returns:
        List of Langchain tools
    """
    # Set global URL
    set_mcp_base_url(mcp_base_url)

    # Return tools (new optimized tools + legacy tools)
    return [
        # Prioritized efficient tools
        list_devices,           # Start here to discover devices
        get_data_schema,        # Then discover available fields
        get_latest_value,       # Most efficient for single values
        aggregate_iot_field,    # Most efficient for statistics
        query_iot_field,        # Efficient for specific fields

        # Legacy tools (less efficient, but still available)
        get_iot_recent_data,    # Returns full records
        get_iot_statistics,     # Returns all stats
        get_user_context,       # Returns all context
    ]
