from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uvicorn
from collections import defaultdict


class MCPServer:
    """MCP Server to expose APIs that inject information into the model."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        Initialize the MCP server.

        Args:
            host: Host to run the server on
            port: Port to run the server on
        """
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="MCP Server - Human Digital Twin",
            description="API per l'interazione autonoma con il modello LLM",
            version="1.0.0"
        )

        # In-memory storage for IoT data and context
        self.iot_data_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.external_data_store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Configures the API routes."""

        @self.app.get("/")
        async def root():
            """Test endpoint."""
            return {"status": "ok", "message": "MCP Server is running"}

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        @self.app.post("/api/iot/data")
        async def receive_iot_data(data: IoTDataModel):
            """
            Receives IoT data from devices.

            Args:
                data: IoT data in JSON format

            Returns:
                Reception confirmation with ID
            """
            # Save data to in-memory storage
            device_id = data.device_id
            data_dict = {
                "device_type": data.device_type,
                "device_id": device_id,
                "timestamp": data.timestamp,
                "data": data.data,
                "metadata": data.metadata
            }
            self.iot_data_store[device_id].append(data_dict)

            return {
                "status": "received",
                "device_id": device_id,
                "data_type": data.device_type,
                "timestamp": data.timestamp,
                "stored_count": len(self.iot_data_store[device_id])
            }

        @self.app.get("/api/iot/recent")
        async def get_recent_iot_data(
            device_id: str = Query(..., description="Device ID"),
            limit: int = Query(10, description="Max number of records to return")
        ):
            """
            Retrieves the most recent IoT data for a device.

            Args:
                device_id: Device ID
                limit: Max number of records

            Returns:
                List of recent IoT data
            """
            if device_id not in self.iot_data_store:
                return {
                    "device_id": device_id,
                    "data": [],
                    "message": "Nessun dato disponibile per questo dispositivo"
                }

            # Retrieve last N records
            recent_data = self.iot_data_store[device_id][-limit:]

            return {
                "device_id": device_id,
                "count": len(recent_data),
                "data": recent_data
            }

        @self.app.get("/api/iot/stats")
        async def get_iot_stats(
            device_id: str = Query(..., description="Device ID")
        ):
            """
            Calculates aggregate statistics on IoT data for a device.

            Args:
                device_id: Device ID

            Returns:
                Aggregate statistics
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]

            if not data_list:
                return {"device_id": device_id, "stats": {}, "message": "No data"}

            # Calculate base statistics
            stats = {
                "total_records": len(data_list),
                "first_timestamp": data_list[0]["timestamp"],
                "last_timestamp": data_list[-1]["timestamp"],
                "device_type": data_list[0]["device_type"]
            }

            # Calculate averages for numeric fields
            numeric_fields = {}
            for record in data_list:
                for key, value in record["data"].items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_fields:
                            numeric_fields[key] = []
                        numeric_fields[key].append(value)

            averages = {}
            for field, values in numeric_fields.items():
                averages[field] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

            stats["metrics"] = averages

            return {
                "device_id": device_id,
                "stats": stats
            }

        @self.app.post("/api/external/gmail")
        async def receive_gmail_data(data: ExternalDataModel):
            """
            Receives data from Gmail or other external services.

            Args:
                data: External data in JSON format

            Returns:
                Reception confirmation
            """
            # Save external data
            source = data.source
            data_dict = {
                "source": source,
                "data_id": data.data_id,
                "timestamp": data.timestamp,
                "content": data.content,
                "metadata": data.metadata
            }
            self.external_data_store[source].append(data_dict)

            return {
                "status": "received",
                "source": source,
                "data_id": data.data_id,
                "stored_count": len(self.external_data_store[source])
            }

        @self.app.get("/api/context")
        async def get_context():
            """
            Returns a lightweight summary of the current context.
            Does NOT return actual data, only metadata and counts.
            Use specific query tools to retrieve actual data.

            Returns:
                Context summary from various sources
            """
            # Aggregate metadata only (no actual data)
            context = {
                "iot_devices": list(self.iot_data_store.keys()),
                "external_sources": list(self.external_data_store.keys()),
                "total_iot_records": sum(len(v) for v in self.iot_data_store.values()),
                "total_external_records": sum(len(v) for v in self.external_data_store.values())
            }

            # Add lightweight device summary (metadata only, no data)
            iot_summary = {}
            for device_id, data_list in self.iot_data_store.items():
                if data_list:
                    latest = data_list[-1]
                    iot_summary[device_id] = {
                        "device_type": latest["device_type"],
                        "last_update": latest["timestamp"],
                        "available_fields": list(latest["data"].keys()),  # Field names only
                        "record_count": len(data_list)
                    }

            context["iot_summary"] = iot_summary

            return {
                "context": context,
                "sources": ["iot", "external"],
                "note": "This is a summary only. Use get_data_schema, query_iot_field, or aggregate_iot_field for actual data."
            }

        @self.app.get("/api/devices")
        async def list_devices():
            """
            Lists all registered IoT devices.

            Returns:
                List of device_ids with basic info
            """
            devices = []
            for device_id, data_list in self.iot_data_store.items():
                if data_list:
                    devices.append({
                        "device_id": device_id,
                        "device_type": data_list[0]["device_type"],
                        "record_count": len(data_list),
                        "last_update": data_list[-1]["timestamp"]
                    })

            return {
                "count": len(devices),
                "devices": devices
            }

        @self.app.get("/api/schema/{device_id}")
        async def get_data_schema(device_id: str):
            """
            Returns the data schema for a specific device.
            Shows available fields and their types without returning actual data.

            Args:
                device_id: ID of the device

            Returns:
                Schema information with field names and types
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                return {"device_id": device_id, "schema": {}, "message": "No data available"}

            # Extract schema from the latest record
            latest_record = data_list[-1]
            schema = {
                "device_type": latest_record["device_type"],
                "fields": {}
            }

            # Analyze fields from multiple records to get accurate types
            for record in data_list[-10:]:  # Check last 10 records
                for field_name, value in record["data"].items():
                    field_type = type(value).__name__
                    if field_name not in schema["fields"]:
                        schema["fields"][field_name] = {
                            "type": field_type,
                            "sample_value": value
                        }

            return {
                "device_id": device_id,
                "schema": schema,
                "total_records": len(data_list)
            }

        @self.app.get("/api/iot/field")
        async def query_iot_field(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name to query"),
            limit: int = Query(10, description="Maximum number of records")
        ):
            """
            Query a specific field from IoT data, returning only that field.

            Args:
                device_id: ID of the device
                field_name: Name of the field to retrieve
                limit: Maximum number of records

            Returns:
                List of values for the specified field
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                return {"device_id": device_id, "field": field_name, "values": [], "message": "No data"}

            # Extract only the requested field
            values = []
            for record in data_list[-limit:]:
                if field_name in record["data"]:
                    values.append({
                        "timestamp": record["timestamp"],
                        "value": record["data"][field_name]
                    })

            return {
                "device_id": device_id,
                "field": field_name,
                "count": len(values),
                "values": values
            }

        @self.app.get("/api/iot/latest")
        async def get_latest_value(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name")
        ):
            """
            Get the latest value for a specific field.

            Args:
                device_id: ID of the device
                field_name: Name of the field

            Returns:
                Latest value with timestamp
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                raise HTTPException(404, f"No data available for device {device_id}")

            # Get latest record
            latest_record = data_list[-1]
            if field_name not in latest_record["data"]:
                raise HTTPException(404, f"Field {field_name} not found in device data")

            return {
                "device_id": device_id,
                "field": field_name,
                "value": latest_record["data"][field_name],
                "timestamp": latest_record["timestamp"]
            }

        @self.app.get("/api/iot/aggregate")
        async def aggregate_iot_field(
            device_id: str = Query(..., description="Device ID"),
            field_name: str = Query(..., description="Field name to aggregate"),
            operation: str = Query(..., description="Operation: avg, min, max, sum, count")
        ):
            """
            Compute server-side aggregation on a field without returning raw data.

            Args:
                device_id: ID of the device
                field_name: Name of the field to aggregate
                operation: Aggregation operation (avg, min, max, sum, count)

            Returns:
                Aggregated value
            """
            if device_id not in self.iot_data_store:
                raise HTTPException(404, f"Device {device_id} not found")

            data_list = self.iot_data_store[device_id]
            if not data_list:
                raise HTTPException(404, f"No data available for device {device_id}")

            # Extract field values
            values = []
            for record in data_list:
                if field_name in record["data"]:
                    value = record["data"][field_name]
                    if isinstance(value, (int, float)):
                        values.append(value)

            if not values:
                raise HTTPException(404, f"No numeric values found for field {field_name}")

            # Compute aggregation
            result = None
            if operation == "avg":
                result = sum(values) / len(values)
            elif operation == "min":
                result = min(values)
            elif operation == "max":
                result = max(values)
            elif operation == "sum":
                result = sum(values)
            elif operation == "count":
                result = len(values)
            else:
                raise HTTPException(400, f"Invalid operation: {operation}. Use: avg, min, max, sum, count")

            return {
                "device_id": device_id,
                "field": field_name,
                "operation": operation,
                "result": result,
                "sample_size": len(values)
            }

    def run(self, debug: bool = False) -> None:
        """
        Starts the MCP server.

        Args:
            debug: If True, enables debug mode
        """
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if debug else "info"
        )

    def get_app(self) -> FastAPI:
        """
        Returns the FastAPI app for testing or deployment.

        Returns:
            The FastAPI instance
        """
        return self.app


# Pydantic models for data validation

class IoTDataModel(BaseModel):
    """Model for IoT data."""
    device_type: str
    device_id: str
    timestamp: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ExternalDataModel(BaseModel):
    """Model for external service data."""
    source: str  # gmail, calendar, etc.
    data_id: str
    timestamp: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
