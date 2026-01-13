from typing import Any, Literal               # For flexible types and fixed value literals
from uuid import uuid4                       # To generate unique request IDs
from pydantic import BaseModel, Field        # For creating robust, validated data models


# -----------------------------------------------------------------------------
# JSONRPCMessage (base class)
# -----------------------------------------------------------------------------
# All messages in JSON-RPC share these fields.
# This is the common parent class for both requests and responses.
class JSONRPCMessage(BaseModel):
    # Always specify the protocol version. "2.0" is the only valid value.
    jsonrpc: Literal["2.0"] = "2.0"

    # The message ID is used to match requests with responses.
    # If not provided, we generate a unique ID using uuid4.
    id: int | str | None = Field(default_factory=lambda: uuid4().hex)


# -----------------------------------------------------------------------------
# JSONRPCRequest
# -----------------------------------------------------------------------------
# A JSON-RPC request to call a method on another agent.
# This is what you send to perform an action.
class JSONRPCRequest(JSONRPCMessage):
    # The name of the method you want to call (e.g., "tasks/send")
    method: str

    # Optional input parameters for the method (can be omitted if not needed)
    params: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# JSONRPCError
# -----------------------------------------------------------------------------
# This represents a standard error object in a JSON-RPC response.
# It's used when the method call fails due to an error.
class JSONRPCError(BaseModel):
    # Numeric error code. Use standard codes if possible (e.g., -32603 for internal error).
    code: int

    # Human-readable message describing the error
    message: str

    # Optional additional information, like a stack trace or internal debug info
    data: Any | None = None


# -----------------------------------------------------------------------------
# JSONRPCResponse
# -----------------------------------------------------------------------------
# A JSON-RPC response that either contains a result or an error.
# Only one of `result` or `error` should be non-null.
class JSONRPCResponse(JSONRPCMessage):
    # The successful result from a method call
    result: Any | None = None

    # The error object if the method failed
    error: JSONRPCError | None = None


# -----------------------------------------------------------------------------
# InternalError (subclass of JSONRPCError)
# -----------------------------------------------------------------------------
# A predefined error for when the agent encounters an unexpected exception.
# This follows the JSON-RPC standard error code for internal errors (-32603).
class InternalError(JSONRPCError):
    # Fixed error code for internal errors
    code: int = -32603

    # Default error message describing the type of error
    message: str = "Internal error"

    # Optional debug details (e.g., traceback or context info)
    data: Any | None = None