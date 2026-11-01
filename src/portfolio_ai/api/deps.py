from fastapi import Request

from portfolio_ai.api.errors import ApiError
from portfolio_ai.container import Services


def get_services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise ApiError(
            503, "service_unavailable", "The assistant is starting up. Please retry shortly."
        )
    return services
