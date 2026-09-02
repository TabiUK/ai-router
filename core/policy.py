from enum import Enum


class RoutingPolicy(str, Enum):
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    LOW_POWER = "low_power"