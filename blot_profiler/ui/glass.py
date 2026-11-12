"""No Liquid Glass material in Blot Profiler: the regions are opaque (tone only)."""


def active() -> bool:
    return False


def attach(_window) -> None:
    return None
