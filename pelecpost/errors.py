"""Public error types and CLI exit-code contracts."""


class PelecPostError(Exception):
    """Base class for expected, user-facing failures."""


class ProjectConfigurationError(PelecPostError):
    """A project file is absent, malformed, or inconsistent."""


class PreflightBlockedError(PelecPostError):
    """Scientific or resource preflight contains one or more blockers."""


class UnsupportedCapabilityError(PelecPostError):
    """The input is valid but the requested dimensional capability is absent."""


CONFIGURATION_EXIT_CODE = 2
RUNTIME_FAILURE_EXIT_CODE = 1

