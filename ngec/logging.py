
import logging

def setup_logging(level: int=logging.INFO, format_string: str | None=None, 
                  quiet_third_party: bool=True) -> None:
    """Setup logging for NGEC.
    
    This is a convenvience function to configure logging for the NGEC package.
    A primary reason is that some of the third-party packages used by NGEC are
    quite verbose at their default logging levels.

    Example:
    
    ```python
    from ngec.logging import setup_logging
    setup_logging(level=logging.DEBUG)
    ```
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if quiet_third_party:
        quiet_third_party_loggers()
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler()]
    )


def quiet_third_party_loggers() -> None:
    """Set third-party loggers to higher levels to reduce noise."""
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.ERROR)