"""Compatibility module for the Cochise-compatible cyberqa CLI."""

from .cli.cochise import async_main, main, parse_args

__all__ = ["async_main", "main", "parse_args"]


if __name__ == "__main__":
    main()
