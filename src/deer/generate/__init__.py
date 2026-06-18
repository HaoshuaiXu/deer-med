"""Generation stage: build DEER prompts and parse named_entities JSON."""
from .generator import (
    build_generation_prompt,
    parse_named_entities,
    generate_entities,
)

__all__ = ["build_generation_prompt", "parse_named_entities", "generate_entities"]
