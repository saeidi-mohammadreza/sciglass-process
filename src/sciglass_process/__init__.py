"""sciglass-process: decode melting and annealing history hidden in SciGlass source records."""
from .parser import MeltStage, Process, parse_process, parse_value
from .dataset import code_dictionary, coverage, process_by_glass, process_by_record, read_table

__all__ = [
    "MeltStage", "Process", "parse_process", "parse_value",
    "code_dictionary", "coverage", "process_by_glass", "process_by_record", "read_table",
]
__version__ = "0.1.0"
