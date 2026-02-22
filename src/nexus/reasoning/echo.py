
"""
Echo cognitive module - demonstrates basic reasoning interface.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class EchoModule:
    """
    Simple echo module that returns the input with metadata.
    Serves as a basic example of a cognitive module.
    """
    
    def __init__(self):
        self.name = "echo"
        self.version = "1.0.0"
        
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input and return echo response.
        
        Args:
            input_data: The input to echo back
            
        Returns:
            Dictionary containing the echoed input and metadata
        """
        logger.info(f"Echo module processing: {input_data}")
        
        return {
            'echoed_input': input_data,
            'module': self.name,
            'version': self.version,
            'type': type(input_data).__name__,
            'length': len(str(input_data)) if hasattr(input_data, '__len__') or isinstance(input_data, str) else None
        }
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Return module capabilities."""
        return {
            'name': self.name,
            'version': self.version,
            'description': 'Echoes input with metadata',
            'input_types': ['any'],
            'output_type': 'dict'
        }
