"""
Data Loader Module - Load and parse evaluation dataset
"""
import json
from typing import List, Dict, Any


class DataLoader:
    """Evaluation data loader"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data file
        """
        self.data_path = data_path
        self.data = []
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load dataset
        
        Returns:
            List of data items
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✓ Successfully loaded {len(self.data)} evaluation items")
        return self.data
    
    def get_item(self, item_id: str) -> Dict[str, Any]:
        """
        Get data item by ID
        
        Args:
            item_id: Data item ID
            
        Returns:
            Data item dictionary
        """
        for item in self.data:
            if item['id'] == item_id:
                return item
        raise ValueError(f"Item with ID {item_id} not found")
    
    def get_by_task_type(self, task_type: str) -> List[Dict[str, Any]]:
        """
        Filter data by task type
        
        Args:
            task_type: Task type
            
        Returns:
            List of matching data items
        """
        return [item for item in self.data if item['task_type'] == task_type]
    
    def get_task_types(self) -> List[str]:
        """
        Get all task types
        
        Returns:
            List of task types
        """
        return list(set(item['task_type'] for item in self.data))
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get dataset statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {}
        for item in self.data:
            task_type = item['task_type']
            stats[task_type] = stats.get(task_type, 0) + 1
        return stats
