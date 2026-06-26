"""
Evaluation Metrics Module
"""
import json
from typing import Dict, Any, List, Tuple


class MetricsCalculator:
    """Evaluation metrics calculator"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, item_id: str, task_type: str, gold_label: Dict[str, Any], 
                   prediction: Dict[str, Any], raw_response: str, is_correct: bool):
        """
        Add single evaluation result
        
        Args:
            item_id: Data item ID
            task_type: Task type
            gold_label: Gold standard answer
            prediction: Model prediction
            raw_response: Raw response
            is_correct: Whether correct
        """
        self.results.append({
            "item_id": item_id,
            "task_type": task_type,
            "gold_label": gold_label,
            "prediction": prediction,
            "raw_response": raw_response,
            "is_correct": is_correct
        })
    
    def calculate_accuracy(self, task_type: str = None) -> float:
        """
        Calculate accuracy
        
        Args:
            task_type: Optional, specify task type. If None, calculate overall accuracy
            
        Returns:
            Accuracy (between 0-1)
        """
        if task_type:
            filtered_results = [r for r in self.results if r['task_type'] == task_type]
        else:
            filtered_results = self.results
        
        if not filtered_results:
            return 0.0
        
        correct_count = sum(1 for r in filtered_results if r['is_correct'])
        return correct_count / len(filtered_results)
    
    def calculate_task_wise_accuracy(self) -> Dict[str, float]:
        """
        Calculate accuracy for each task type
        
        Returns:
            Mapping from task type to accuracy
        """
        task_types = set(r['task_type'] for r in self.results)
        return {
            task_type: self.calculate_accuracy(task_type)
            for task_type in task_types
        }
    
    def get_error_analysis(self) -> List[Dict[str, Any]]:
        """
        Get error case analysis
        
        Returns:
            List of error cases
        """
        return [r for r in self.results if not r['is_correct']]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get evaluation summary
        
        Returns:
            Summary dictionary with various metrics
        """
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        wrong = total - correct
        
        task_wise_acc = self.calculate_task_wise_accuracy()
        
        return {
            "total_items": total,
            "correct": correct,
            "wrong": wrong,
            "overall_accuracy": f"{self.calculate_accuracy() * 100:.2f}%",
            "task_wise_accuracy": {k: f"{v * 100:.2f}%" for k, v in task_wise_acc.items()},
            "error_count": wrong
        }
    
    def print_summary(self):
        """Print evaluation summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 Evaluation Summary")
        print("="*60)
        print(f"Total Items: {summary['total_items']}")
        print(f"Correct: {summary['correct']}")
        print(f"Wrong: {summary['wrong']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']}")
        print("\nTask-wise Accuracy:")
        for task_type, acc in summary['task_wise_accuracy'].items():
            print(f"  - {task_type}: {acc}")
        print("="*60)
    
    def print_error_analysis(self):
        """Print error case analysis"""
        errors = self.get_error_analysis()
        
        if not errors:
            print("\n✓ No errors!")
            return
        
        print("\n" + "="*60)
        print(f"❌ Error Analysis ({len(errors)} errors)")
        print("="*60)
        
        for i, error in enumerate(errors, 1):
            print(f"\nError #{i} (ID: {error['item_id']})")
            print(f"Task Type: {error['task_type']}")
            print(f"Gold Label: {error['gold_label']}")
            print(f"Prediction: {error['prediction']}")
            print(f"Raw Response: {error['raw_response'][:100]}...")
            print("-" * 40)


def compare_labels(gold_label: Dict[str, Any], prediction: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Compare gold label and prediction
    
    Args:
        gold_label: Gold standard answer
        prediction: Model prediction
        
    Returns:
        (is_match, match_detail)
    """
    # Handle None or empty prediction
    if prediction is None or not prediction:
        return False, "Empty prediction"
    
    # Handle JSON parsing errors
    if "ERROR" in str(prediction):
        return False, "Model returned error"
    
    # Different comparison logic for different task types
    
    # 1. Simple label matching
    if "label" in gold_label and "label" in prediction:
        is_match = gold_label["label"].lower() == prediction["label"].lower()
        return is_match, f"Label match: {gold_label['label']} vs {prediction.get('label', 'N/A')}"
    
    # 2. Entity type matching
    if "entity" in gold_label and "type" in gold_label:
        # Can check only type, or both entity and type
        if "type" in prediction:
            is_match = gold_label["type"].lower() == prediction["type"].lower()
            detail = f"Type match: {gold_label['type']} vs {prediction.get('type', 'N/A')}"
        else:
            is_match = False
            detail = "Prediction missing type field"
        return is_match, detail
    
    # 3. Domain classification matching
    if "domain" in gold_label and "domain" in prediction:
        is_match = gold_label["domain"].lower() == prediction["domain"].lower()
        return is_match, f"Domain match: {gold_label['domain']} vs {prediction.get('domain', 'N/A')}"
    
    # 4. Numeric result matching (e.g., robustness check)
    if "result" in gold_label and "result" in prediction:
        try:
            # Handle both int and float comparisons
            gold_val = float(gold_label["result"])
            pred_val = float(prediction["result"])
            is_match = abs(gold_val - pred_val) < 0.0001  # Allow small floating point errors
            return is_match, f"Result match: {gold_label['result']} vs {prediction.get('result', 'N/A')}"
        except (ValueError, TypeError):
            return False, "Result type mismatch"
    
    # 5. Default: exact matching
    is_match = gold_label == prediction
    return is_match, f"Exact match: {is_match}"


def parse_model_response(response: str) -> Dict[str, Any]:
    """
    Parse model response to JSON format
    
    Args:
        response: Model raw response
        
    Returns:
        Parsed dictionary
    """
    if not response:
        return {}
    
    # Handle error responses
    if response.startswith("ERROR_QUOTA:"):
        return {"ERROR": "QUOTA_EXCEEDED", "message": response.replace("ERROR_QUOTA: ", "")}
    elif response.startswith("ERROR_RATE_LIMIT:"):
        return {"ERROR": "RATE_LIMIT", "message": response.replace("ERROR_RATE_LIMIT: ", "")}
    elif response.startswith("ERROR:"):
        return {"ERROR": response}
    
    # Try direct JSON parsing
    try:
        # Remove possible markdown code block markers
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON object
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # If unable to parse, return raw text
    return {"raw": response, "ERROR": "JSON parsing failed"}
