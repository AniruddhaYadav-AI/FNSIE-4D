"""
Inference Engine Core Module
"""
import json
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from data_loader import DataLoader
from model_adapters import BaseModelAdapter
from metrics import MetricsCalculator, parse_model_response, compare_labels


class InferenceEngine:
    """LLM Inference Evaluation Engine"""
    
    def __init__(self, model_adapter: BaseModelAdapter, data_loader: DataLoader):
        """
        Initialize inference engine
        
        Args:
            model_adapter: Model adapter
            data_loader: Data loader
        """
        self.model_adapter = model_adapter
        self.data_loader = data_loader
        self.metrics = MetricsCalculator()
    
    def build_prompt(self, item: Dict[str, Any]) -> str:
        """
        Build prompt based on data item
        
        Args:
            item: Data item
            
        Returns:
            Prompt string
        """
        task_type = item['task_type']
        input_text = item['input']
        classes = item.get('classes', [])
        
        # Build different prompts for different task types
        if task_type == "topic_classification":
            prompt = f"""Please classify the following text into one of the given topics.

Text: {input_text}

Available categories: {', '.join(classes)}

Return your answer in JSON format:
{{"label": "your_chosen_category"}}

Only return JSON, no other explanation."""
        
        elif task_type == "sentiment_analysis":
            prompt = f"""Please analyze the sentiment of the following text.

Text: {input_text}

Available sentiments: {', '.join(classes)}

Return your answer in JSON format:
{{"label": "your_chosen_sentiment"}}

Only return JSON, no other explanation."""
        
        elif task_type == "entity_type":
            prompt = f"""Please identify the main entity and its type in the following text.

Text: {input_text}

Available entity types: {', '.join(classes)}

Return your answer in JSON format:
{{"entity": "entity_name", "type": "entity_type"}}

Only return JSON, no other explanation."""
        
        elif task_type == "domain_classification":
            prompt = f"""Please determine which professional domain the following text belongs to.

Text: {input_text}

Available domains: {', '.join(classes)}

Return your answer in JSON format:
{{"domain": "your_chosen_domain"}}

Only return JSON, no other explanation."""
        
        elif task_type == "yes_no":
            prompt = f"""Please answer the following question.

Question: {input_text}

Available answers: {', '.join(classes)}

Return your answer in JSON format:
{{"label": "your_answer"}}

Only return JSON, no other explanation."""
        
        elif task_type == "intent_classification":
            prompt = f"""Please determine the intent of the following user input.

User input: {input_text}

Available intents: {', '.join(classes)}

Return your answer in JSON format:
{{"label": "your_determined_intent"}}

Only return JSON, no other explanation."""
        
        elif task_type == "polarity_detection":
            prompt = f"""Please determine the type of the following sentence.

Sentence: {input_text}

Available types: {', '.join(classes)}

Return your answer in JSON format:
{{"label": "your_determined_type"}}

Only return JSON, no other explanation."""
        
        elif task_type == "robustness_check":
            prompt = f"""Please answer the following question (note: there may be spelling errors in the question).

Question: {input_text}

{classes if isinstance(classes, str) else ', '.join(classes)}

Only return JSON, no other explanation."""
        
        else:
            # Default prompt
            prompt = f"""Please process the following task.

Input: {input_text}

Requirement: {classes if isinstance(classes, str) else ', '.join(classes)}

Please return your answer in JSON format."""
        
        return prompt
    
    def run_single_item(self, item: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """
        Run inference on single data item
        
        Args:
            item: Data item
            verbose: Whether to print detailed information
            
        Returns:
            Inference result
        """
        # Build prompt
        prompt = self.build_prompt(item)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Item #{item['id']} - {item['task_type']}")
            print(f"{'='*60}")
            print(f"Input: {item['input']}")
        
        # Call model
        start_time = time.time()
        raw_response = self.model_adapter.generate(prompt, temperature=0.0, max_tokens=500)
        inference_time = time.time() - start_time
        
        if verbose:
            print(f"Model Response: {raw_response}")
        
        # Parse response
        prediction = parse_model_response(raw_response)
        
        # Compare results
        is_correct, match_detail = compare_labels(item['gold_label'], prediction)
        
        if verbose:
            print(f"Gold Label: {item['gold_label']}")
            print(f"Prediction: {prediction}")
            print(f"Result: {'✓ Correct' if is_correct else '✗ Wrong'}")
            print(f"Inference Time: {inference_time:.2f}s")
        
        # Record result
        self.metrics.add_result(
            item_id=item['id'],
            task_type=item['task_type'],
            gold_label=item['gold_label'],
            prediction=prediction,
            raw_response=raw_response,
            is_correct=is_correct
        )
        
        return {
            "item_id": item['id'],
            "task_type": item['task_type'],
            "gold_label": item['gold_label'],
            "prediction": prediction,
            "raw_response": raw_response,
            "is_correct": is_correct,
            "inference_time": inference_time
        }
    
    def run_evaluation(self, verbose: bool = False, sleep_time: float = 0.5) -> Dict[str, Any]:
        """
        Run complete evaluation
        
        Args:
            verbose: Whether to print detailed information
            sleep_time: Sleep time between requests (seconds)
            
        Returns:
            Evaluation results
        """
        data = self.data_loader.data
        
        print(f"\n🚀 Starting evaluation on {len(data)} items")
        print(f"Model: {self.model_adapter.model_name}")
        print("-" * 60)
        
        results = []
        
        # Track quota errors
        quota_error_count = 0
        quota_error_threshold = 5  # Stop after 5 consecutive quota errors
        
        # Use progress bar
        for item in tqdm(data, desc="Evaluation Progress", ncols=80):
            result = self.run_single_item(item, verbose=verbose)
            results.append(result)
            
            # Check for quota errors
            if "ERROR_QUOTA" in result.get('raw_response', ''):
                quota_error_count += 1
                if quota_error_count >= quota_error_threshold:
                    print(f"\n⚠️  Quota error detected {quota_error_count} times in a row.")
                    print("   Stopping evaluation to save API calls.")
                    print("   Please check your API quota and billing:")
                    if "openai" in self.model_adapter.model_name.lower():
                        print("   https://platform.openai.com/account/billing")
                    break
            else:
                quota_error_count = 0  # Reset counter on success
            
            # Avoid API rate limits
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Print summary
        self.metrics.print_summary()
        
        # Print error analysis
        if not verbose:
            self.metrics.print_error_analysis()
        
        return {
            "results": results,
            "summary": self.metrics.get_summary(),
            "model_stats": self.model_adapter.get_stats()
        }
    
    def run_subset(self, item_ids: List[str] = None, task_type: str = None, 
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Run subset evaluation
        
        Args:
            item_ids: Specified data item ID list
            task_type: Specified task type
            verbose: Whether to print detailed information
            
        Returns:
            Evaluation results
        """
        data = self.data_loader.data
        
        # Filter data
        if item_ids:
            data = [item for item in data if item['id'] in item_ids]
        elif task_type:
            data = [item for item in data if item['task_type'] == task_type]
        
        print(f"\n🚀 Starting subset evaluation on {len(data)} items")
        print(f"Model: {self.model_adapter.model_name}")
        
        results = []
        for item in data:
            result = self.run_single_item(item, verbose=verbose)
            results.append(result)
            time.sleep(0.5)
        
        self.metrics.print_summary()
        
        return {
            "results": results,
            "summary": self.metrics.get_summary()
        }
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """
        Save evaluation results to file
        
        Args:
            output_path: Output file path
        """
        results_data = {
            "model": self.model_adapter.model_name,
            "summary": self.metrics.get_summary(),
            "model_stats": self.model_adapter.get_stats(),
            "detailed_results": self.metrics.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Evaluation results saved to: {output_path}")
