"""
Dataset Creator for Office Domain Intent Recognition

This module generates synthetic training data for office domain intent classification.
It creates diverse sample queries for each intent to train the RoBERTa model effectively.

The dataset includes 7 intent categories:
1. salary_inquiry - Questions about salary, pay, compensation
2. meeting_room_booking - Requests to book meeting rooms or conference facilities
3. leave_request - Requests for time off, vacation, sick leave
4. directory_search - Searching for employee contact information
5. company_info - General company information queries
6. employee_info - Specific employee information requests
7. employee_search - Finding employees by company abbreviation and name
"""

import json
import pandas as pd
import random
from typing import List, Dict, Tuple
from pathlib import Path


class OfficeIntentDatasetCreator:
    """
    Creates a synthetic dataset for office domain intent recognition.
    
    This class generates diverse training examples for each intent category,
    ensuring the model can learn to recognize various ways users might express
    their intentions in an office environment.
    """
    
    def __init__(self, samples_per_intent: int = 100):
        """
        Initialize the dataset creator.
        
        Args:
            samples_per_intent: Number of training samples to generate per intent
        """
        self.samples_per_intent = samples_per_intent
        self.intent_labels = [
            "salary_inquiry",
            "meeting_room_booking", 
            "leave_request",
            "directory_search",
            "company_info",
            "employee_info",
            "employee_search"
        ]
        
        # Define templates and vocabulary for each intent
        self._define_intent_templates()
    
    def _define_intent_templates(self):
        """
        Define template patterns and vocabulary for generating diverse queries.
        
        Each intent has multiple templates with different phrasings and structures
        to create realistic variations of how users might express their needs.
        """
        
        # Salary inquiry templates and vocabulary
        self.salary_templates = [
            "What is my {period} salary?",
            "Can you show me my {type} information?",
            "I need to check my {period} {type}",
            "How much do I earn {period}?",
            "What's my current {type}?",
            "Can I see my {type} details?",
            "I want to know my {period} compensation",
            "Please show my {type} breakdown",
            "What am I being paid {period}?",
            "I need my {type} statement"
        ]
        
        self.salary_vocab = {
            "period": ["monthly", "annual", "yearly", "current", "this month", "this year"],
            "type": ["salary", "pay", "compensation", "wage", "earnings", "paycheck", "income"]
        }
        
        # Meeting room booking templates
        self.meeting_templates = [
            "I need to book a {room_type} for {time}",
            "Can I reserve {room_type} {time}?",
            "Book me a {room_type} for {time}",
            "I want to schedule a {room_type} {time}",
            "Is {room_type} available {time}?",
            "Reserve a {room_type} for {time}",
            "I need a {room_type} {time}",
            "Can you book {room_type} {time}?",
            "Schedule me a {room_type} for {time}",
            "I'd like to reserve a {room_type} {time}"
        ]
        
        self.meeting_vocab = {
            "room_type": ["meeting room", "conference room", "boardroom", "room", "conference hall", "meeting space"],
            "time": ["tomorrow", "next week", "today", "this afternoon", "Monday", "Tuesday", "Friday", "next month", "at 2pm", "for 3 hours"]
        }
        
        # Leave request templates
        self.leave_templates = [
            "I want to take {leave_type} {time}",
            "Can I request {leave_type} for {time}?",
            "I need {leave_type} {time}",
            "I'd like to apply for {leave_type} {time}",
            "Request {leave_type} for {time}",
            "I want to book {leave_type} {time}",
            "Can I have {leave_type} {time}?",
            "I need to take {leave_type} {time}",
            "Apply for {leave_type} {time}",
            "I'm requesting {leave_type} for {time}"
        ]
        
        self.leave_vocab = {
            "leave_type": ["vacation", "sick leave", "time off", "personal leave", "annual leave", "PTO", "holiday"],
            "time": ["next week", "tomorrow", "Monday", "this Friday", "next month", "two days", "a week", "three days"]
        }
        
        # Directory search templates
        self.directory_templates = [
            "Find {person} in the {directory}",
            "Look up {person} in {directory}",
            "Search for {person} contact",
            "I need {person}'s {contact_type}",
            "What's {person}'s {contact_type}?",
            "Can you find {person}'s details?",
            "Search {directory} for {person}",
            "I'm looking for {person}",
            "Find {person}'s {contact_type}",
            "Look up {person}'s information"
        ]
        
        self.directory_vocab = {
            "person": ["John Smith", "Sarah Johnson", "Mike Brown", "Lisa Davis", "Tom Wilson", "employee", "person", "staff member"],
            "directory": ["company directory", "employee directory", "staff directory", "phonebook", "contact list"],
            "contact_type": ["phone number", "email", "extension", "contact info", "details", "information"]
        }
        
        # Company info templates
        self.company_templates = [
            "What is the company {info_type}?",
            "Tell me about our {info_type}",
            "I need {info_type} information",
            "Can you provide {info_type} details?",
            "What's our company {info_type}?",
            "I want to know about {info_type}",
            "Show me {info_type} information",
            "What is our {info_type}?",
            "I need to know the {info_type}",
            "Can I get {info_type} details?"
        ]
        
        self.company_vocab = {
            "info_type": ["policy", "mission", "values", "history", "address", "phone number", "website", "headquarters", "locations", "departments"]
        }
        
        # Employee info templates
        self.employee_templates = [
            "What is {employee}'s {info_type}?",
            "Tell me about {employee}",
            "I need {employee}'s {info_type}",
            "Can you show {employee}'s details?",
            "What's {employee}'s {info_type}?",
            "I want {employee}'s information",
            "Show me {employee}'s {info_type}",
            "Get {employee}'s {info_type}",
            "I need to know {employee}'s {info_type}",
            "Find {employee}'s {info_type}"
        ]
        
        self.employee_vocab = {
            "employee": ["John's", "Sarah's", "my manager's", "the HR director's", "my colleague's", "the CEO's"],
            "info_type": ["department", "position", "title", "role", "email", "phone", "office location", "reports"]
        }
        
        # Employee search templates
        self.search_templates = [
            "Find employees at {company} named {name}",
            "Search for {name} at {company}",
            "Look up {name} from {company}",
            "Who is {name} at {company}?",
            "Find {name} working at {company}",
            "Search {company} for {name}",
            "I'm looking for {name} at {company}",
            "Find staff named {name} at {company}",
            "Look for {name} in {company}",
            "Search for employees named {name} at {company}"
        ]
        
        self.search_vocab = {
            "company": ["ABC Corp", "XYZ Inc", "TechCorp", "GlobalTech", "InnovateCo", "the company", "our organization"],
            "name": ["Smith", "Johnson", "Brown", "Davis", "Wilson", "Taylor", "Anderson", "Thomas"]
        }
    
    def _generate_samples_for_intent(self, intent: str, templates: List[str], vocab: Dict[str, List[str]]) -> List[str]:
        """
        Generate training samples for a specific intent.
        
        Args:
            intent: The intent label
            templates: List of template strings with placeholders
            vocab: Dictionary mapping placeholders to possible values
            
        Returns:
            List of generated text samples
        """
        samples = []
        
        for _ in range(self.samples_per_intent):
            # Randomly select a template
            template = random.choice(templates)
            
            # Fill in placeholders with random vocabulary
            sample = template
            for placeholder, options in vocab.items():
                if f"{{{placeholder}}}" in sample:
                    sample = sample.replace(f"{{{placeholder}}}", random.choice(options))
            
            samples.append(sample)
        
        return samples
    
    def generate_dataset(self) -> Tuple[List[str], List[str]]:
        """
        Generate the complete dataset for all intents.
        
        Returns:
            Tuple of (texts, labels) where texts are the input queries
            and labels are the corresponding intent labels
        """
        texts = []
        labels = []
        
        # Generate samples for each intent
        for intent in self.intent_labels:
            if intent == "salary_inquiry":
                samples = self._generate_samples_for_intent(intent, self.salary_templates, self.salary_vocab)
            elif intent == "meeting_room_booking":
                samples = self._generate_samples_for_intent(intent, self.meeting_templates, self.meeting_vocab)
            elif intent == "leave_request":
                samples = self._generate_samples_for_intent(intent, self.leave_templates, self.leave_vocab)
            elif intent == "directory_search":
                samples = self._generate_samples_for_intent(intent, self.directory_templates, self.directory_vocab)
            elif intent == "company_info":
                samples = self._generate_samples_for_intent(intent, self.company_templates, self.company_vocab)
            elif intent == "employee_info":
                samples = self._generate_samples_for_intent(intent, self.employee_templates, self.employee_vocab)
            elif intent == "employee_search":
                samples = self._generate_samples_for_intent(intent, self.search_templates, self.search_vocab)
            
            # Add samples and labels
            texts.extend(samples)
            labels.extend([intent] * len(samples))
        
        return texts, labels
    
    def create_train_test_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split of the dataset.
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df) DataFrames
        """
        texts, labels = self.generate_dataset()
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split by intent to ensure balanced representation
        train_dfs = []
        test_dfs = []
        
        for intent in self.intent_labels:
            intent_df = df[df['label'] == intent]
            split_idx = int(len(intent_df) * (1 - test_size))
            
            train_dfs.append(intent_df.iloc[:split_idx])
            test_dfs.append(intent_df.iloc[split_idx:])
        
        train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        return train_df, test_df
    
    def save_dataset(self, output_dir: str = "data"):
        """
        Generate and save the dataset to files.
        
        Args:
            output_dir: Directory to save the dataset files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate train/test split
        train_df, test_df = self.create_train_test_split()
        
        # Save as CSV files
        train_df.to_csv(output_path / "train.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        # Save as JSON files for easier loading
        train_data = {
            'texts': train_df['text'].tolist(),
            'labels': train_df['label'].tolist()
        }
        test_data = {
            'texts': test_df['text'].tolist(),
            'labels': test_df['label'].tolist()
        }
        
        with open(output_path / "train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path / "test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save label mapping
        label_mapping = {
            'intent_labels': self.intent_labels,
            'label_to_id': {label: i for i, label in enumerate(self.intent_labels)},
            'id_to_label': {i: label for i, label in enumerate(self.intent_labels)}
        }
        
        with open(output_path / "label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Intents: {len(self.intent_labels)}")
        
        # Print sample distribution
        print("\nSample distribution:")
        print(train_df['label'].value_counts().sort_index())


def main():
    """Main function to create and save the dataset."""
    print("Creating Office Domain Intent Recognition Dataset...")
    
    # Create dataset with 100 samples per intent (700 total samples)
    creator = OfficeIntentDatasetCreator(samples_per_intent=100)
    
    # Save dataset to data directory
    creator.save_dataset("data")
    
    print("\nDataset creation completed!")
    print("Files created:")
    print("- data/train.csv")
    print("- data/test.csv") 
    print("- data/train.json")
    print("- data/test.json")
    print("- data/label_mapping.json")


if __name__ == "__main__":
    main()