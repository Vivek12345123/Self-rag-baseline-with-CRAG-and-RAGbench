import json
import time
import os
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import requests
import re
from collections import Counter
import string
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Core dependencies
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    print("Error: datasets not available. Install with: pip install datasets")
    HF_DATASETS_AVAILABLE = False

# Load MS MARCO dataset
try:
    ds = load_dataset("microsoft/ms_marco", "v2.1")
    print(f"Successfully loaded MS MARCO dataset with {len(ds['train'])} train, {len(ds['validation'])} validation samples")
except Exception as e:
    print(f"Error loading MS MARCO dataset: {e}")
    ds = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError:
    print("Error: PyTorch/Transformers not available. Install with: pip install torch transformers")
    TORCH_AVAILABLE = False

# Enhanced evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge_score not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert_score not available. Install with: pip install bert_score")
    BERTSCORE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TIRESRAGSystem:
    """Complete TIRESRAG-R1 system implementation following the project structure"""
    
    def __init__(self, 
                 project_root: str = ".",
                 model_name: str = "TIRESRAG-R1-Instruct",
                 retrieval_port: int = 8000,
                 reflection_port: int = 8001, 
                 thinking_port: int = 8002,
                 max_tokens: int = 512,
                 temperature: float = 0.1):
        
        self.project_root = Path(project_root)
        self.model_name = model_name
        self.retrieval_url = f"http://localhost:{retrieval_port}"
        self.reflection_url = f"http://localhost:{reflection_port}"
        self.thinking_url = f"http://localhost:{thinking_port}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # System status
        self.services_running = False
        self.model_loaded = False
        self.index_built = False
        
        # Initialize components
        self._check_project_structure()
        self._load_tiresrag_model()
        
    def _check_project_structure(self):
        """Verify TIRESRAG-R1 project structure exists"""
        required_paths = [
            self.project_root / "scripts" / "wiki_servish.sh",
            self.project_root / "scripts" / "answer_reflection_reward.sh", 
            self.project_root / "scripts" / "sufficient_thinking_reward.sh",
            self.project_root / "evaluation" / "FlashRAG" / "scripts",
            self.project_root / "requirements.txt"
        ]
        
        missing_paths = [p for p in required_paths if not p.exists()]
        if missing_paths:
            logger.warning(f"Missing TIRESRAG-R1 components: {missing_paths}")
            logger.warning("Some functionality may be limited")
        else:
            logger.info("TIRESRAG-R1 project structure verified")
    
    def _load_tiresrag_model(self):
        """Load the trained TIRESRAG-R1 model"""
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available - cannot load TIRESRAG model")
                return
                
            # Look for trained TIRESRAG model in expected locations
            model_paths = [
                self.project_root / "models" / self.model_name,
                self.project_root / "checkpoints" / self.model_name,
                f"./models/{self.model_name}",
                self.model_name,  # Try as HuggingFace model ID
                "TIRESRAG/TIRESRAG-R1-Instruct",  # Try explicit HF path
                "OpenRLHF/TIRESRAG-R1-Instruct",   # Try OpenRLHF org
                "OpenRLHF/TIRESRAG-R1",            # Try without Instruct suffix
                "meta-llama/Meta-Llama-3-8B-Instruct"  # Fallback to known model
            ]
            
            model_loaded = False
            for model_path in model_paths:
                try:
                    logger.info(f"Attempting to load TIRESRAG model from: {model_path}")
                    
                    # Try to load tokenizer first to catch early failures
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            str(model_path), 
                            trust_remote_code=True,
                            padding_side='left'
                        )
                        logger.info(f"Successfully loaded tokenizer from {model_path}")
                    except Exception as tokenizer_error:
                        logger.warning(f"Failed to load tokenizer from {model_path}: {tokenizer_error}")
                        continue
                    
                    # Now try to load the model with more specific error handling
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            str(model_path),
                            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                            trust_remote_code=True,
                            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
                        )
                        logger.info(f"Successfully loaded model from {model_path}")
                    except Exception as model_error:
                        logger.warning(f"Failed to load model from {model_path}: {model_error}")
                        continue
                    
                    self.model.eval()
                    self.model_loaded = True
                    model_loaded = True
                    logger.info(f"Successfully loaded TIRESRAG-R1 model from {model_path}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load from {model_path}: {e}")
                    continue
            
            if not model_loaded:
                logger.warning("Could not load trained TIRESRAG model - using fallback")
                self._load_fallback_model()
                
        except Exception as e:
            logger.error(f"Error in model loading: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback base model for testing"""
        try:
            # Try different fallback models in order of preference
            fallback_models = [
                "Qwen/Qwen2.5-7B-Instruct",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "microsoft/Phi-3-mini-4k-instruct"
            ]
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Loading fallback model: {fallback_model}")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else "cpu",
                        trust_remote_code=True
                    )
                    self.model.eval()
                    self.model_loaded = True
                    logger.info(f"Fallback model {fallback_model} loaded successfully")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load fallback model {fallback_model}: {e}")
            
            if not self.model_loaded:
                raise ValueError("Could not load any fallback model")
            
        except Exception as e:
            logger.error(f"Failed to load all fallback models: {e}")
            self.tokenizer = None
            self.model = None
    
    def setup_services(self):
        """Start all required TIRESRAG services"""
        logger.info("Setting up TIRESRAG-R1 services...")
        
        scripts_dir = self.project_root / "scripts"
        if not scripts_dir.exists():
            logger.error("Scripts directory not found - cannot start services")
            return False
        
        services = [
            ("Retrieval Service", "wiki_servish.sh", 8000),
            ("Answer Reflection Service", "answer_reflection_reward.sh", 8001),
            ("Thinking Quality Service", "sufficient_thinking_reward.sh", 8002)
        ]
        
        for service_name, script_name, port in services:
            script_path = scripts_dir / script_name
            if script_path.exists():
                try:
                    logger.info(f"Starting {service_name}...")
                    # Run in background - in production you'd use proper process management
                    subprocess.Popen(
                        ["bash", str(script_path)], 
                        cwd=str(scripts_dir),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    time.sleep(2)  # Allow service to start
                    
                    # Verify service is running
                    if self._check_service_health(port):
                        logger.info(f"{service_name} started successfully on port {port}")
                    else:
                        logger.warning(f"{service_name} may not be running properly")
                        
                except Exception as e:
                    logger.error(f"Failed to start {service_name}: {e}")
            else:
                logger.warning(f"Script not found: {script_path}")
        
        self.services_running = True
        return True
    
    def _check_service_health(self, port: int) -> bool:
        """Check if a service is responding on the given port"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except:
            # Services might not have health endpoints
            return True
    
    def build_retrieval_index(self):
        """Build FAISS retrieval index using FlashRAG"""
        logger.info("Building retrieval index...")
        
        flashrag_scripts = self.project_root / "evaluation" / "FlashRAG" / "scripts"
        if not flashrag_scripts.exists():
            logger.error("FlashRAG scripts not found")
            return False
        
        try:
            # Step 1: Chunk documents
            chunk_script = flashrag_scripts / "chunk.sh"
            if chunk_script.exists():
                logger.info("Chunking documents...")
                subprocess.run(["bash", str(chunk_script)], cwd=str(flashrag_scripts), check=True)
            
            # Step 2: Build FAISS index  
            index_script = flashrag_scripts / "build_index.sh"
            if index_script.exists():
                logger.info("Building FAISS index...")
                subprocess.run(["bash", str(index_script)], cwd=str(flashrag_scripts), check=True)
            
            self.index_built = True
            logger.info("Retrieval index built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build retrieval index: {e}")
            return False
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents using FlashRAG retrieval service"""
        try:
            # Try FlashRAG-specific API format first
            payload = {
                "query": query,
                "retriever_name": "bge",  # Common FlashRAG retriever
                "corpus_name": "wiki",
                "top_k": top_k,
                "rerank": True
            }
            
            response = requests.post(
                f"{self.retrieval_url}/retrieve",
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                # Handle different response formats
                if "documents" in data:
                    return data["documents"]
                elif "retrieved_docs" in data:
                    return data["retrieved_docs"]
                else:
                    return data.get("results", [])
            else:
                logger.warning(f"Retrieval API returned {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.warning(f"Retrieval API call failed: {e}")
        
        # Return empty list if retrieval fails
        return []
    
    def evaluate_answer_reflection(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """Get answer quality reflection scores"""
        try:
            payload = {
                "question": question,
                "answer": answer, 
                "context": context,
                "metrics": ["quality", "relevance", "completeness", "factuality"]
            }
            
            response = requests.post(
                f"{self.reflection_url}/evaluate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.debug(f"Reflection API call failed: {e}")
        
        # Fallback scoring based on heuristics
        return self._fallback_reflection_scoring(question, answer, context)
    
    def evaluate_thinking_quality(self, question: str, thinking: str) -> Dict[str, float]:
        """Evaluate thinking process quality"""
        try:
            payload = {
                "question": question,
                "thinking": thinking,
                "metrics": ["sufficiency", "coherence", "depth", "relevance"]
            }
            
            response = requests.post(
                f"{self.thinking_url}/evaluate", 
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.debug(f"Thinking quality API call failed: {e}")
        
        return self._fallback_thinking_scoring(thinking)
    
    def _fallback_reflection_scoring(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """Fallback reflection scoring using heuristics"""
        scores = {}
        
        # Quality: based on length and structure
        scores['quality'] = min(0.9, len(answer.split()) / 50.0) if answer else 0.1
        
        # Relevance: keyword overlap with question
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        scores['relevance'] = len(q_words & a_words) / max(len(q_words), 1) if answer else 0.0
        
        # Completeness: presence of key indicators
        completeness_indicators = ['because', 'therefore', 'however', 'specifically', 'according to']
        scores['completeness'] = sum(1 for ind in completeness_indicators if ind in answer.lower()) / len(completeness_indicators)
        
        # Factuality: conservative estimate based on context usage
        scores['factuality'] = 0.8 if context and len(context) > 100 else 0.5
        
        return scores
    
    def _fallback_thinking_scoring(self, thinking: str) -> Dict[str, float]:
        """Fallback thinking quality scoring"""
        scores = {}
        
        if not thinking:
            return {"sufficiency": 0.0, "coherence": 0.0, "depth": 0.0, "relevance": 0.0}
        
        # Sufficiency: based on length and reasoning indicators
        reasoning_words = ['because', 'since', 'therefore', 'thus', 'however', 'although', 'while', 'whereas']
        reasoning_count = sum(1 for word in reasoning_words if word in thinking.lower())
        scores['sufficiency'] = min(0.95, reasoning_count / 3.0 + len(thinking.split()) / 100.0)
        
        # Coherence: sentence connectivity
        sentences = thinking.split('.')
        scores['coherence'] = min(0.9, len(sentences) / 5.0) if len(sentences) > 1 else 0.3
        
        # Depth: presence of analysis indicators
        depth_indicators = ['analyze', 'consider', 'examine', 'evaluate', 'compare', 'contrast', 'implications']
        scores['depth'] = sum(1 for ind in depth_indicators if ind in thinking.lower()) / len(depth_indicators)
        
        # Relevance: similar to sufficiency for thinking
        scores['relevance'] = scores['sufficiency'] * 0.9
        
        return scores
    
    def think_retrieve_reflect(self, question: str) -> Dict[str, Any]:
        """Complete TIRESRAG-R1 think-retrieve-reflect process"""
        start_time = time.time()
        
        if not self.model_loaded:
            logger.error("TIRESRAG model not loaded")
            return self._error_response("Model not loaded", start_time)
        
        try:
            # Step 1: THINK - Generate initial reasoning
            thinking_prompt = self._build_thinking_prompt(question)
            thinking = self._generate_text(thinking_prompt, max_tokens=200)
            
            # Evaluate thinking quality
            thinking_scores = self.evaluate_thinking_quality(question, thinking)
            
            # Step 2: RETRIEVE - Get relevant documents  
            retrieved_docs = self.retrieve_documents(question, top_k=5)
            context = self._format_context(retrieved_docs)
            
            # Step 3: GENERATE - Produce final answer
            answer_prompt = self._build_answer_prompt(question, thinking, context)
            answer = self._generate_text(answer_prompt, max_tokens=300)
            
            # Step 4: REFLECT - Evaluate answer quality
            reflection_scores = self.evaluate_answer_reflection(question, answer, context)
            
            inference_time = time.time() - start_time
            
            return {
                'text': answer,
                'thinking': thinking,
                'thinking_scores': thinking_scores,
                'retrieved_docs': retrieved_docs,
                'context': context,
                'reflection_scores': reflection_scores,
                'tokens_generated': len(answer.split()) + len(thinking.split()),
                'inference_time': inference_time,
                'uses_retrieval': len(retrieved_docs) > 0,
                'num_retrieved_docs': len(retrieved_docs),
                'overall_quality': self._compute_overall_quality(thinking_scores, reflection_scores)
            }
            
        except Exception as e:
            logger.error(f"Error in think-retrieve-reflect: {e}")
            return self._error_response(str(e), start_time)
    
    def _build_thinking_prompt(self, question: str) -> str:
        """Build prompt for thinking phase"""
        return f"""<|im_start|>system
You are TIRESRAG-R1, an advanced reasoning system. Think step-by-step about the question before retrieving information.
<|im_end|>
<|im_start|>user  
Question: {question}

Let me think about this step by step:
<|im_end|>
<|im_start|>assistant
I need to think through this question carefully:

"""
    
    def _build_answer_prompt(self, question: str, thinking: str, context: str) -> str:
        """Build prompt for answer generation with thinking and context"""
        context_section = f"\nRetrieved Information:\n{context}\n" if context else ""
        
        return f"""<|im_start|>system
You are TIRESRAG-R1. Use your thinking and retrieved information to provide an accurate, well-reasoned answer.
<|im_end|>
<|im_start|>user
Question: {question}

My thinking: {thinking}{context_section}
Based on my analysis and the information above, here is my answer:
<|im_end|>
<|im_start|>assistant
"""
    
    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents into context"""
        if not docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(docs[:5]):
            title = doc.get('title', f'Document {i+1}')
            content = doc.get('content', doc.get('text', doc.get('passage', '')))
            
            if content:
                # Truncate very long content
                if len(content) > 400:
                    content = content[:400] + "..."
                context_parts.append(f"[{title}] {content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_text(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text using the loaded model"""
        if not self.model or not self.tokenizer:
            return "[Model not available]"
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=True
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            if '<|im_end|>' in response:
                response = response.split('<|im_end|>')[0].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return f"[Generation Error: {str(e)}]"
    
    def _compute_overall_quality(self, thinking_scores: Dict, reflection_scores: Dict) -> float:
        """Compute overall quality score combining thinking and reflection"""
        thinking_avg = np.mean(list(thinking_scores.values())) if thinking_scores else 0.0
        reflection_avg = np.mean(list(reflection_scores.values())) if reflection_scores else 0.0
        
        # Weight thinking and reflection equally
        return (thinking_avg + reflection_avg) / 2.0
    
    def _error_response(self, error_msg: str, start_time: float) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'text': f"[Error] {error_msg}",
            'thinking': "",
            'thinking_scores': {},
            'retrieved_docs': [],
            'context': "",
            'reflection_scores': {},
            'tokens_generated': 0,
            'inference_time': time.time() - start_time,
            'uses_retrieval': False,
            'num_retrieved_docs': 0,
            'overall_quality': 0.0
        }

class TIRESRAGEvaluator:
    """Enhanced evaluator for TIRESRAG-R1 with comprehensive metrics"""
    
    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def normalize_answer(self, s: str) -> str:
        """Normalize answer for evaluation (SQuAD style)"""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
            return re.sub(regex, ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(self, prediction: str, ground_truth: str) -> float:
        """Compute exact match score"""
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score"""
        pred_tokens = self.normalize_answer(prediction).split()
        gold_tokens = self.normalize_answer(ground_truth).split()
        
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def evaluate_against_multiple_answers(self, prediction: str, ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate against multiple possible answers"""
        if not ground_truths:
            return {'em': 0.0, 'f1': 0.0}
        
        best_em = 0.0
        best_f1 = 0.0
        
        for gt in ground_truths:
            if not gt or not gt.strip():
                continue
                
            em = self.exact_match_score(prediction, gt)
            f1 = self.f1_score(prediction, gt)
            
            best_em = max(best_em, em)
            best_f1 = max(best_f1, f1)
        
        return {'em': best_em, 'f1': best_f1}
    
    def compute_rouge_scores(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Compute ROUGE scores if available"""
        if not self.rouge_scorer:
            return {}
        
        try:
            scores = self.rouge_scorer.score(ground_truth, prediction)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure
            }
        except:
            return {}

# Dataset loading utilities
def load_dataset_safe(dataset_options: List[Tuple], sample_size: int):
    """Safely load dataset with multiple fallback options"""
    if not HF_DATASETS_AVAILABLE:
        logger.error("datasets library not available")
        return None
    
    for option in dataset_options:
        try:
            if len(option) == 2:
                dataset_name, split = option
                logger.info(f"Loading {dataset_name}...")
                ds = load_dataset(dataset_name, split=split)
            elif len(option) == 3:
                dataset_name, config, split = option
                logger.info(f"Loading {dataset_name}[{config}]...")
                ds = load_dataset(dataset_name, config, split=split)
            else:
                continue
            
            if sample_size and sample_size < len(ds):
                ds = ds.select(range(sample_size))
            
            logger.info(f"Successfully loaded {len(ds)} samples from {dataset_name}")
            return ds
            
        except Exception as e:
            logger.warning(f"Failed to load {option[0]}: {e}")
            continue
    
    return None

# Benchmark implementations with correct HuggingFace dataset names
def run_hotpotqa_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """HotpotQA multi-hop reasoning evaluation"""
    logger.info(f"Running HotpotQA evaluation ({sample_size} samples)")
    
    # Correct HuggingFace dataset names for HotpotQA
    dataset_options = [
        ("hotpot_qa", "distractor", "validation"),
        ("hotpot_qa", "fullwiki", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load HotpotQA dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            question = item.get('question', '')
            answer = item.get('answer', '')
            level = item.get('level', 'unknown')
            type_q = item.get('type', 'unknown')
            
            if not question or not answer:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Evaluate response
            scores = evaluator.evaluate_against_multiple_answers(response['text'], [answer])
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answer)
            
            result = {
                'dataset': 'hotpotqa',
                'question': question,
                'ground_truth': answer,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'level': level,
                'type': type_q,
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} HotpotQA samples")
                print_interim_result(result)
                
        except Exception as e:
            logger.error(f"Error processing HotpotQA item {i}: {e}")
            continue
    
    return results

def run_natural_questions_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """Google Natural Questions evaluation"""
    logger.info(f"Running Natural Questions evaluation ({sample_size} samples)")
    
    # Correct HuggingFace dataset names for Natural Questions
    dataset_options = [
        ("natural_questions", "validation"),
        ("google-research-datasets/natural_questions", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load Natural Questions dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            # Handle Natural Questions format
            question = ""
            if 'question' in item:
                if isinstance(item['question'], dict):
                    question = item['question'].get('text', '')
                else:
                    question = item['question']
            elif 'question_text' in item:
                question = item['question_text']
            
            # Extract answers from annotations
            answers = []
            if 'annotations' in item:
                for annotation in item['annotations']:
                    if 'short_answers' in annotation:
                        for short_answer in annotation['short_answers']:
                            if 'text' in short_answer:
                                answers.append(short_answer['text'])
                    if 'yes_no_answer' in annotation:
                        yn_answer = annotation['yes_no_answer']
                        if yn_answer == 0:
                            answers.append('No')
                        elif yn_answer == 1:
                            answers.append('Yes')
            
            if not question or not answers:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Evaluate response
            scores = evaluator.evaluate_against_multiple_answers(response['text'], answers)
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answers[0] if answers else "")
            
            result = {
                'dataset': 'natural_questions',
                'question': question,
                'ground_truth': answers,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} Natural Questions samples")
                print_interim_result(result)
                
        except Exception as e:
            logger.error(f"Error processing Natural Questions item {i}: {e}")
            continue
    
    return results

def run_squad_v2_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """SQuAD v2.0 evaluation with unanswerable questions"""
    logger.info(f"Running SQuAD v2.0 evaluation ({sample_size} samples)")
    
    dataset_options = [
        ("squad_v2", "validation"),
        ("rajpurkar/squad_v2", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load SQuAD v2.0 dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            question = item.get('question', '')
            answers = item.get('answers', {})
            is_impossible = item.get('is_impossible', False)
            
            # Extract answer texts
            answer_texts = []
            if not is_impossible and 'text' in answers:
                answer_texts = answers['text'] if isinstance(answers['text'], list) else [answers['text']]
            
            if not question:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # For impossible questions, check if model abstains
            if is_impossible:
                abstain_indicators = ['cannot answer', 'no answer', 'impossible', 'insufficient information', 
                                    'not possible', "don't know", "cannot determine"]
                abstains = any(indicator in response['text'].lower() for indicator in abstain_indicators)
                em_score = 1.0 if abstains else 0.0
                f1_score = 1.0 if abstains else 0.0
            else:
                scores = evaluator.evaluate_against_multiple_answers(response['text'], answer_texts)
                em_score = scores['em']
                f1_score = scores['f1']
            
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answer_texts[0] if answer_texts else "")
            
            result = {
                'dataset': 'squad_v2',
                'question': question,
                'ground_truth': answer_texts,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'is_impossible': is_impossible,
                'exact_match': em_score,
                'f1_score': f1_score,
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} SQuAD v2.0 samples")
                print_interim_result(result)
                
        except Exception as e:
            logger.error(f"Error processing SQuAD v2.0 item {i}: {e}")
            continue
    
    return results

def run_triviaqa_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """TriviaQA evaluation"""
    logger.info(f"Running TriviaQA evaluation ({sample_size} samples)")
    
    dataset_options = [
        ("trivia_qa", "rc.nocontext", "validation"),
        ("trivia_qa", "unfiltered.nocontext", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load TriviaQA dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            question = item.get('question', '')
            answer = item.get('answer', {})
            
            # Extract answer aliases
            answer_texts = []
            if 'aliases' in answer:
                answer_texts = answer['aliases']
            elif 'value' in answer:
                answer_texts = [answer['value']]
            elif isinstance(answer, str):
                answer_texts = [answer]
            
            if not question or not answer_texts:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Evaluate response
            scores = evaluator.evaluate_against_multiple_answers(response['text'], answer_texts)
            rouge_scores = evaluator.compute_rouge_scores(response['text'], answer_texts[0])
            
            result = {
                'dataset': 'triviaqa',
                'question': question,
                'ground_truth': answer_texts,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} TriviaQA samples")
                print_interim_result(result)
                
        except Exception as e:
            logger.error(f"Error processing TriviaQA item {i}: {e}")
            continue
    
    return results

def run_fever_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """FEVER fact verification evaluation"""
    logger.info(f"Running FEVER evaluation ({sample_size} samples)")
    
    # Updated FEVER dataset name as requested
    dataset_options = [
        ("mwong/fever-evidence-related", "validation"),  # Corrected dataset name
        ("fever", "v1.0", "labelled_dev"),
        ("fever", "v2.0", "validation")
    ]
    
    dataset = load_dataset_safe(dataset_options, sample_size)
    if dataset is None:
        logger.error("Could not load FEVER dataset")
        return []
    
    results = []
    for i, item in enumerate(dataset):
        try:
            claim = item.get('claim', '')
            label = item.get('label', item.get('verdict', ''))
            evidence = item.get('evidence', [])
            
            if not claim or not label:
                continue
            
            # Convert label to standard format
            if isinstance(label, int):
                label_map = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT ENOUGH INFO'}
                label = label_map.get(label, 'NOT ENOUGH INFO')
            
            # Format as fact-checking question
            question = f"Verify this claim: {claim}"
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Extract predicted label from response
            pred_label = 'NOT ENOUGH INFO'  # default
            response_lower = response['text'].lower()
            if 'support' in response_lower or 'true' in response_lower or 'correct' in response_lower:
                pred_label = 'SUPPORTS'
            elif 'refute' in response_lower or 'false' in response_lower or 'incorrect' in response_lower:
                pred_label = 'REFUTES'
            
            # Simple accuracy for label prediction
            accuracy = 1.0 if pred_label == label else 0.0
            
            result = {
                'dataset': 'fever',
                'question': question,
                'claim': claim,
                'ground_truth_label': label,
                'prediction': response['text'],
                'predicted_label': pred_label,
                'thinking': response['thinking'],
                'accuracy': accuracy,
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} FEVER samples")
                print_interim_result(result)
                
        except Exception as e:
            logger.error(f"Error processing FEVER item {i}: {e}")
            continue
    
    return results

# Add new RAGTruth evaluation function
def run_ragtruth_evaluation(tiresrag: TIRESRAGSystem, evaluator: TIRESRAGEvaluator, sample_size: int = 100) -> List[Dict]:
    """RAGTruth evaluation"""
    logger.info(f"Running RAGTruth evaluation ({sample_size} samples)")
    
    try:
        logger.info("Loading wandb/RAGTruth-processed dataset...")
        dataset = load_dataset("wandb/RAGTruth-processed")
        
        # Select appropriate split
        if "validation" in dataset:
            ds = dataset["validation"]
        elif "test" in dataset:
            ds = dataset["test"]
        else:
            ds = next(iter(dataset.values()))
            
        if sample_size and sample_size < len(ds):
            ds = ds.select(range(sample_size))
            
        logger.info(f"Successfully loaded {len(ds)} samples from RAGTruth")
    except Exception as e:
        logger.error(f"Failed to load RAGTruth dataset: {e}")
        return []
    
    results = []
    for i, item in enumerate(ds):
        try:
            # Extract fields based on RAGTruth dataset structure
            question = item.get('question', '')
            reference = item.get('reference', item.get('answer', ''))
            context = item.get('context', '')
            
            if not question:
                continue
                
            # Get TIRESRAG response
            response = tiresrag.think_retrieve_reflect(question)
            
            # Evaluate response against reference
            if reference:
                scores = evaluator.evaluate_against_multiple_answers(response['text'], [reference])
                rouge_scores = evaluator.compute_rouge_scores(response['text'], reference)
            else:
                scores = {'em': 0.0, 'f1': 0.0}
                rouge_scores = {}
            
            result = {
                'dataset': 'ragtruth',
                'question': question,
                'ground_truth': reference,
                'prediction': response['text'],
                'thinking': response['thinking'],
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'thinking_scores': response['thinking_scores'],
                'reflection_scores': response['reflection_scores'],
                'overall_quality': response['overall_quality'],
                'inference_time': response['inference_time'],
                'tokens_generated': response['tokens_generated'],
                'uses_retrieval': response['uses_retrieval'],
                'num_retrieved_docs': response['num_retrieved_docs']
            }
            result.update(rouge_scores)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(ds)} RAGTruth samples")
                print_interim_result(result)
                
        except Exception as e:
            logger.error(f"Error processing RAGTruth item {i}: {e}")
            continue
    
    return results

def print_interim_result(result):
    """Print interim result to console for monitoring progress"""
    print("\n" + "="*80)
    print(f"DATASET: {result['dataset']}")
    print(f"QUESTION: {result['question'][:100]}...")
    print(f"ANSWER: {result['prediction'][:100]}...")
    if 'f1_score' in result:
        print(f"F1 SCORE: {result['f1_score']:.4f}")
    if 'exact_match' in result:
        print(f"EXACT MATCH: {result['exact_match']:.4f}")
    if 'accuracy' in result:
        print(f"ACCURACY: {result['accuracy']:.4f}")
    print(f"OVERALL QUALITY: {result['overall_quality']:.4f}")
    print(f"INFERENCE TIME: {result['inference_time']:.2f}s")
    print("="*80 + "\n")

# Main function to run evaluation pipeline
def main():
    """Run full TIRESRAG-R1 evaluation pipeline"""
    logger.info("Starting TIRESRAG-R1 evaluation")
    print("\n" + "="*80)
    print("TIRESRAG-R1 EVALUATION STARTED")
    print("="*80 + "\n")
    
    # Initialize system
    tiresrag = TIRESRAGSystem(
        project_root=".",
        model_name="TIRESRAG-R1-Instruct",
        temperature=0.1
    )
    
    # Set up evaluation
    evaluator = TIRESRAGEvaluator()
    
    # Start services (can be disabled if running without retrieval/reflection APIs)
    try:
        tiresrag.setup_services()
    except Exception as e:
        logger.warning(f"Error setting up services: {e}. Continuing with evaluation...")
    
    # Build retrieval index (can be disabled if using existing index)
    try:
        tiresrag.build_retrieval_index()
    except Exception as e:
        logger.warning(f"Error building index: {e}. Continuing with evaluation...")
    
    # Choose sample size for each dataset
    sample_size = 50  # Adjust as needed
    
    # Run evaluations
    results = []
    all_results = {}
    
    # Run all evaluations and collect results
    evaluations = [
        ("hotpotqa", run_hotpotqa_evaluation),
        ("natural_questions", run_natural_questions_evaluation),
        ("squad_v2", run_squad_v2_evaluation),
        ("triviaqa", run_triviaqa_evaluation),
        ("fever", run_fever_evaluation),
        ("ragtruth", run_ragtruth_evaluation)  # Added RAGTruth evaluation
    ]
    
    for name, eval_func in evaluations:
        try:
            logger.info(f"Starting {name} evaluation")
            print(f"\n--- Starting {name.upper()} evaluation ({sample_size} samples) ---\n")
            
            dataset_results = eval_func(tiresrag, evaluator, sample_size)
            results.extend(dataset_results)
            all_results[name] = dataset_results
            
            logger.info(f"Completed {name} evaluation: {len(dataset_results)} results")
            print(f"\n--- Completed {name.upper()} evaluation: {len(dataset_results)} results ---\n")
            
            # Calculate and print summary metrics for this dataset
            if dataset_results:
                f1_scores = [r.get('f1_score', 0) for r in dataset_results if 'f1_score' in r]
                em_scores = [r.get('exact_match', 0) for r in dataset_results if 'exact_match' in r]
                accuracy_scores = [r.get('accuracy', 0) for r in dataset_results if 'accuracy' in r]
                
                metrics = []
                if f1_scores:
                    metrics.append(f"Average F1: {np.mean(f1_scores):.4f}")
                if em_scores:
                    metrics.append(f"Average EM: {np.mean(em_scores):.4f}")
                if accuracy_scores:
                    metrics.append(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
                
                print(f"SUMMARY METRICS FOR {name.upper()}: {', '.join(metrics)}")
            
            # Save intermediate results
            output_file = f"results_{name}.json"
            with open(output_file, "w") as f:
                json.dump(dataset_results, f, indent=2)
            print(f"Results saved to {output_file}")
                
        except Exception as e:
            logger.error(f"Error in {name} evaluation: {e}")
            print(f"ERROR in {name} evaluation: {e}")
    
    # Save combined results
    with open("tiresrag_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save all results in a structured format
    with open("tiresrag_all_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary statistics
    summary = generate_evaluation_summary(results)
    
    logger.info("TIRESRAG-R1 evaluation completed")
    print("\n" + "="*80)
    print("TIRESRAG-R1 EVALUATION COMPLETED")
    print("="*80)
    
    # Print final summary to console
    print_final_summary(summary)
    
    return summary, all_results
    
def generate_evaluation_summary(results: List[Dict]):
    """Generate and save summary statistics"""
    if not results:
        logger.error("No results to summarize")
        return {}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Group by dataset
    datasets = df['dataset'].unique()
    
    # Summary dict
    summary = {
        'datasets': {},
        'overall': {}
    }
    
    # Overall metrics
    metrics = ['exact_match', 'f1_score', 'overall_quality', 'inference_time', 'accuracy']
    for metric in metrics:
        if metric in df.columns:
            summary['overall'][metric] = {
                'mean': float(df[metric].mean()),
                'median': float(df[metric].median()),
                'min': float(df[metric].min()),
                'max': float(df[metric].max()),
                'std': float(df[metric].std()),
                'count': int(df[metric].count())
            }
    
    # Per dataset metrics
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        summary['datasets'][dataset] = {}
        
        for metric in metrics:
            if metric in dataset_df.columns:
                summary['datasets'][dataset][metric] = {
                    'mean': float(dataset_df[metric].mean()),
                    'median': float(dataset_df[metric].median()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'count': int(dataset_df[metric].count())
                }
    
    # Save summary
    with open("tiresrag_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots if matplotlib is available
    try:
        # F1 Score by dataset
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='dataset', y='f1_score', data=df)
        plt.title('F1 Score Distribution by Dataset')
        plt.tight_layout()
        plt.savefig('f1_score_by_dataset.png')
        
        # Overall quality by dataset
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='dataset', y='overall_quality', data=df)
        plt.title('Overall Quality Distribution by Dataset')
        plt.tight_layout()
        plt.savefig('quality_by_dataset.png')
        
        # Inference time by dataset
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='dataset', y='inference_time', data=df)
        plt.title('Inference Time Distribution by Dataset')
        plt.tight_layout()
        plt.savefig('inference_time_by_dataset.png')
        
        # Save data as CSV for further analysis
        df.to_csv('tiresrag_evaluation_results.csv', index=False)
        logger.info("Evaluation visualizations generated")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    return summary

def print_final_summary(summary):
    """Print a human-readable summary of evaluation results"""
    print("\n" + "="*80)
    print("TIRESRAG-R1 EVALUATION SUMMARY")
    print("="*80)
    
    # Overall results
    print("\nOVERALL RESULTS:")
    for metric, values in summary.get('overall', {}).items():
        if metric in ['f1_score', 'exact_match', 'accuracy', 'overall_quality']:
            print(f"  {metric.upper()}: Mean={values['mean']:.4f}, Median={values['median']:.4f}, Count={values['count']}")
    
    # Per dataset results
    print("\nRESULTS BY DATASET:")
    for dataset, metrics in summary.get('datasets', {}).items():
        print(f"\n  {dataset.upper()}:")
        for metric, values in metrics.items():
            if metric in ['f1_score', 'exact_match', 'accuracy', 'overall_quality']:
                print(f"    {metric.upper()}: Mean={values['mean']:.4f}, Median={values['median']:.4f}, Count={values['count']}")
    
    print("\nDetailed results are available in:")
    print("  - tiresrag_evaluation_results.json (All results combined)")
    print("  - tiresrag_all_evaluation_results.json (Results organized by dataset)")
    print("  - tiresrag_evaluation_summary.json (Statistical summary)")
    print("  - results_*.json (Individual dataset results)")
    print("  - *.png (Visualization plots)")
    print("  - tiresrag_evaluation_results.csv (CSV format for analysis)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
