from vllm import LLM, SamplingParams
import json, time, os, sys, logging, re, string
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
from datasets import load_dataset
from datasets.utils.file_utils import DownloadConfig
import time, random
from datasets import load_dataset
import itertools
import atexit, datetime
import torch.distributed as dist
from difflib import SequenceMatcher
import math


def load_dataset_retry(*args, retries=5, base_sleep=2.0, jitter=0.75, **kwargs):
    """
    Retry wrapper for HF load_dataset with exponential backoff + jitter.
    Works with both normal and streaming modes.
    """
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                raise
            sleep = (base_sleep ** (attempt - 1)) + random.uniform(0.0, jitter)
            logger.warning(
                f"load_dataset failed (attempt {attempt}/{retries}): {e}. "
                f"Retrying in {sleep:.1f}s"
            )
            time.sleep(sleep)

# Optional metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except Exception:
    print("Warning: rouge_score not available. pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except Exception:
    print("Warning: bert_score not available. pip install bert_score")
    BERTSCORE_AVAILABLE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except Exception:
    print("Warning: nltk not available. pip install nltk")
    NLTK_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    print("Warning: sentence-transformers not available. pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("selfrag_eval")

# Global download config (helps with transient 50x like your 504)
DC = DownloadConfig(max_retries=5)

# ----------------------- Model & Evaluator -----------------------

class SelfRAGModel:
    def __init__(self,
                 model_path: str = "selfrag/selfrag_llama2_7b",
                 download_dir: str = "/gscratch/h2lab/akari/model_cache",
                 dtype: str = "half"):
        self.model = LLM(model_path, download_dir=download_dir, dtype=dtype)
        # FIXED: Increased max_tokens from 512 to 1024 for better responses and scoring
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=2048, skip_special_tokens=False
        )

    def format_prompt(self, input_text, paragraph=None):
        prompt = f"### Instruction:\n{input_text}\n\n### Response:\n"
        if paragraph:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
        return prompt

    def extract_utility_score(self, text: str) -> int:
        """Enhanced utility score extraction with multiple patterns"""
        if not text:
            return 0
        
        # Try different patterns
        patterns = [
            r'\[Utility:(\d)\]',
            r'\[Utility:\s*(\d)\]',
            r'Utility:\s*(\d)',
            r'utility\s*:\s*(\d)',
            r'score:\s*(\d)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    score = int(matches[-1])  # Take the last match
                    if 1 <= score <= 5:
                        return score
                except ValueError:
                    continue
        
        return 0

    def extract_relevance(self, text: str) -> bool:
        """Enhanced relevance detection"""
        if not text:
            return False
        
        relevant_patterns = [
            r'\[Relevant\]',
            r'\[RELEVANT\]',
            r'relevant',
            r'RELEVANT'
        ]
        
        irrelevant_patterns = [
            r'\[Irrelevant\]',
            r'\[IRRELEVANT\]',
            r'irrelevant',
            r'IRRELEVANT'
        ]
        
        text_lower = text.lower()
        
        # Check for irrelevant first (more specific)
        for pattern in irrelevant_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Check for relevant
        for pattern in relevant_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False

    def extract_support(self, text: str) -> str:
        """Enhanced support level extraction"""
        if not text:
            return "unknown"
            
        text_lower = text.lower()
        
        if any(pattern in text_lower for pattern in ["fully supported", "full support", "[fully supported]"]):
            return "fully_supported"
        if any(pattern in text_lower for pattern in ["partially supported", "partial support", "[partially supported]"]):
            return "partially_supported"
        if any(pattern in text_lower for pattern in ["no support", "not supported", "contradictory", "[no support", "contradicted"]):
            return "no_support"
            
        return "unknown"

    def uses_retrieval(self, text: str) -> bool:
        """Enhanced retrieval detection"""
        if not text:
            return False
            
        retrieval_patterns = [
            r'\[Retrieve\]',
            r'\[RETRIEVE\]',
            r'\[Retrieval\]',
            r'<paragraph>',
            r'retrieve',
            r'retrieval'
        ]
        
        for pattern in retrieval_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def extract_final_answer(self, text: str) -> str:
        """
        IMPROVED: Better answer extraction that preserves meaningful content
        """
        if not text:
            return ""
            
        # Keep the original text for debugging
        original = text
        
        # Remove SelfRAG special tokens but preserve content between them
        # First, handle paragraph content specially - keep the content but remove tags
        text = re.sub(r'<paragraph>(.*?)</paragraph>', r'\1', text, flags=re.DOTALL)
        
        # Remove other special tokens more carefully
        special_tokens = [
            r'\[Retrieve\]',
            r'\[Retrieval\]',
            r'\[Relevant\]',
            r'\[Irrelevant\]',
            r'\[Fully supported\]',
            r'\[Partially supported\]',
            r'\[No support / Contradictory\]',
            r'\[Utility:\d\]',
            r'\[Continue to Use Evidence\]',
            r'\[No Retrieval\]',
        ]
        
        for token in special_tokens:
            text = re.sub(token, '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace but preserve structure
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        # If the result is too short or empty, try alternative extraction
        if len(text.strip()) < 5:
            # Look for content after "Response:" or similar patterns
            response_patterns = [
                r'Response:\s*(.+?)(?:\[|$)',
                r'Answer:\s*(.+?)(?:\[|$)',
                r'### Response:\s*(.+?)(?:\[|$)',
            ]
            
            for pattern in response_patterns:
                match = re.search(pattern, original, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    if len(extracted) > len(text):
                        text = extracted
                        break
        
        return text.strip()

class SelfRAGEvaluator:
    def __init__(self):
        self.rouge_scorer = (
            rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
            if ROUGE_AVAILABLE else None
        )
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")

    def normalize_answer(self, s: str) -> str:
        """Improved normalization that's less aggressive"""
        if not s:
            return ""
            
        def remove_articles(text): 
            return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)
        def white_space_fix(text): 
            return " ".join(text.split())
        def remove_punc(text): 
            # Keep some punctuation that might be meaningful (like apostrophes)
            return "".join(ch for ch in text if ch not in set(string.punctuation) or ch in ["'", "-"])
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def soft_normalize_answer(self, s: str) -> str:
        """Even softer normalization for better matching"""
        if not s:
            return ""
        return " ".join(s.lower().split())

    def exact_match_score(self, pred, gt) -> float:
        """Improved exact match with multiple normalization levels"""
        if not pred or not gt:
            return 0.0
            
        # Try different levels of normalization
        normalizations = [
            (self.soft_normalize_answer, "soft"),
            (self.normalize_answer, "standard"),
            (lambda x: x.lower().strip(), "minimal"),
        ]
        
        for norm_func, level in normalizations:
            if norm_func(pred) == norm_func(gt):
                return 1.0
        
        # Check if prediction contains the ground truth or vice versa
        pred_soft = self.soft_normalize_answer(pred)
        gt_soft = self.soft_normalize_answer(gt)
        
        if pred_soft and gt_soft:
            if gt_soft in pred_soft or pred_soft in gt_soft:
                # Partial match based on length ratio
                shorter = min(len(pred_soft), len(gt_soft))
                longer = max(len(pred_soft), len(gt_soft))
                if shorter > 0 and shorter / longer > 0.8:  # 80% length similarity
                    return 0.8
        
        return 0.0

    def f1_score(self, pred, gt) -> float:
        """Improved F1 score calculation"""
        pred_normalized = self.normalize_answer(pred)
        gt_normalized = self.normalize_answer(gt)
        
        pred_tokens = pred_normalized.split()
        gt_tokens = gt_normalized.split()
        
        if not pred_tokens and not gt_tokens: 
            return 1.0
        if not pred_tokens or not gt_tokens: 
            return 0.0
            
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0: 
            return 0.0
            
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        
        return 2 * precision * recall / (precision + recall)

    def rouge_scores(self, pred: str, gt: str) -> Dict[str, float]:
        """Calculate ROUGE scores if available"""
        if not self.rouge_scorer or not pred or not gt:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(gt, pred)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE scoring error: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def bleu_score(self, pred: str, gt: str) -> float:
        """Calculate BLEU score if NLTK is available"""
        if not NLTK_AVAILABLE or not pred or not gt:
            return 0.0
        
        try:
            pred_tokens = pred.lower().split()
            gt_tokens = [gt.lower().split()]  # BLEU expects list of reference lists
            
            if not pred_tokens or not gt_tokens[0]:
                return 0.0
            
            smoothie = SmoothingFunction().method4
            return sentence_bleu(gt_tokens, pred_tokens, smoothing_function=smoothie)
        except Exception as e:
            logger.warning(f"BLEU scoring error: {e}")
            return 0.0

    def semantic_similarity(self, pred: str, gt: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        if not self.sentence_model or not pred or not gt:
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([pred, gt])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity error: {e}")
            return 0.0

    def string_similarity(self, pred: str, gt: str) -> float:
        """Calculate string similarity using difflib"""
        if not pred or not gt:
            return 0.0
        
        return SequenceMatcher(None, pred.lower(), gt.lower()).ratio()

    def answer_length_ratio(self, pred: str, gt: str) -> float:
        """Calculate the ratio of answer lengths"""
        if not pred and not gt:
            return 1.0
        if not pred or not gt:
            return 0.0
        
        pred_len = len(pred.split())
        gt_len = len(gt.split())
        
        if gt_len == 0:
            return 0.0
        
        return min(pred_len, gt_len) / max(pred_len, gt_len)

    def evaluate_multiple_answers(self, prediction, ground_truths):
        """Comprehensive evaluation with multiple metrics"""
        if not ground_truths: 
            return self._empty_scores()
        
        # Remove empty ground truths
        ground_truths = [gt for gt in ground_truths if gt and gt.strip()]
        if not ground_truths:
            return self._empty_scores()
        
        best_scores = self._empty_scores()
        
        # Evaluate against each ground truth and keep the best scores
        for gt in ground_truths:
            scores = {
                'em': self.exact_match_score(prediction, gt),
                'f1': self.f1_score(prediction, gt),
                'string_sim': self.string_similarity(prediction, gt),
                'length_ratio': self.answer_length_ratio(prediction, gt),
                'semantic_sim': self.semantic_similarity(prediction, gt),
                'bleu': self.bleu_score(prediction, gt),
            }
            
            # Add ROUGE scores
            rouge_scores = self.rouge_scores(prediction, gt)
            scores.update(rouge_scores)
            
            # Keep best scores
            for metric, score in scores.items():
                best_scores[metric] = max(best_scores.get(metric, 0.0), score)
        
        return best_scores

    def _empty_scores(self):
        """Return empty score dictionary"""
        return {
            'em': 0.0,
            'f1': 0.0,
            'string_sim': 0.0,
            'length_ratio': 0.0,
            'semantic_sim': 0.0,
            'bleu': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }

evaluator = SelfRAGEvaluator()

# Safe generation wrapper
def safe_generate(model: SelfRAGModel, prompt: str):
    try:
        out = model.model.generate([prompt], model.sampling_params)[0]
        if not getattr(out, "outputs", None):
            return "", 0
        first = out.outputs[0]
        raw_response = first.text or ""
        
        # DEBUGGING: Log raw response occasionally
        if random.random() < 0.01:  # 1% of the time
            logger.info(f"Raw response sample: {raw_response[:200]}...")
        
        clean_response = model.extract_final_answer(raw_response)
        return clean_response, len(first.token_ids or [])
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "", 0

# ----------------------- Benchmarks -----------------------

def run_natural_questions_benchmark(model, sample_size: int = 200, streaming=False):
    """
    FIXED: Use proper Natural Questions format with correct answer extraction
    """
    logger.info(f"Running NQ with sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            # Use the correct official dataset
            ds_iter = load_dataset_retry("natural_questions", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("natural_questions", split="validation", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))

        results = []
        for i, item in enumerate(ds):
            try:
                # Extract question properly
                question_data = item.get("question", {})
                question = question_data.get("text", "") if isinstance(question_data, dict) else str(question_data)
                
                # CRITICAL FIX: NQ has annotations with short_answers (this is what NQ evaluation uses)
                annotations = item.get("annotations", [])
                answer_texts = []
                has_answer = False
                
                if annotations:
                    for ann in annotations:
                        short_answers = ann.get("short_answers", [])
                        if short_answers:
                            has_answer = True
                            for sa in short_answers:
                                if isinstance(sa, dict) and "text" in sa:
                                    answer_texts.append(sa["text"])
                                elif isinstance(sa, dict) and "start_token" in sa and "end_token" in sa:
                                    # Extract from document text using token positions
                                    document = item.get("document", {})
                                    tokens = document.get("tokens", [])
                                    if tokens:
                                        start_idx = sa["start_token"]
                                        end_idx = sa["end_token"]
                                        if 0 <= start_idx < len(tokens) and 0 <= end_idx <= len(tokens):
                                            answer_text = " ".join([tokens[j].get("token", "") for j in range(start_idx, end_idx)])
                                            if answer_text.strip():
                                                answer_texts.append(answer_text.strip())
                
                # Get document context (use limited HTML content)
                document = item.get("document", {})
                context_text = ""
                if "html" in document:
                    html_content = document["html"]
                    # Extract meaningful text from HTML (simplified)
                    import re
                    text_content = re.sub(r'<[^>]+>', ' ', html_content)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    context_text = text_content[:2000]  # Limit context length
                
                prompt = model.format_prompt(question, context_text if context_text.strip() else None)
                t0 = time.time()
                resp, tok_count = safe_generate(model, prompt)
                dt = time.time() - t0
                
                # DEBUGGING: Log some examples
                if i < 5:
                    logger.info(f"NQ Example {i}: Q='{question}' A='{resp}' GT={answer_texts} HasAnswer={has_answer}")
                
                # Handle no answer cases properly (many NQ questions have no answer)
                if not has_answer or not answer_texts:
                    no_ans_indicators = ["no answer", "cannot answer", "not provided", "unknown", "unanswerable", "not mentioned"]
                    detected_no_answer = any(ind in (resp.lower() if resp else "") for ind in no_ans_indicators)
                    scores = evaluator._empty_scores()
                    scores['em'] = 1.0 if detected_no_answer else 0.0
                    scores['f1'] = 1.0 if detected_no_answer else 0.0
                else:
                    scores = evaluator.evaluate_multiple_answers(resp, answer_texts)
                
                result = {
                    'dataset':'natural_questions','question':question,'response':resp,
                    'ground_truth_answers':answer_texts,'has_answer':has_answer,
                    'inference_time':dt,'tokens_generated':tok_count,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp)
                }
                result.update(scores)
                results.append(result)
                
                if (i+1)%10==0: 
                    avg_em = np.mean([r.get('em', 0) for r in results])
                    avg_f1 = np.mean([r.get('f1', 0) for r in results])
                    logger.info(f"NQ processed {i+1}/{len(ds) if not streaming else sample_size} (EM: {avg_em:.3f}, F1: {avg_f1:.3f})")
            except Exception as e:
                logger.error(f"NQ item {i} error: {e}", exc_info=True)
        logger.info(f"Natural Questions completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running Natural Questions: {e}", exc_info=True)
        return []

def run_trivia_qa_benchmark(model, sample_size: int = 200, streaming=False):
    """
    IMPROVED: TriviaQA with proper multi-evidence handling
    """
    logger.info(f"Running TriviaQA sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("trivia_qa", "rc", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("trivia_qa", "rc", split="validation", download_config=DC)
            if sample_size < len(ds): ds = ds.select(range(sample_size))

        results=[]
        for i,item in enumerate(ds):
            try:
                question = item.get("question","")
                
                # IMPROVED: Better answer extraction from TriviaQA format
                ans = item.get("answer", {}) or {}
                answer_texts = []
                if isinstance(ans, dict):
                    if "value" in ans and ans["value"]:
                        answer_texts.append(ans["value"])
                    if "aliases" in ans and ans["aliases"]:
                        answer_texts.extend([a for a in ans["aliases"] if a])
                elif isinstance(ans, str):
                    answer_texts.append(ans)

                # IMPROVED: Use multiple evidence sources as intended for TriviaQA
                context_parts = []
                
                # Add entity pages (Wikipedia-style evidence)
                entity_pages = item.get("entity_pages", [])
                for page in entity_pages[:2]:  # Use top 2 entity pages
                    if isinstance(page, dict):
                        wiki_context = page.get("wiki_context", "")
                        if wiki_context:
                            context_parts.append(wiki_context[:800])  # Limit per source
                
                # Add search results (web search evidence)
                search_results = item.get("search_results", [])
                for result in search_results[:2]:  # Use top 2 search results
                    if isinstance(result, dict):
                        search_context = result.get("search_context", "")
                        if search_context:
                            context_parts.append(search_context[:800])
                
                # Fallback to original context if no multi-evidence available
                if not context_parts:
                    context_text = item.get("context","") or ""
                    if context_text.strip():
                        context_parts.append(context_text)
                
                # Combine evidence sources
                context_text = "\n\n".join(context_parts) if context_parts else ""

                prompt = model.format_prompt(question, context_text if context_text.strip() else None)
                t0=time.time(); resp, tok = safe_generate(model, prompt); dt=time.time()-t0
                
                # DEBUGGING: Log some examples
                if i < 5:
                    logger.info(f"TriviaQA Example {i}: Q='{question}' A='{resp}' GT={answer_texts} NumEvidence={len(context_parts)}")
                
                scores = evaluator.evaluate_multiple_answers(resp, answer_texts) if answer_texts else evaluator._empty_scores()

                result = {
                    'dataset':'trivia_qa','question':question,'response':resp,
                    'ground_truth_answers':answer_texts,'inference_time':dt,'tokens_generated':tok,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp),
                    'has_context': bool(context_text), 'num_evidence_sources': len(context_parts)
                }
                result.update(scores)
                results.append(result)
                
                if (i+1)%10==0: 
                    avg_em = np.mean([r.get('em', 0) for r in results])
                    avg_f1 = np.mean([r.get('f1', 0) for r in results])
                    logger.info(f"TriviaQA processed {i+1}/{len(ds) if not streaming else sample_size} (EM: {avg_em:.3f}, F1: {avg_f1:.3f})")
            except Exception as e:
                logger.error(f"TriviaQA item {i} error: {e}", exc_info=True)
        logger.info(f"TriviaQA completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running TriviaQA: {e}", exc_info=True)
        return []

def run_hotpot_qa_benchmark(model, sample_size: int = 200, streaming=False):
    logger.info(f"Running HotpotQA(distractor) sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("hotpotqa/hotpot_qa","distractor", split="validation", download_config=DC)
            if sample_size < len(ds): ds = ds.select(range(sample_size))

        results=[]
        for i,item in enumerate(ds):
            try:
                question = item.get("question","")
                answer = item.get("answer","") or ""
                level = item.get("level","unknown")
                qtype = item.get("type","unknown")

                # context: list of [title, [sentences...]]
                context_texts=[]
                for pair in item.get("context", []):
                    if isinstance(pair,(list,tuple)) and len(pair)==2:
                        title, sentences = pair
                        if sentences:
                            context_texts.append(f"{title}: {' '.join(sentences)}")
                context_text = "\n".join(context_texts[:5])

                prompt = model.format_prompt(question, context_text if context_text.strip() else None)
                t0=time.time(); resp, tok = safe_generate(model, prompt); dt=time.time()-t0
                
                # DEBUGGING: Log some examples
                if i < 5:
                    logger.info(f"HotpotQA Example {i}: Q='{question}' A='{resp}' GT='{answer}'")
                
                scores = evaluator.evaluate_multiple_answers(resp, [answer]) if answer else evaluator._empty_scores()

                result = {
                    'dataset':'hotpot_qa','question':question,'response':resp,'ground_truth_answer':answer,
                    'level':level,'type':qtype,'inference_time':dt,'tokens_generated':tok,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp),
                    'num_context_paragraphs': len(context_texts)
                }
                result.update(scores)
                results.append(result)
                
                if (i+1)%10==0: 
                    avg_em = np.mean([r.get('em', 0) for r in results])
                    avg_f1 = np.mean([r.get('f1', 0) for r in results])
                    logger.info(f"HotpotQA processed {i+1}/{len(ds) if not streaming else sample_size} (EM: {avg_em:.3f}, F1: {avg_f1:.3f})")
            except Exception as e:
                logger.error(f"HotpotQA item {i} error: {e}", exc_info=True)
        logger.info(f"HotpotQA completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running HotpotQA: {e}", exc_info=True)
        return []

def run_squad_v2_benchmark(model, sample_size: int = 200, streaming=False):
    logger.info(f"Running SQuAD v2 sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("rajpurkar/squad_v2", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("rajpurkar/squad_v2", split="validation", download_config=DC)
            if sample_size < len(ds): ds = ds.select(range(sample_size))

        results=[]
        for i,item in enumerate(ds):
            try:
                question = item.get("question","")
                context = item.get("context","") or ""
                answers = item.get("answers", {}) or {}
                answer_texts = [a for a in (answers.get("text") or []) if a]
                is_impossible = (len(answer_texts)==0)

                prompt = model.format_prompt(question, context if context.strip() else None)
                t0=time.time(); resp, tok = safe_generate(model, prompt); dt=time.time()-t0

                # DEBUGGING: Log some examples
                if i < 5:
                    logger.info(f"SQuAD Example {i}: Q='{question}' A='{resp}' GT={answer_texts} Impossible={is_impossible}")

                if not is_impossible and answer_texts:
                    scores = evaluator.evaluate_multiple_answers(resp, answer_texts)
                else:
                    no_ans = ["no answer","cannot answer","not provided","unknown","unanswerable","no information"]
                    detected = any(ind in (resp.lower() if resp else "") for ind in no_ans)
                    scores = evaluator._empty_scores()
                    scores['em'] = 1.0 if detected else 0.0
                    scores['f1'] = 1.0 if detected else 0.0

                result = {
                    'dataset':'squad_v2','question':question,'response':resp,
                    'ground_truth_answers':answer_texts,'is_impossible':is_impossible,
                    'inference_time':dt,'tokens_generated':tok,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp)
                }
                result.update(scores)
                results.append(result)
                
                if (i+1)%10==0: 
                    avg_em = np.mean([r.get('em', 0) for r in results])
                    avg_f1 = np.mean([r.get('f1', 0) for r in results])
                    logger.info(f"SQuAD v2 processed {i+1}/{len(ds) if not streaming else sample_size} (EM: {avg_em:.3f}, F1: {avg_f1:.3f})")
            except Exception as e:
                logger.error(f"SQuAD v2 item {i} error: {e}", exc_info=True)
        logger.info(f"SQuAD v2 completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running SQuAD v2: {e}", exc_info=True)
        return []

def run_fever_benchmark(model, sample_size: int = 200, streaming: bool = False):
    """
    FIXED: FEVER (Fact Extraction and VERification) benchmark
    """
    logger.info(f"Running FEVER sample_size={sample_size} (streaming={streaming})")
    try:
        # Use validation split which has proper labels
        if streaming:
            ds_iter = load_dataset_retry("mwong/fever-evidence-related", streaming=True, download_config=DC)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("mwong/fever-evidence-related", split="paper_dev", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Failed to load FEVER: {e}", exc_info=True)
        return []

    results = []
    for i, item in enumerate(ds):
        try:
            claim = item.get("claim", "") or ""
            label = item.get("label", "") or ""   # 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'
            evidence = item.get("evidence", None)

            # Extract evidence text if available
            context_text = ""
            if evidence and isinstance(evidence, list):
                evidence_texts = []
                for ev_group in evidence:
                    if isinstance(ev_group, list):
                        for ev_item in ev_group:
                            if isinstance(ev_item, list) and len(ev_item) >= 3:
                                # Evidence format: [annotation_id, evidence_id, wiki_url, sent_id]
                                # Get the text from the evidence
                                ev_text = str(ev_item[2]) if len(ev_item) > 2 else ""
                                if ev_text and ev_text != "":
                                    evidence_texts.append(ev_text)
                context_text = "\n".join(evidence_texts[:3])  # Limit context

            # Build prompt for fact verification
            fact_prompt = f"Given the evidence, classify this claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO: {claim}"
            prompt = model.format_prompt(fact_prompt, context_text if context_text.strip() else None)

            t0 = time.time()
            resp, tok = safe_generate(model, prompt)
            dt = time.time() - t0

            # DEBUGGING: Log some examples
            if i < 5:
                logger.info(f"FEVER Example {i}: Claim='{claim}' A='{resp}' GT='{label}'")

            # Better FEVER evaluation - check for label keywords in response
            resp_upper = resp.upper()
            predicted_label = ""
            if "SUPPORT" in resp_upper and "NOT" not in resp_upper:
                predicted_label = "SUPPORTS"
            elif "REFUTE" in resp_upper:
                predicted_label = "REFUTES"
            elif "NOT ENOUGH" in resp_upper or "INSUFFICIENT" in resp_upper:
                predicted_label = "NOT ENOUGH INFO"
            
            em_score = 1.0 if predicted_label == label else 0.0
            scores = evaluator._empty_scores()
            scores['em'] = em_score
            scores['f1'] = em_score  # For classification, EM=F1

            result = {
                'dataset': 'fever',
                'claim': claim,
                'response': resp,
                'label': label,
                'predicted_label': predicted_label,
                'inference_time': dt,
                'tokens_generated': tok,
                'utility_score': model.extract_utility_score(resp),
                'is_relevant': model.extract_relevance(resp),
                'support_level': model.extract_support(resp),
                'uses_retrieval': model.uses_retrieval(resp),
                'has_context': bool(context_text.strip())
            }
            result.update(scores)
            results.append(result)

            if (i + 1) % 10 == 0:
                avg_em = np.mean([r.get('em', 0) for r in results])
                logger.info(f"FEVER processed {i + 1}/{len(ds) if not streaming else sample_size} (Accuracy: {avg_em:.3f})")

        except Exception as e:
            logger.error(f"FEVER item {i} error: {e}", exc_info=True)

    logger.info(f"FEVER completed with {len(results)} samples")
    return results

    
def run_ms_marco_benchmark(model, sample_size: int = 200, streaming=False):
    """
    IMPROVED: MS MARCO with better format handling
    """
    logger.info(f"Running MS MARCO sample_size={sample_size} (streaming={streaming})")
    try:
        # Try different MS MARCO dataset variants
        dataset_loaded = False
        ds = None
        
        # Try multiple MS MARCO variants to find the working one
        variants = [
            ("ms_marco", "v1.1"),
            ("ms_marco", "v2.1"), 
            ("microsoft/ms_marco", "v2.1"),
            ("microsoft/ms_marco", "v1.1")
        ]
        
        for dataset_name, config in variants:
            try:
                if streaming:
                    ds_iter = load_dataset_retry(dataset_name, config, split="validation", streaming=True)
                    ds = list(itertools.islice(ds_iter, sample_size))
                else:
                    ds = load_dataset_retry(dataset_name, config, split="validation", download_config=DC)
                    if sample_size < len(ds): ds = ds.select(range(sample_size))
                dataset_loaded = True
                logger.info(f"Successfully loaded {dataset_name} {config}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name} {config}: {e}")
                continue
        
        if not dataset_loaded:
            logger.error("Could not load any MS MARCO variant")
            return []

        results=[]
        for i,item in enumerate(ds):
            try:
                # IMPROVED: Better field extraction for different MS MARCO formats
                query = item.get("query","") or item.get("question", "") or ""

                # Handle different passage formats
                passages = item.get("passages", {})
                context_text = ""
                
                if isinstance(passages, dict):
                    # Format: {"passage_text": [...], "is_selected": [...]}
                    texts = passages.get("passage_text", [])
                    is_selected = passages.get("is_selected", [])
                    
                    if isinstance(texts, list):
                        # Prioritize selected passages
                        selected_texts = []
                        other_texts = []
                        for j, text in enumerate(texts):
                            if j < len(is_selected) and is_selected[j]:
                                selected_texts.append(str(text))
                            else:
                                other_texts.append(str(text))
                        
                        # Use selected passages first, then others
                        all_passages = selected_texts + other_texts
                        context_text = "\n".join(all_passages[:5])
                        
                elif isinstance(passages, list):
                    # Format: [{"passage_text": "...", "is_selected": bool}, ...]
                    passage_texts = []
                    for passage in passages:
                        if isinstance(passage, dict):
                            text = passage.get("passage_text", "")
                            if text:
                                passage_texts.append(str(text))
                    context_text = "\n".join(passage_texts[:5])

                # IMPROVED: Better answer extraction
                answer_texts = []
                
                # Try different answer field names
                for field in ["answers", "wellFormedAnswers", "answer"]:
                    field_value = item.get(field, [])
                    if field_value:
                        if isinstance(field_value, list):
                            answer_texts.extend([str(a) for a in field_value if a])
                        elif isinstance(field_value, str):
                            answer_texts.append(field_value)

                prompt = model.format_prompt(query, context_text if context_text.strip() else None)
                t0=time.time(); resp, tok = safe_generate(model, prompt); dt=time.time()-t0
                
                # DEBUGGING: Log some examples
                if i < 5:
                    logger.info(f"MSMarco Example {i}: Q='{query}' A='{resp}' GT={answer_texts}")
                
                scores = evaluator.evaluate_multiple_answers(resp, answer_texts) if answer_texts else evaluator._empty_scores()

                result = {
                    'dataset':'msmarco','query':query,'response':resp,
                    'ground_truth_answers':answer_texts,'inference_time':dt,'tokens_generated':tok,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp),
                    'num_passages': len(passages.get("passage_text", [])) if isinstance(passages, dict) else len(passages) if isinstance(passages, list) else 0
                }
                result.update(scores)
                results.append(result)
                
                if (i+1)%10==0: 
                    avg_em = np.mean([r.get('em', 0) for r in results])
                    avg_f1 = np.mean([r.get('f1', 0) for r in results])
                    logger.info(f"MSMarco processed {i+1}/{len(ds) if not streaming else sample_size} (EM: {avg_em:.3f}, F1: {avg_f1:.3f})")
            except Exception as e:
                logger.error(f"MSMarco item {i} error: {e}", exc_info=True)
        logger.info(f"MSMarco completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running MSMarco: {e}", exc_info=True)
        return []

def run_ragtruth_benchmark(model, sample_size: int = 200, streaming: bool = False):
    """
    RAGTruth benchmark from wandb/RAGTruth-processed
    """
    logger.info(f"Running RAGTruth sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("wandb/RAGTruth-processed", split="train", streaming=True, download_config=DC)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("wandb/RAGTruth-processed", split="train", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Failed to load RAGTruth: {e}", exc_info=True)
        return []

    results = []
    for i, item in enumerate(ds):
        try:
            # RAGTruth fields
            question = item.get("question", "") or item.get("query", "") or ""
            context = item.get("context", "") or item.get("passage", "") or ""
            answer = item.get("answer", "") or item.get("ground_truth", "") or ""
            
            # Handle different answer formats
            if isinstance(answer, list):
                answer_texts = [str(a) for a in answer if a]
            else:
                answer_texts = [str(answer)] if answer else []

            prompt = model.format_prompt(question, context if context.strip() else None)
            t0 = time.time()
            resp, tok = safe_generate(model, prompt)
            dt = time.time() - t0

            # DEBUGGING: Log some examples
            if i < 5:
                logger.info(f"RAGTruth Example {i}: Q='{question}' A='{resp}' GT={answer_texts}")

            scores = evaluator.evaluate_multiple_answers(resp, answer_texts) if answer_texts else evaluator._empty_scores()

            result = {
                'dataset': 'ragtruth',
                'question': question,
                'response': resp,
                'ground_truth_answers': answer_texts,
                'inference_time': dt,
                'tokens_generated': tok,
                'utility_score': model.extract_utility_score(resp),
                'is_relevant': model.extract_relevance(resp),
                'support_level': model.extract_support(resp),
                'uses_retrieval': model.uses_retrieval(resp),
                'has_context': bool(context.strip())
            }
            result.update(scores)
            results.append(result)

            if (i + 1) % 10 == 0:
                avg_em = np.mean([r.get('em', 0) for r in results])
                avg_f1 = np.mean([r.get('f1', 0) for r in results])
                logger.info(f"RAGTruth processed {i + 1}/{len(ds) if not streaming else sample_size} (EM: {avg_em:.3f}, F1: {avg_f1:.3f})")

        except Exception as e:
            logger.error(f"RAGTruth item {i} error: {e}", exc_info=True)

    logger.info(f"RAGTruth completed with {len(results)} samples")
    return results

# ----------------------- Aggregation & I/O -----------------------

def compute_aggregate_metrics(results):
    """Enhanced aggregation with comprehensive metrics"""
    if not results: 
        return {}
    
    # Define all metrics to aggregate
    numerical_metrics = [
        'em', 'f1', 'utility_score', 'string_sim', 'length_ratio', 
        'semantic_sim', 'bleu', 'rouge1', 'rouge2', 'rougeL'
    ]
    
    boolean_metrics = ['is_relevant', 'uses_retrieval']
    
    aggregated = {}
    
    # Aggregate numerical metrics
    for metric in numerical_metrics:
        vals = [r.get(metric, 0.0) for r in results if metric in r and r.get(metric) is not None]
        if vals:
            aggregated[metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'count': len(vals),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals)),
                'q25': float(np.percentile(vals, 25)),
                'q75': float(np.percentile(vals, 75))
            }
        else:
            # Provide empty stats if no values
            aggregated[metric] = {
                'mean': 0.0, 'std': 0.0, 'count': 0, 'min': 0.0, 'max': 0.0,
                'median': 0.0, 'q25': 0.0, 'q75': 0.0
            }
    
    # Aggregate boolean metrics
    for metric in boolean_metrics:
        vals = [float(r.get(metric, False)) for r in results if metric in r]
        if vals:
            aggregated[metric] = {
                'mean': float(np.mean(vals)),
                'count': len(vals),
                'total_true': sum(vals),
                'total_false': len(vals) - sum(vals)
            }
    
    # Support level distribution
    support = Counter([r.get('support_level', 'unknown') for r in results])
    aggregated['support_distribution'] = dict(support)
    
    # Overall statistics
    aggregated['dataset_stats'] = {
        'total_samples': len(results),
        'avg_inference_time': float(np.mean([r.get('inference_time', 0) for r in results])),
        'total_inference_time': float(sum([r.get('inference_time', 0) for r in results])),
        'avg_tokens_generated': float(np.mean([r.get('tokens_generated', 0) for r in results])),
        'total_tokens_generated': sum([r.get('tokens_generated', 0) for r in results])
    }
    
    return aggregated

def save_results_to_json(results, filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}", exc_info=True)

def print_detailed_summary(results):
    """Print detailed summary with all metrics"""
    print("\n" + "="*80)
    print("🏆 SELF-RAG EVALUATION COMPLETE - DETAILED SUMMARY")
    print("="*80)
    
    succ = sum(1 for v in results.values() if v.get('total_samples', 0) > 0)
    total = sum(v.get('total_samples', 0) for v in results.values())
    
    for k, v in results.items():
        name = k.upper().replace("_", " ")
        if v.get('total_samples', 0) > 0:
            ag = v['aggregated_metrics']
            print(f"\n📈 {name}: n={v['total_samples']}  time={v.get('execution_time', 0):.2f}s")
            
            # Core metrics
            if 'em' in ag and ag['em']['count'] > 0:
                em = ag['em']
                print(f"   📍 Exact Match: {em['mean']:.3f} ± {em['std']:.3f} (median: {em['median']:.3f})")
            
            if 'f1' in ag and ag['f1']['count'] > 0:
                f1 = ag['f1']
                print(f"   🎯 F1 Score: {f1['mean']:.3f} ± {f1['std']:.3f} (median: {f1['median']:.3f})")
            
            # Additional metrics
            if 'string_sim' in ag and ag['string_sim']['count'] > 0:
                ss = ag['string_sim']
                print(f"   📝 String Similarity: {ss['mean']:.3f} ± {ss['std']:.3f}")
            
            if 'semantic_sim' in ag and ag['semantic_sim']['count'] > 0:
                sem = ag['semantic_sim']
                print(f"   🧠 Semantic Similarity: {sem['mean']:.3f} ± {sem['std']:.3f}")
            
            if 'rouge1' in ag and ag['rouge1']['count'] > 0:
                r1 = ag['rouge1']
                print(f"   🔴 ROUGE-1: {r1['mean']:.3f} ± {r1['std']:.3f}")
            
            if 'bleu' in ag and ag['bleu']['count'] > 0:
                bleu = ag['bleu']
                print(f"   🔵 BLEU: {bleu['mean']:.3f} ± {bleu['std']:.3f}")
            
            # Utility and retrieval info
            if 'utility_score' in ag:
                u = ag['utility_score']
                print(f"   ⚡ Utility Score: {u['mean']:.3f} ± {u['std']:.3f}")
            
            if 'uses_retrieval' in ag:
                ret = ag['uses_retrieval']
                print(f"   🔍 Uses Retrieval: {ret['mean']*100:.1f}% ({ret['total_true']}/{ret['count']})")
            
            if 'is_relevant' in ag:
                rel = ag['is_relevant']
                print(f"   ✅ Relevant Responses: {rel['mean']*100:.1f}% ({rel['total_true']}/{rel['count']})")
            
        else:
            status = v.get('status', 'no-data')
            error = v.get('error_message', '')
            print(f"\n❌ {name}: {status}")
            if error:
                print(f"    Error: {error[:100]}...")
    
    print("\n" + "="*80)
    print(f"📊 OVERALL SUMMARY:")
    print(f"   ✅ Successful benchmarks: {succ}/7")
    print(f"   📝 Total samples processed: {total}")
    
    # Calculate overall averages across all benchmarks
    all_ems = []
    all_f1s = []
    for v in results.values():
        if v.get('total_samples', 0) > 0:
            ag = v['aggregated_metrics']
            if 'em' in ag and ag['em']['count'] > 0:
                all_ems.append(ag['em']['mean'])
            if 'f1' in ag and ag['f1']['count'] > 0:
                all_f1s.append(ag['f1']['mean'])
    
    if all_ems:
        print(f"   🎯 Average EM across benchmarks: {np.mean(all_ems):.3f}")
    if all_f1s:
        print(f"   🎯 Average F1 across benchmarks: {np.mean(all_f1s):.3f}")
    
    print("="*80)

# ----------------------- Main -----------------------

def main():
    print("="*70)
    print("SELF-RAG EVALUATION (Enhanced with Comprehensive Scoring)")
    print("="*70)

    logger.info("Initializing Self-RAG model...")
    try:
        model = SelfRAGModel(
            model_path="selfrag/selfrag_llama2_7b",
            download_dir="/gscratch/h2lab/akari/model_cache",
            dtype="half"
        )
        logger.info("✅ Model init OK")
    except Exception as e:
        logger.error(f"❌ Model init failed: {e}", exc_info=True)
        return

    # Check available metrics
    logger.info(f"Available metrics: ROUGE={ROUGE_AVAILABLE}, BERT_Score={BERTSCORE_AVAILABLE}, NLTK={NLTK_AVAILABLE}, SentenceTransformers={SENTENCE_TRANSFORMERS_AVAILABLE}")

    # Smaller default to prove the loop, then scale up after it works
    sample_size = int(os.environ.get("SR_SAMPLE_SIZE", "200"))
    streaming = os.environ.get("SR_STREAMING", "0") == "1"

    results = {}
    # All benchmarks including fixed FEVER and RAGTruth
    benchmarks = [
        ("Natural Questions", run_natural_questions_benchmark),
        ("TriviaQA", run_trivia_qa_benchmark),
        ("HotpotQA", run_hotpot_qa_benchmark),
        ("SQuAD v2", run_squad_v2_benchmark),
        ("FEVER", run_fever_benchmark),
        ("MSMarco", run_ms_marco_benchmark),
        ("RAGTruth", run_ragtruth_benchmark),
    ]

    logger.info(f"Running {len(benchmarks)} benchmarks; sample_size={sample_size}; streaming={streaming}")
    for name, func in benchmarks:
        print(f"\n{'='*60}\n🚀 RUNNING: {name}\n{'='*60}")
        try:
            t0 = time.time()
            bench_results = func(model, sample_size=sample_size, streaming=streaming)
            t1 = time.time()
            key = name.lower().replace(" ", "_")
            if bench_results:
                aggregated = compute_aggregate_metrics(bench_results)
                results[key] = {
                    'individual_results': bench_results,
                    'aggregated_metrics': aggregated,
                    'total_samples': len(bench_results),
                    'execution_time': t1 - t0
                }
                logger.info(f"✅ {name}: {len(bench_results)} samples in {t1-t0:.2f}s")
                
                # Print quick stats
                if 'em' in aggregated and aggregated['em']['count'] > 0:
                    em_mean = aggregated['em']['mean']
                    f1_mean = aggregated['f1']['mean']
                    logger.info(f"   Quick stats - EM: {em_mean:.3f}, F1: {f1_mean:.3f}")
                    
            else:
                results[key] = {
                    'individual_results': [],
                    'aggregated_metrics': {},
                    'total_samples': 0,
                    'execution_time': t1 - t0,
                    'status': 'failed'
                }
                logger.warning(f"⚠️ {name} produced no results")
            save_results_to_json(results, f"selfrag_results_partial_{int(time.time())}.json")
        except Exception as e:
            logger.error(f"❌ Error running {name}: {e}", exc_info=True)
            key = name.lower().replace(" ", "_")
            results[key] = {
                'individual_results': [],
                'aggregated_metrics': {},
                'total_samples': 0,
                'execution_time': 0,
                'status': 'error',
                'error_message': str(e)
            }

    final = f"selfrag_evaluation_final_{int(time.time())}.json"
    save_results_to_json(results, final)

    # Detailed console summary
    print_detailed_summary(results)
    print(f"🗂  Results saved to: {final}")
    
    return results

def _dist_cleanup():
    try:
        if dist.is_available() and dist.is_initialized():
            # Optional: try a short barrier so peers don't race the destroy
            try:
                dist.barrier(timeout=datetime.timedelta(seconds=5))
            except Exception:
                pass
            dist.destroy_process_group()
    except Exception:
        pass

atexit.register(_dist_cleanup)

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print("🔥 SELF-RAG EVALUATION SYSTEM (Enhanced with Comprehensive Metrics)")
    print("="*70)
    print("🔍 Pre-flight checks...")

    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory/1e9
            print(f"✅ GPU: {name} ({mem:.1f} GB), {n} visible")
        else:
            print("⚠️ No GPU detected")
    except Exception:
        print("⚠️ PyTorch not available for GPU check")

    for pkg in ['vllm', 'datasets', 'transformers', 'torch']:
        try:
            __import__(pkg)
            print(f"✅ {pkg} available")
        except Exception:
            print(f"❌ {pkg} missing")

    print("\n🚀 Starting evaluation...")
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal: {e}", exc_info=True)
        print("❌ Evaluation failed. See logs.")
    finally:
        _dist_cleanup()
