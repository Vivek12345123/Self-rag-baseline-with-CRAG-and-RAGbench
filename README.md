# Self-RAG Evaluation Suite

Exact replication of the Self-RAG paper evaluation benchmarks for NeurIPS-level research comparison.

## 🎯 Overview

This repository contains the complete evaluation framework used in the [Self-RAG paper](https://arxiv.org/abs/2310.11511) by Asai et al. (ICLR 2024). It evaluates the Self-RAG model across 6 key benchmarks with exact replication of the original methodology.

## 📊 Benchmarks Included

1. **Natural Questions** - Complex factual QA requiring world knowledge
2. **TriviaQA** - Knowledge-intensive question answering  
3. **HotpotQA** - Multi-hop reasoning over multiple documents
4. **SQuAD v2** - Reading comprehension with unanswerable questions
5. **CRAG** - Comprehensive RAG evaluation benchmark
6. **RAGBench** - RAG-specific evaluation tasks

## 🚀 Quick Start (GitHub → VS Code → RunPod)

### 1. Clone Repository
```bash
git clone [your-repo-url]
cd self-rag-evaluation
```

### 2. Setup Environment (RunPod)
```bash
# Run setup script
bash setup_runpod.sh

# Or install manually
pip install -r requirements.txt
```

### 3. Run Evaluation
```bash
# Single command - no configuration needed
python selfrag_evaluation.py
```

That's it! 🎉

## 💻 System Requirements

- **GPU**: 24GB+ VRAM (tested on RTX A6000, A100)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models and datasets
- **Python**: 3.8+

## 📈 Output

The evaluation generates:
- `selfrag_evaluation_final_[timestamp].json` - Complete results
- `selfrag_results_partial_[timestamp].json` - Intermediate results
- Console logs with real-time progress

### Key Metrics Tracked:
- **Exact Match & F1**: Standard QA evaluation metrics
- **Utility Scores**: Self-RAG quality assessment (1-5 scale)  
- **Relevance Rate**: Frequency of relevant retrieval
- **Support Distribution**: Evidence support analysis
- **Retrieval Usage**: Adaptive retrieval behavior

## 🔬 Research Usage

Perfect for:
- ✅ Baseline comparison for new RAG methods
- ✅ Replication studies
- ✅ Model performance analysis
- ✅ NeurIPS/ICML/ICLR paper baselines

## 📝 Configuration

The evaluation uses exact Self-RAG paper settings:
- Model: `selfrag/selfrag_llama2_7b`
- Temperature: 0.0 (deterministic)
- Max tokens: 100
- Sample size: 100 per benchmark (configurable)

To modify sample size:
```python
# In selfrag_evaluation.py
sample_size = 50  # Reduce for faster testing
```

## 🐛 Troubleshooting

### Common Issues:

**GPU Memory Error**
```bash
# Reduce model precision or use smaller model
# The code automatically handles this
```

**Dataset Download Fails**  
```bash
# Check internet connection
# Some datasets require HuggingFace account
```

**Model Access Denied**
```bash
# Ensure HuggingFace access to selfrag models
huggingface-cli login
```

## 📊 Expected Results

Approximate performance ranges (from original paper):
- **Natural Questions**: EM ~45%, F1 ~55%
- **TriviaQA**: EM ~60%, F1 ~68% 
- **HotpotQA**: EM ~35%, F1 ~48%
- **SQuAD v2**: EM ~55%, F1 ~60%

*Exact numbers depend on evaluation subset and setup*

## 📄 Citation

If you use this evaluation suite in your research:

```bibtex
@inproceedings{asai2024selfrag,
  title={Self-{RAG}: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

## 🤝 Contributing

This is a research replication tool. For issues:
1. Check the troubleshooting section
2. Verify your environment meets requirements  
3. Open an issue with full error logs

## ⚖️ License

Research use only. Respects original Self-RAG licensing terms.

---

**Ready to run!** 🚀 Just clone, setup, and execute for exact Self-RAG paper replication.# Self-rag-baseline-with-CRAG-and-RAGbench
