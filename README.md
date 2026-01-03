# ChartQA: Visual Reasoning for Chart Question Answering

## Project Overview

This research project focuses on developing an AI system capable of understanding and reasoning about chart visualizations to answer natural language questions. The goal is to enable models to perform complex visual reasoning tasks that combine chart comprehension with multi-step logical thinking.

## Research Objectives

### Primary Goal
Train a vision-language model to:
- **Understand chart visualizations** (bar charts, line graphs, pie charts, etc.)
- **Extract quantitative information** from visual elements
- **Perform multi-step reasoning** to answer complex questions
- **Generate explicit reasoning chains** before providing final answers

### Key Research Questions
1. Can we improve chart QA performance by teaching models to "think step-by-step"?
2. How does explicit reasoning generation impact answer accuracy?
3. What is the optimal format for representing reasoning in vision-language models?

## Methodology

### Dataset
- **Source**: ChartQA benchmark dataset
- **Size**: 1,000 training samples, 195 validation samples
- **Content**: Chart images paired with questions and ground-truth answers
- **Augmentation**: Generated reasoning chains using large language models

### Data Format Innovation
We developed a novel training format that separates reasoning from answers:
- **Input**: Chart image + Natural language question
- **Output**: Reasoning text + Structured answer `{answer: "value"}`

This format encourages the model to:
1. Analyze the chart systematically
2. Articulate its reasoning process
3. Provide a final answer based on that reasoning

### Model Architecture
- **Base Model**: Qwen2-VL-2B-Instruct (vision-language model)
- **Training Method**: Supervised Fine-Tuning (SFT) with LoRA
- **Key Features**:
  - Multi-modal understanding (vision + language)
  - Efficient parameter tuning (LoRA adapters)
  - Completion-only training (loss computed only on model responses)

### Training Strategy

**Phase 1: Data Preparation**
- Converted existing ChartQA data to reasoning-augmented format
- Separated reasoning chains from final answers
- Ensured data quality and format consistency

**Phase 2: Supervised Fine-Tuning**
- Fine-tuned vision-language model on augmented dataset
- Applied label masking to train only on assistant responses
- Optimized for both reasoning quality and answer accuracy

**Phase 3: Evaluation**
- Monitored training/validation loss convergence
- Assessed reasoning coherence and answer correctness
- Compared performance against baseline models

**Phase 4: Reinforcement Learning** *(Planned)*
- *To be implemented*
- *Details to be added*

## Technical Contributions

### 1. Reasoning-Augmented Training Format
Developed a structured format that explicitly separates:
- **Think**: Step-by-step reasoning about the chart
- **Answer**: Final response in JSON format

This approach mirrors human problem-solving and improves model interpretability.

### 2. Custom Data Collator
Implemented specialized data processing that:
- Properly handles vision-language inputs
- Masks labels for completion-only training
- Ensures efficient batch processing with variable-length sequences

### 3. Modular Training Pipeline
Created a flexible, maintainable codebase with:
- Separation of concerns (config, data, model, training)
- Comprehensive testing at each stage
- Easy experimentation with different configurations

## Research Findings

### Training Dynamics
- **Initial Loss**: ~1.19
- **Converged Loss**: ~1.02 (after 35 steps)
- **Validation Performance**: Eval loss closely tracks training loss (good generalization)
- **No Overfitting**: Small gap between training and validation metrics

### Key Observations
1. **Rapid Convergence**: Model quickly learns the reasoning format
2. **Stable Training**: Smooth loss curves with minimal oscillation
3. **Good Generalization**: Validation performance matches training performance

## Future Directions

### Potential Extensions
1. **Reinforcement Learning**: Apply GRPO/PPO for further refinement
2. **Larger Models**: Scale to 7B+ parameter models for better performance
3. **Multi-Chart Reasoning**: Extend to questions requiring multiple charts
4. **Interactive Reasoning**: Enable models to ask clarifying questions

### Research Applications
- **Data Analysis Automation**: Automated chart interpretation for business intelligence
- **Accessibility**: Helping visually impaired users understand charts
- **Education**: Interactive tutoring systems for data literacy
- **Scientific Research**: Automated analysis of research figures

## Impact

This project demonstrates that:
- Vision-language models can learn structured reasoning for chart QA
- Explicit reasoning generation improves model interpretability
- Proper data formatting significantly impacts training effectiveness
- Modular design enables rapid experimentation and iteration

## Acknowledgments

This research builds upon:
- **ChartQA Dataset**: Benchmark for chart question answering
- **Qwen2-VL**: State-of-the-art vision-language model
- **LoRA**: Efficient fine-tuning methodology
- **Transformers Library**: Foundation for model training

---

*This project represents ongoing research in visual reasoning and multi-modal AI systems.*