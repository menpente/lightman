# Phase 1 Implementation Guide - Core Fake News Detection System

## Overview
Phase 1 focuses on building the essential fake news detection capabilities with basic content analysis, source credibility checking, manual review interface, and a simple dashboard.

## Architecture Components

### 1. Core Langflow Workflow Structure

```
Input Sources → Preprocessing → Content Analysis Agents → Decision Engine → Output Handler
     ↓              ↓                    ↓                    ↓             ↓
RSS/Manual → Text Cleaner → [Phrase + Language + → Evidence → Dashboard +
  Input        + Metadata     Commonsense + URL]    Aggregator   Database
```

## Detailed Implementation

### 1. Input Layer Implementation

#### A. RSS Feed Integration
**Langflow Nodes:**
- **RSS Reader Node** (Custom Component)
- **URL Validator Node**
- **Content Extractor Node**

**Configuration:**
```python
# RSS Reader Node Implementation
import feedparser
import requests
from datetime import datetime

class RSSFeedNode:
    def __init__(self):
        self.feed_urls = [
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.npr.org/1001/rss.xml",
            "https://www.politico.com/rss/politicopicks.xml"
        ]
    
    def process(self):
        articles = []
        for url in self.feed_urls:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                article = {
                    "title": entry.title,
                    "url": entry.link,
                    "published": entry.published,
                    "summary": entry.summary,
                    "source_domain": self.extract_domain(entry.link)
                }
                articles.append(article)
        return articles
```

#### B. Manual Input Interface
**Langflow Nodes:**
- **Text Input Node**
- **URL Input Node**
- **Form Validation Node**

### 2. Preprocessing Pipeline

#### A. Text Cleaning and Normalization
**Langflow Configuration:**
```yaml
Text Preprocessor Node:
  - Remove HTML tags
  - Normalize whitespace
  - Handle special characters
  - Extract key metadata (word count, caps ratio)
```

#### B. Content Classification
**Langflow Node: Content Classifier**
```python
# LLM Prompt for Content Classification
CLASSIFICATION_PROMPT = """
Analyze the following news article and classify it into one of these categories:
- Political
- Entertainment/Celebrity
- Health/Medical
- Technology
- Sports
- Business/Finance
- General News

Article Title: {title}
Article Content: {content}

Respond with just the category name and confidence (0-1):
Category: [category]
Confidence: [confidence]
"""
```

### 3. Core Analysis Agents

#### A. Phrase Analysis Agent
**Langflow Workflow:**
```
Input Text → LLM Analysis → Score Calculator → Evidence Collector
```

**LLM Prompt:**
```python
PHRASE_ANALYSIS_PROMPT = """
Analyze the following news article for indicators of potentially fake news based on language patterns:

Article: "{text}"

Check for:
1. Sensational teasers ("You won't believe...", "SHOCKING", etc.)
2. Emotional manipulation language
3. Excessive use of superlatives
4. All-caps words or phrases
5. Multiple exclamation marks
6. Vague or unsubstantiated claims

For each indicator found:
- Quote the specific text
- Explain why it's concerning
- Rate severity (1-5)

Format your response as:
FINDINGS: [list of specific issues found]
SEVERITY_SCORE: [1-5, where 5 is most concerning]
EXPLANATION: [brief reasoning]
"""
```

#### B. Language Quality Agent
**LLM Prompt:**
```python
LANGUAGE_QUALITY_PROMPT = """
Evaluate the following text for language quality issues that might indicate fake news:

Text: "{text}"

Analyze for:
1. Grammar errors
2. Spelling mistakes
3. Awkward phrasing
4. Inconsistent style
5. Poor punctuation usage
6. Non-native speaker patterns

Rate each category (0-5) and provide examples:

GRAMMAR_SCORE: [0-5]
SPELLING_SCORE: [0-5] 
STYLE_SCORE: [0-5]
OVERALL_QUALITY: [0-5]
EXAMPLES: [specific issues found]
ASSESSMENT: [summary of language quality]
"""
```

#### C. Common Sense Verification Agent
**LLM Prompt:**
```python
COMMONSENSE_PROMPT = """
Evaluate this news claim for common sense and logical consistency:

Claim: "{text}"

Consider:
1. Does this contradict well-known facts?
2. Are the claims plausible given current knowledge?
3. Do the details make logical sense?
4. Are there internal contradictions?
5. Does this sound like gossip rather than news?

PLAUSIBILITY: [High/Medium/Low]
CONTRADICTIONS: [list any found]
RED_FLAGS: [concerning elements]
REASONING: [explain your assessment]
"""
```

#### D. Source Credibility Agent
**Implementation:**
```python
# Domain Credibility Database (simplified)
DOMAIN_CREDIBILITY = {
    "bbc.com": {"score": 0.95, "bias": "center"},
    "cnn.com": {"score": 0.85, "bias": "center-left"},
    "reuters.com": {"score": 0.95, "bias": "center"},
    "infowars.com": {"score": 0.15, "bias": "extreme-right"},
    # Add more domains...
}

URL_ANALYSIS_PROMPT = """
Analyze this URL and domain for credibility indicators:

URL: {url}
Domain: {domain}

Consider:
1. Domain reputation and history
2. Professional appearance
3. Contact information availability
4. About page transparency
5. SSL certificate presence

CREDIBILITY_SCORE: [0-1]
WARNING_SIGNS: [list any concerns]
ASSESSMENT: [brief evaluation]
"""
```

### 4. Decision Engine Implementation

#### A. Evidence Aggregation Node
**Langflow Configuration:**
```python
class EvidenceAggregator:
    def __init__(self):
        self.weights = {
            "phrase_analysis": 0.25,
            "language_quality": 0.20,
            "commonsense": 0.30,
            "source_credibility": 0.25
        }
    
    def aggregate_scores(self, evidence):
        weighted_score = 0
        explanations = []
        
        for agent, score in evidence.items():
            if agent in self.weights:
                weighted_score += score * self.weights[agent]
                explanations.append(evidence[f"{agent}_explanation"])
        
        # Convert to credibility score (higher = more credible)
        credibility_score = 1 - weighted_score
        
        return {
            "credibility_score": credibility_score,
            "risk_level": self.get_risk_level(credibility_score),
            "detailed_analysis": explanations
        }
    
    def get_risk_level(self, score):
        if score >= 0.8: return "LOW_RISK"
        elif score >= 0.6: return "MEDIUM_RISK" 
        elif score >= 0.4: return "HIGH_RISK"
        else: return "VERY_HIGH_RISK"
```

### 5. Manual Review Interface

#### A. Review Dashboard Components
**Technology Stack:**
- Frontend: Streamlit or Gradio for rapid prototyping
- Backend: FastAPI
- Database: PostgreSQL

**Database Schema:**
```sql
-- Articles table
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    source_domain VARCHAR(255),
    published_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    credibility_score FLOAT,
    risk_level VARCHAR(20),
    review_status VARCHAR(20) DEFAULT 'pending'
);

-- Analysis results table
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    agent_name VARCHAR(50),
    score FLOAT,
    explanation TEXT,
    evidence JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Human reviews table
CREATE TABLE human_reviews (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    reviewer_id VARCHAR(100),
    verdict VARCHAR(20), -- 'real', 'fake', 'uncertain'
    confidence INTEGER, -- 1-5
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### B. Streamlit Review Interface
```python
import streamlit as st
import pandas as pd
from datetime import datetime

def create_review_interface():
    st.title("Fake News Detection - Manual Review")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    risk_filter = st.sidebar.selectbox(
        "Risk Level", 
        ["All", "VERY_HIGH_RISK", "HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]
    )
    
    # Main content area
    articles = load_pending_articles(risk_filter)
    
    for idx, article in articles.iterrows():
        with st.expander(f"Article {article['id']}: {article['title'][:100]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Title:** {article['title']}")
                st.write(f"**Source:** {article['source_domain']}")
                st.write(f"**URL:** {article['url']}")
                st.write(f"**Content Preview:** {article['content'][:500]}...")
                
            with col2:
                st.metric("Credibility Score", f"{article['credibility_score']:.2f}")
                st.metric("Risk Level", article['risk_level'])
                
                # Analysis breakdown
                st.write("**Agent Analysis:**")
                analysis = load_analysis_results(article['id'])
                for _, result in analysis.iterrows():
                    st.write(f"- {result['agent_name']}: {result['score']:.2f}")
                    st.write(f"  {result['explanation']}")
            
            # Review form
            st.write("**Your Assessment:**")
            verdict = st.radio(
                "Verdict:", 
                ["Real", "Fake", "Uncertain"], 
                key=f"verdict_{article['id']}"
            )
            confidence = st.slider(
                "Confidence:", 1, 5, 3, 
                key=f"confidence_{article['id']}"
            )
            notes = st.text_area(
                "Notes:", 
                key=f"notes_{article['id']}"
            )
            
            if st.button(f"Submit Review", key=f"submit_{article['id']}"):
                save_review(article['id'], verdict, confidence, notes)
                st.success("Review submitted!")
```

### 6. Basic Dashboard Implementation

#### A. Real-time Monitoring Dashboard
```python
def create_monitoring_dashboard():
    st.title("Fake News Detection - Live Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_articles = get_total_articles_today()
        st.metric("Articles Processed Today", total_articles)
    
    with col2:
        high_risk_count = get_high_risk_articles_today()
        st.metric("High Risk Articles", high_risk_count)
    
    with col3:
        accuracy = get_system_accuracy()
        st.metric("System Accuracy", f"{accuracy:.1%}")
    
    with col4:
        pending_reviews = get_pending_review_count()
        st.metric("Pending Reviews", pending_reviews)
    
    # Recent detections
    st.header("Recent High-Risk Detections")
    recent_high_risk = load_recent_high_risk_articles()
    st.dataframe(recent_high_risk[['title', 'source_domain', 'credibility_score', 'risk_level']])
    
    # Trends chart
    st.header("Detection Trends")
    trend_data = load_detection_trends()
    st.line_chart(trend_data)
```

## LLM Traceability with Opik Integration

### 1. Opik Setup and Configuration

#### A. Installation and Basic Setup
```bash
pip install opik
pip install opik[langchain]  # For LangChain integration
```

#### B. Opik Configuration
```python
# opik_config.py
import opik
from opik import Opik
from opik.integrations.langchain import OpikLangChainTracer
import os

# Initialize Opik client
opik_client = Opik(
    api_key=os.getenv("OPIK_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE", "fake-news-detection")
)

# Configure tracing
tracer = OpikLangChainTracer(
    project_name="fake-news-detection-v1",
    tags=["production", "phase1"]
)

# Custom trace decorator for our analysis functions
def trace_analysis(agent_name: str, input_data: dict):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with opik_client.trace(
                name=f"{agent_name}_analysis",
                input=input_data,
                tags=[agent_name, "analysis"],
                metadata={"agent_type": agent_name, "version": "1.0"}
            ) as trace:
                try:
                    result = func(*args, **kwargs)
                    trace.log_output(result)
                    trace.log_feedback_score(
                        name="processing_success", 
                        value=1.0
                    )
                    return result
                except Exception as e:
                    trace.log_output({"error": str(e)})
                    trace.log_feedback_score(
                        name="processing_success", 
                        value=0.0
                    )
                    raise e
        return wrapper
    return decorator
```

### 2. Enhanced Analysis Agents with Tracing

#### A. Phrase Analysis Agent with Opik Integration
```python
import opik
from opik.decorators import track

class PhrasAnalysisAgent:
    def __init__(self):
        self.opik_client = opik.Opik()
    
    @track(
        project_name="fake-news-detection",
        tags=["phrase-analysis", "content-analysis"]
    )
    def analyze_phrases(self, text: str, metadata: dict = None):
        """Analyze text for sensational phrases and emotional manipulation"""
        
        # Log input data
        input_data = {
            "text_length": len(text),
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
            "source_domain": metadata.get("source_domain", "unknown"),
            "article_id": metadata.get("article_id")
        }
        
        with self.opik_client.trace(
            name="phrase_analysis_detailed",
            input=input_data,
            tags=["llm-call", "gpt-3.5-turbo"]
        ) as trace:
            
            # Enhanced prompt with structured output
            prompt = f"""
            Analyze the following news article for indicators of potentially fake news based on language patterns:

            Article: "{text}"

            Check for:
            1. Sensational teasers ("You won't believe...", "SHOCKING", etc.)
            2. Emotional manipulation language
            3. Excessive use of superlatives
            4. All-caps words or phrases
            5. Multiple exclamation marks
            6. Vague or unsubstantiated claims

            Respond in JSON format:
            {{
                "sensational_phrases": ["phrase1", "phrase2"],
                "emotional_triggers": ["trigger1", "trigger2"],
                "caps_usage": {{"count": 0, "examples": []}},
                "exclamation_count": 0,
                "severity_score": 0.0,
                "explanation": "detailed reasoning",
                "confidence": 0.0
            }}
            """
            
            # Make LLM call with tracing
            try:
                response = self.llm_call(prompt, trace)
                result = self.parse_response(response)
                
                # Log structured output
                trace.log_output(result)
                
                # Log metrics
                trace.log_feedback_score(
                    name="severity_score", 
                    value=result.get("severity_score", 0)
                )
                trace.log_feedback_score(
                    name="confidence", 
                    value=result.get("confidence", 0)
                )
                
                # Log custom metrics
                self.log_phrase_metrics(trace, result)
                
                return result
                
            except Exception as e:
                trace.log_output({"error": str(e), "status": "failed"})
                raise e
    
    def log_phrase_metrics(self, trace, result):
        """Log custom metrics for phrase analysis"""
        metrics = {
            "sensational_phrase_count": len(result.get("sensational_phrases", [])),
            "emotional_trigger_count": len(result.get("emotional_triggers", [])),
            "caps_usage_count": result.get("caps_usage", {}).get("count", 0),
            "exclamation_count": result.get("exclamation_count", 0)
        }
        
        for metric_name, value in metrics.items():
            trace.log_feedback_score(name=metric_name, value=float(value))
```

#### B. Enhanced Evidence Aggregator with Comprehensive Tracing
```python
class TracedEvidenceAggregator:
    def __init__(self):
        self.opik_client = opik.Opik()
        self.weights = {
            "phrase_analysis": 0.25,
            "language_quality": 0.20,
            "commonsense": 0.30,
            "source_credibility": 0.25
        }
    
    @track(
        project_name="fake-news-detection",
        tags=["evidence-aggregation", "decision-making"]
    )
    def aggregate_evidence(self, evidence_data: dict, article_metadata: dict):
        """Aggregate evidence from all analysis agents with full traceability"""
        
        with self.opik_client.trace(
            name="evidence_aggregation_process",
            input={
                "article_id": article_metadata.get("id"),
                "evidence_sources": list(evidence_data.keys()),
                "aggregation_weights": self.weights
            },
            tags=["aggregation", "final-decision"]
        ) as main_trace:
            
            # Step 1: Validate and normalize evidence
            normalized_evidence = self.normalize_evidence(evidence_data, main_trace)
            
            # Step 2: Calculate weighted scores
            weighted_scores = self.calculate_weighted_scores(normalized_evidence, main_trace)
            
            # Step 3: Apply decision logic
            final_decision = self.make_final_decision(weighted_scores, main_trace)
            
            # Step 4: Generate explanation
            explanation = self.generate_explanation(evidence_data, final_decision, main_trace)
            
            # Log comprehensive output
            result = {
                "credibility_score": final_decision["credibility_score"],
                "risk_level": final_decision["risk_level"],
                "confidence": final_decision["confidence"],
                "detailed_analysis": explanation,
                "individual_scores": normalized_evidence,
                "weighted_scores": weighted_scores,
                "decision_factors": final_decision["factors"]
            }
            
            main_trace.log_output(result)
            
            # Log key metrics
            self.log_aggregation_metrics(main_trace, result)
            
            return result
    
    def log_aggregation_metrics(self, trace, result):
        """Log detailed metrics for the aggregation process"""
        trace.log_feedback_score(
            name="final_credibility_score", 
            value=result["credibility_score"]
        )
        trace.log_feedback_score(
            name="decision_confidence", 
            value=result["confidence"]
        )
        trace.log_feedback_score(
            name="risk_level_numeric", 
            value=self.risk_level_to_numeric(result["risk_level"])
        )
```

### 3. End-to-End Workflow Tracing

#### A. Main Detection Pipeline with Opik
```python
class TracedFakeNewsDetector:
    def __init__(self):
        self.opik_client = opik.Opik()
        self.phrase_agent = PhrasAnalysisAgent()
        self.language_agent = LanguageQualityAgent()
        self.commonsense_agent = CommonsenseAgent()
        self.credibility_agent = SourceCredibilityAgent()
        self.aggregator = TracedEvidenceAggregator()
    
    @track(
        project_name="fake-news-detection",
        tags=["end-to-end", "detection-pipeline"]
    )
    def detect_fake_news(self, article: dict):
        """Complete fake news detection pipeline with full traceability"""
        
        article_id = article.get("id", "unknown")
        
        with self.opik_client.trace(
            name="fake_news_detection_pipeline",
            input={
                "article_id": article_id,
                "title": article.get("title", "")[:100],
                "source_domain": article.get("source_domain"),
                "content_length": len(article.get("content", ""))
            },
            tags=["pipeline", "detection", f"article-{article_id}"]
        ) as pipeline_trace:
            
            results = {}
            
            # Run all agents in parallel with individual tracing
            with pipeline_trace.span(name="parallel_agent_execution") as agents_span:
                
                # Phrase Analysis
                with agents_span.span(name="phrase_analysis") as phrase_span:
                    results["phrase_analysis"] = self.phrase_agent.analyze_phrases(
                        article["content"], 
                        metadata=article
                    )
                    phrase_span.log_output(results["phrase_analysis"])
                
                # Language Quality Analysis  
                with agents_span.span(name="language_analysis") as lang_span:
                    results["language_quality"] = self.language_agent.analyze_language(
                        article["content"],
                        metadata=article
                    )
                    lang_span.log_output(results["language_quality"])
                
                # Commonsense Analysis
                with agents_span.span(name="commonsense_analysis") as cs_span:
                    results["commonsense"] = self.commonsense_agent.verify_claims(
                        article["content"],
                        metadata=article
                    )
                    cs_span.log_output(results["commonsense"])
                
                # Source Credibility Analysis
                with agents_span.span(name="credibility_analysis") as cred_span:
                    results["source_credibility"] = self.credibility_agent.assess_source(
                        article.get("url", ""),
                        article.get("source_domain", ""),
                        metadata=article
                    )
                    cred_span.log_output(results["source_credibility"])
            
            # Evidence Aggregation
            with pipeline_trace.span(name="evidence_aggregation") as agg_span:
                final_result = self.aggregator.aggregate_evidence(results, article)
                agg_span.log_output(final_result)
            
            # Log pipeline metrics
            self.log_pipeline_metrics(pipeline_trace, results, final_result)
            
            # Store results with trace ID for future reference
            final_result["trace_id"] = pipeline_trace.id
            final_result["individual_agent_results"] = results
            
            return final_result
    
    def log_pipeline_metrics(self, trace, agent_results, final_result):
        """Log comprehensive pipeline metrics"""
        
        # Agent performance metrics
        for agent_name, result in agent_results.items():
            if "confidence" in result:
                trace.log_feedback_score(
                    name=f"{agent_name}_confidence",
                    value=result["confidence"]
                )
        
        # Overall pipeline metrics
        trace.log_feedback_score(
            name="pipeline_success",
            value=1.0 if final_result.get("credibility_score") is not None else 0.0
        )
        
        trace.log_feedback_score(
            name="processing_time_seconds",
            value=trace.end_time - trace.start_time if trace.end_time else 0
        )
```

### 4. Dashboard Integration with Opik Analytics

#### A. Enhanced Monitoring Dashboard
```python
import streamlit as st
from opik import Opik
import plotly.express as px
import plotly.graph_objects as go

def create_enhanced_dashboard():
    st.title("Fake News Detection - Enhanced Dashboard with Opik Analytics")
    
    opik_client = Opik()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Detection", 
        "LLM Performance", 
        "Trace Analysis", 
        "Model Insights"
    ])
    
    with tab1:
        # Existing dashboard content
        display_live_detection_metrics()
    
    with tab2:
        st.header("LLM Performance Analytics")
        
        # Get traces from Opik
        traces = opik_client.get_traces(
            project_name="fake-news-detection",
            limit=100
        )
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_processing_time = calculate_avg_processing_time(traces)
            st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
        
        with col2:
            success_rate = calculate_success_rate(traces)
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        with col3:
            total_llm_calls = count_llm_calls(traces)
            st.metric("Total LLM Calls", total_llm_calls)
        
        # Agent performance comparison
        st.subheader("Agent Performance Comparison")
        agent_metrics = analyze_agent_performance(traces)
        
        fig = px.bar(
            agent_metrics, 
            x="agent", 
            y="avg_confidence",
            title="Average Confidence by Agent"
        )
        st.plotly_chart(fig)
        
        # Processing time trends
        st.subheader("Processing Time Trends")
        time_trends = extract_time_trends(traces)
        
        fig = px.line(
            time_trends, 
            x="timestamp", 
            y="processing_time",
            title="Processing Time Over Time"
        )
        st.plotly_chart(fig)
    
    with tab3:
        st.header("Detailed Trace Analysis")
        
        # Trace search and filter
        trace_id = st.text_input("Search by Trace ID")
        
        if trace_id:
            trace_details = opik_client.get_trace(trace_id)
            display_trace_details(trace_details)
        
        # Recent high-risk detections with traces
        st.subheader("Recent High-Risk Detections")
        high_risk_traces = get_high_risk_traces(opik_client)
        
        for trace in high_risk_traces:
            with st.expander(f"Trace: {trace.id} - Risk Level: {trace.output.get('risk_level')}"):
                display_trace_summary(trace)
    
    with tab4:
        st.header("Model Insights & Analytics")
        
        # Error analysis
        st.subheader("Error Analysis")
        error_traces = get_error_traces(opik_client)
        
        if error_traces:
            error_analysis = analyze_errors(error_traces)
            
            fig = px.pie(
                values=error_analysis["counts"],
                names=error_analysis["error_types"],
                title="Error Distribution"
            )
            st.plotly_chart(fig)
        
        # Confidence distribution
        st.subheader("Confidence Score Distribution")
        confidence_data = extract_confidence_scores(traces)
        
        fig = px.histogram(
            confidence_data,
            x="confidence_score",
            nbins=20,
            title="Distribution of Confidence Scores"
        )
        st.plotly_chart(fig)

def display_trace_details(trace):
    """Display detailed information about a specific trace"""
    st.json(trace.input)
    st.json(trace.output)
    
    # Display spans
    if hasattr(trace, 'spans'):
        st.subheader("Execution Spans")
        for span in trace.spans:
            with st.expander(f"Span: {span.name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Input:**")
                    st.json(span.input)
                with col2:
                    st.write("**Output:**")
                    st.json(span.output)
```

### 5. Human Feedback Integration with Opik

#### A. Enhanced Review Interface with Feedback Logging
```python
def create_enhanced_review_interface():
    st.title("Fake News Detection - Manual Review with Feedback")
    
    opik_client = Opik()
    
    # Load articles with trace information
    articles = load_pending_articles_with_traces()
    
    for idx, article in articles.iterrows():
        with st.expander(f"Article {article['id']}: {article['title'][:100]}..."):
            
            # Display article content
            display_article_content(article)
            
            # Display AI analysis with trace link
            st.subheader("AI Analysis")
            if article.get('trace_id'):
                st.write(f"**Trace ID:** `{article['trace_id']}`")
                
                # Get detailed trace information
                trace = opik_client.get_trace(article['trace_id'])
                
                # Display agent results
                for agent_name, result in article['agent_results'].items():
                    with st.expander(f"{agent_name.title()} Analysis"):
                        st.write(f"**Score:** {result.get('score', 'N/A')}")
                        st.write(f"**Confidence:** {result.get('confidence', 'N/A')}")
                        st.write(f"**Reasoning:** {result.get('explanation', 'N/A')}")
            
            # Human review form
            st.subheader("Your Assessment")
            
            verdict = st.radio(
                "Verdict:", 
                ["Real", "Fake", "Uncertain"], 
                key=f"verdict_{article['id']}"
            )
            
            confidence = st.slider(
                "Confidence:", 1, 5, 3, 
                key=f"confidence_{article['id']}"
            )
            
            notes = st.text_area(
                "Notes:", 
                key=f"notes_{article['id']}"
            )
            
            # Agreement with AI assessment
            ai_verdict = article.get('risk_level', 'Unknown')
            agreement = st.radio(
                f"Do you agree with AI assessment ({ai_verdict})?",
                ["Agree", "Partially Agree", "Disagree"],
                key=f"agreement_{article['id']}"
            )
            
            if st.button(f"Submit Review", key=f"submit_{article['id']}"):
                # Save review to database
                save_review(article['id'], verdict, confidence, notes)
                
                # Log feedback to Opik
                if article.get('trace_id'):
                    log_human_feedback_to_opik(
                        opik_client,
                        article['trace_id'],
                        {
                            "human_verdict": verdict.lower(),
                            "human_confidence": confidence / 5.0,
                            "agreement_with_ai": agreement.lower(),
                            "notes": notes,
                            "reviewer_id": "manual_reviewer"  # Could be actual user ID
                        }
                    )
                
                st.success("Review submitted and feedback logged!")

def log_human_feedback_to_opik(opik_client, trace_id, feedback_data):
    """Log human feedback to Opik for model improvement"""
    
    # Log feedback scores
    opik_client.log_feedback_score(
        trace_id=trace_id,
        name="human_agreement",
        value=1.0 if feedback_data["agreement_with_ai"] == "agree" else 
              0.5 if feedback_data["agreement_with_ai"] == "partially agree" else 0.0
    )
    
    opik_client.log_feedback_score(
        trace_id=trace_id,
        name="human_confidence",
        value=feedback_data["human_confidence"]
    )
    
    # Log categorical feedback
    opik_client.log_feedback_score(
        trace_id=trace_id,
        name="human_verdict_match",
        value=1.0 if feedback_data["human_verdict"] in ["real", "fake"] else 0.5
    )
```

### 6. Performance Monitoring and Alerts

#### A. Automated Performance Monitoring
```python
import schedule
import time
from datetime import datetime, timedelta

class OpikPerformanceMonitor:
    def __init__(self):
        self.opik_client = Opik()
        self.alert_thresholds = {
            "success_rate": 0.95,
            "avg_processing_time": 10.0,  # seconds
            "error_rate": 0.05
        }
    
    def monitor_performance(self):
        """Monitor system performance and send alerts if needed"""
        
        # Get recent traces (last hour)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        traces = self.opik_client.get_traces(
            project_name="fake-news-detection",
            start_time=start_time,
            end_time=end_time
        )
        
        if not traces:
            return
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(traces)
        
        # Check thresholds and send alerts
        self.check_and_alert(metrics)
        
        # Log monitoring data
        self.log_monitoring_data(metrics)
    
    def calculate_performance_metrics(self, traces):
        """Calculate key performance metrics from traces"""
        
        total_traces = len(traces)
        successful_traces = sum(1 for t in traces if not t.output.get("error"))
        
        processing_times = [
            (t.end_time - t.start_time).total_seconds() 
            for t in traces if t.end_time and t.start_time
        ]
        
        return {
            "total_traces": total_traces,
            "success_rate": successful_traces / total_traces if total_traces > 0 else 0,
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "error_rate": (total_traces - successful_traces) / total_traces if total_traces > 0 else 0,
            "timestamp": datetime.now()
        }
    
    def check_and_alert(self, metrics):
        """Check metrics against thresholds and send alerts"""
        
        alerts = []
        
        if metrics["success_rate"] < self.alert_thresholds["success_rate"]:
            alerts.append(f"Low success rate: {metrics['success_rate']:.2%}")
        
        if metrics["avg_processing_time"] > self.alert_thresholds["avg_processing_time"]:
            alerts.append(f"High processing time: {metrics['avg_processing_time']:.2f}s")
        
        if metrics["error_rate"] > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics['error_rate']:.2%}")
        
        if alerts:
            self.send_alerts(alerts, metrics)
    
    def send_alerts(self, alerts, metrics):
        """Send performance alerts"""
        
        alert_message = f"""
        FAKE NEWS DETECTION SYSTEM ALERT
        
        Time: {metrics['timestamp']}
        
        Issues detected:
        {chr(10).join(f"- {alert}" for alert in alerts)}
        
        Current metrics:
        - Success rate: {metrics['success_rate']:.2%}
        - Avg processing time: {metrics['avg_processing_time']:.2f}s
        - Error rate: {metrics['error_rate']:.2%}
        - Total traces: {metrics['total_traces']}
        """
        
        # Send to monitoring system (Slack, email, etc.)
        print(alert_message)  # Replace with actual alerting mechanism

# Schedule monitoring
monitor = OpikPerformanceMonitor()
schedule.every(15).minutes.do(monitor.monitor_performance)

# Run monitoring loop
def run_monitoring():
    while True:
        schedule.run_pending()
        time.sleep(60)
```

## Deployment Configuration

### 1. Docker Setup
### 1. Updated Docker Setup with Opik
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Set environment variables for Opik
ENV OPIK_API_KEY=${OPIK_API_KEY}
ENV OPIK_WORKSPACE=${OPIK_WORKSPACE}

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Updated Requirements with Opik
```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.28.0
langchain==0.0.340
openai==1.3.0
opik==0.2.0
opik[langchain]==0.2.0
psycopg2-binary==2.9.7
redis==5.0.1
pandas==2.1.3
plotly==5.17.0
feedparser==6.0.10
requests==2.31.0
python-multipart==0.0.6
schedule==1.2.0
```

### 3. Environment Configuration with Opik
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/fakenews
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPIK_API_KEY=${OPIK_API_KEY}
      - OPIK_WORKSPACE=fake-news-detection
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  streamlit:
    build: .
    command: streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/fakenews
      - OPIK_API_KEY=${OPIK_API_KEY}
      - OPIK_WORKSPACE=fake-news-detection
    depends_on:
      - db
      - redis
  
  monitoring:
    build: .
    command: python monitoring.py
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/fakenews
      - OPIK_API_KEY=${OPIK_API_KEY}
      - OPIK_WORKSPACE=fake-news-detection
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=fakenews
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:alpine
    
volumes:
  postgres_data:
```

### 4. Updated Database Schema with Trace Integration
```sql
-- Enhanced articles table with trace information
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    source_domain VARCHAR(255),
    published_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    credibility_score FLOAT,
    risk_level VARCHAR(20),
    review_status VARCHAR(20) DEFAULT 'pending',
    
    -- Opik trace integration
    trace_id VARCHAR(255),
    trace_url TEXT,
    processing_time_seconds FLOAT,
    llm_calls_count INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0
);

-- Enhanced analysis results with detailed tracing
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    agent_name VARCHAR(50),
    score FLOAT,
    confidence FLOAT,
    explanation TEXT,
    evidence JSONB,
    
    -- Tracing information
    span_id VARCHAR(255),
    processing_time_ms INTEGER,
    tokens_used INTEGER,
    model_used VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Human reviews enhanced with feedback tracking
CREATE TABLE human_reviews (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id),
    reviewer_id VARCHAR(100),
    verdict VARCHAR(20), -- 'real', 'fake', 'uncertain'
    confidence INTEGER, -- 1-5
    notes TEXT,
    
    -- AI agreement tracking
    agreement_with_ai VARCHAR(20), -- 'agree', 'partially_agree', 'disagree'
    trace_id VARCHAR(255),
    feedback_logged BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance monitoring table
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- System metrics
    total_articles_processed INTEGER,
    success_rate FLOAT,
    avg_processing_time FLOAT,
    error_rate FLOAT,
    
    -- LLM metrics
    total_llm_calls INTEGER,
    total_tokens_used INTEGER,
    avg_tokens_per_call FLOAT,
    
    -- Agent performance
    agent_metrics JSONB,
    
    -- Trace references
    sample_trace_ids TEXT[]
);

-- Error tracking table
CREATE TABLE error_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    
    -- Error details
    error_type VARCHAR(100),
    error_message TEXT,
    stack_trace TEXT,
    
    -- Context
    article_id INTEGER,
    agent_name VARCHAR(50),
    trace_id VARCHAR(255),
    
    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT
);
```

### 5. Complete Langflow Configuration with Opik Integration

```python
# langflow_opik_config.py
from langflow import LangFlow
from opik.integrations.langchain import OpikLangChainTracer
import opik

class OpikLangflowIntegration:
    def __init__(self):
        self.opik_client = opik.Opik()
        self.tracer = OpikLangChainTracer(
            project_name="fake-news-detection-langflow",
            tags=["langflow", "production"]
        )
    
    def create_traced_flow(self):
        """Create Langflow configuration with Opik tracing"""
        
        flow_config = {
            "flow_name": "fake_news_detection_with_tracing",
            "description": "Complete fake news detection pipeline with Opik tracing",
            "nodes": [
                {
                    "id": "input_handler",
                    "type": "TextInput",
                    "name": "Article Input",
                    "config": {
                        "multiline": True,
                        "placeholder": "Enter article content or URL"
                    },
                    "opik_config": {
                        "trace_inputs": True,
                        "log_metadata": {"node_type": "input"}
                    }
                },
                {
                    "id": "content_extractor",
                    "type": "PythonFunction",
                    "name": "Content Extractor",
                    "config": {
                        "function": "extract_article_content",
                        "code": """
def extract_article_content(input_data):
    import opik
    from opik.decorators import track
    
    @track(name="content_extraction", tags=["preprocessing"])
    def extract_content(data):
        # Extract title, content, URL, metadata
        if data.startswith('http'):
            # URL provided - extract content
            return extract_from_url(data)
        else:
            # Direct text provided
            return {
                'title': data[:100] + '...',
                'content': data,
                'url': None,
                'source_domain': 'manual_input'
            }
    
    return extract_content(input_data)
"""
                    }
                },
                {
                    "id": "phrase_analyzer",
                    "type": "LLMChain",
                    "name": "Phrase Analysis Agent",
                    "config": {
                        "llm": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0,
                            "callbacks": [self.tracer]
                        },
                        "prompt": """
                        Analyze the following news article for sensational language patterns.
                        
                        Article: {content}
                        
                        Look for:
                        1. Clickbait phrases
                        2. Emotional manipulation
                        3. Excessive superlatives
                        4. All-caps usage
                        5. Multiple exclamation marks
                        
                        Respond in JSON format:
                        {{
                            "sensational_phrases": ["phrase1", "phrase2"],
                            "emotional_triggers": ["trigger1", "trigger2"],
                            "severity_score": 0.0,
                            "confidence": 0.0,
                            "explanation": "detailed reasoning"
                        }}
                        """,
                        "output_parser": "json"
                    },
                    "opik_config": {
                        "tags": ["phrase-analysis", "llm-agent"],
                        "log_tokens": True,
                        "log_latency": True
                    }
                },
                {
                    "id": "language_analyzer",
                    "type": "LLMChain",
                    "name": "Language Quality Agent",
                    "config": {
                        "llm": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0,
                            "callbacks": [self.tracer]
                        },
                        "prompt": """
                        Evaluate the language quality of this news article:
                        
                        Text: {content}
                        
                        Analyze for:
                        1. Grammar errors
                        2. Spelling mistakes
                        3. Writing style consistency
                        4. Professional language use
                        
                        Respond in JSON format:
                        {{
                            "grammar_score": 0.0,
                            "spelling_score": 0.0,
                            "style_score": 0.0,
                            "overall_quality": 0.0,
                            "issues_found": ["issue1", "issue2"],
                            "explanation": "quality assessment"
                        }}
                        """,
                        "output_parser": "json"
                    },
                    "opik_config": {
                        "tags": ["language-analysis", "llm-agent"],
                        "log_tokens": True,
                        "log_latency": True
                    }
                },
                {
                    "id": "commonsense_analyzer",
                    "type": "LLMChain",
                    "name": "Commonsense Verification Agent",
                    "config": {
                        "llm": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0,
                            "callbacks": [self.tracer]
                        },
                        "prompt": """
                        Evaluate this news claim for common sense and logical consistency:
                        
                        Claim: {content}
                        
                        Consider:
                        1. Factual plausibility
                        2. Logical consistency
                        3. Contradiction with known facts
                        4. Similarity to gossip vs. news
                        
                        Respond in JSON format:
                        {{
                            "plausibility": "High/Medium/Low",
                            "logical_consistency": 0.0,
                            "factual_concerns": ["concern1", "concern2"],
                            "confidence": 0.0,
                            "reasoning": "detailed analysis"
                        }}
                        """,
                        "output_parser": "json"
                    },
                    "opik_config": {
                        "tags": ["commonsense-analysis", "llm-agent"],
                        "log_tokens": True,
                        "log_latency": True
                    }
                },
                {
                    "id": "source_credibility_analyzer",
                    "type": "PythonFunction",
                    "name": "Source Credibility Agent",
                    "config": {
                        "function": "analyze_source_credibility",
                        "code": """
import opik
from opik.decorators import track

@track(name="source_credibility_analysis", tags=["credibility", "source-check"])
def analyze_source_credibility(article_data):
    # Domain credibility database lookup
    domain_scores = {
        'bbc.com': 0.95, 'reuters.com': 0.95, 'cnn.com': 0.85,
        'foxnews.com': 0.75, 'breitbart.com': 0.30, 'infowars.com': 0.15
    }
    
    domain = article_data.get('source_domain', '').lower()
    base_score = domain_scores.get(domain, 0.5)  # Default neutral score
    
    # Additional checks
    url = article_data.get('url', '')
    
    credibility_factors = {
        'domain_reputation': base_score,
        'https_usage': 0.1 if url.startswith('https://') else -0.1,
        'domain_age': 0.0,  # Would need external service
        'professional_appearance': 0.0  # Would need website analysis
    }
    
    final_score = max(0, min(1, sum(credibility_factors.values())))
    
    return {
        'credibility_score': final_score,
        'domain_reputation': base_score,
        'credibility_factors': credibility_factors,
        'explanation': f"Domain {domain} has reputation score {base_score}"
    }
"""
                    },
                    "opik_config": {
                        "tags": ["credibility-analysis", "domain-check"],
                        "log_metadata": {"analysis_type": "source_credibility"}
                    }
                },
                {
                    "id": "evidence_aggregator",
                    "type": "PythonFunction",
                    "name": "Evidence Aggregation Engine",
                    "config": {
                        "function": "aggregate_evidence_with_tracing",
                        "code": """
import opik
from opik.decorators import track
import json

@track(name="evidence_aggregation", tags=["aggregation", "decision-making"])
def aggregate_evidence_with_tracing(phrase_result, language_result, commonsense_result, credibility_result):
    
    # Parse JSON results
    try:
        phrase_data = json.loads(phrase_result) if isinstance(phrase_result, str) else phrase_result
        language_data = json.loads(language_result) if isinstance(language_result, str) else language_result
        commonsense_data = json.loads(commonsense_result) if isinstance(commonsense_result, str) else commonsense_result
        credibility_data = credibility_result if isinstance(credibility_result, dict) else json.loads(credibility_result)
    except json.JSONDecodeError as e:
        return {'error': f'Failed to parse agent results: {str(e)}'}
    
    # Weights for different factors
    weights = {
        'phrase_analysis': 0.25,
        'language_quality': 0.20,
        'commonsense': 0.30,
        'source_credibility': 0.25
    }
    
    # Extract scores (normalize to 0-1 scale where 1 = suspicious)
    scores = {
        'phrase_analysis': phrase_data.get('severity_score', 0) / 5.0,
        'language_quality': 1 - (language_data.get('overall_quality', 0.5)),
        'commonsense': 1 - (1 if commonsense_data.get('plausibility') == 'High' else 0.5 if commonsense_data.get('plausibility') == 'Medium' else 0),
        'source_credibility': 1 - credibility_data.get('credibility_score', 0.5)
    }
    
    # Calculate weighted risk score
    risk_score = sum(scores[factor] * weights[factor] for factor in weights.keys())
    credibility_score = 1 - risk_score  # Invert for credibility
    
    # Determine risk level
    if credibility_score >= 0.8:
        risk_level = 'LOW_RISK'
    elif credibility_score >= 0.6:
        risk_level = 'MEDIUM_RISK'
    elif credibility_score >= 0.4:
        risk_level = 'HIGH_RISK'
    else:
        risk_level = 'VERY_HIGH_RISK'
    
    # Calculate overall confidence
    confidences = [
        phrase_data.get('confidence', 0.5),
        language_data.get('confidence', 0.5) if 'confidence' in language_data else 0.5,
        commonsense_data.get('confidence', 0.5),
        credibility_data.get('confidence', 0.8)  # Domain scores are generally reliable
    ]
    overall_confidence = sum(confidences) / len(confidences)
    
    result = {
        'credibility_score': round(credibility_score, 3),
        'risk_level': risk_level,
        'confidence': round(overall_confidence, 3),
        'individual_scores': scores,
        'weighted_scores': {k: round(scores[k] * weights[k], 3) for k in weights.keys()},
        'detailed_analysis': {
            'phrase_analysis': phrase_data,
            'language_quality': language_data,
            'commonsense_verification': commonsense_data,
            'source_credibility': credibility_data
        },
        'explanation': generate_explanation(scores, risk_level, credibility_score)
    }
    
    # Log key metrics to current trace
    opik.log_feedback_score('final_credibility_score', credibility_score)
    opik.log_feedback_score('overall_confidence', overall_confidence)
    opik.log_feedback_score('risk_level_numeric', {'LOW_RISK': 0.25, 'MEDIUM_RISK': 0.5, 'HIGH_RISK': 0.75, 'VERY_HIGH_RISK': 1.0}[risk_level])
    
    return result

def generate_explanation(scores, risk_level, credibility_score):
    explanations = []
    
    if scores['phrase_analysis'] > 0.6:
        explanations.append("High use of sensational language detected")
    if scores['language_quality'] > 0.6:
        explanations.append("Poor language quality indicators found")
    if scores['commonsense'] > 0.6:
        explanations.append("Claims contradict common sense or known facts")
    if scores['source_credibility'] > 0.6:
        explanations.append("Source has low credibility rating")
    
    if not explanations:
        explanations.append("No significant risk indicators detected")
    
    return f"Risk Level: {risk_level} (Score: {credibility_score:.2f}). " + "; ".join(explanations)
"""
                    },
                    "opik_config": {
                        "tags": ["aggregation", "final-decision"],
                        "log_metadata": {"stage": "final_decision"}
                    }
                },
                {
                    "id": "result_formatter",
                    "type": "PythonFunction",
                    "name": "Result Formatter & Database Writer",
                    "config": {
                        "function": "format_and_store_result",
                        "code": """
import opik
from opik.decorators import track
import json
from datetime import datetime

@track(name="result_formatting", tags=["output", "database"])
def format_and_store_result(aggregated_result, original_article):
    
    # Get current trace ID for storage
    current_trace = opik.get_current_trace()
    trace_id = current_trace.id if current_trace else None
    
    # Parse aggregated result if it's a string
    if isinstance(aggregated_result, str):
        try:
            result_data = json.loads(aggregated_result)
        except json.JSONDecodeError:
            result_data = {'error': 'Failed to parse aggregation result'}
    else:
        result_data = aggregated_result
    
    # Format final output
    formatted_result = {
        'article': {
            'title': original_article.get('title', ''),
            'content_preview': original_article.get('content', '')[:200] + '...',
            'source_domain': original_article.get('source_domain', ''),
            'url': original_article.get('url', '')
        },
        'detection_result': {
            'credibility_score': result_data.get('credibility_score', 0),
            'risk_level': result_data.get('risk_level', 'UNKNOWN'),
            'confidence': result_data.get('confidence', 0),
            'explanation': result_data.get('explanation', 'No explanation available')
        },
        'detailed_analysis': result_data.get('detailed_analysis', {}),
        'trace_information': {
            'trace_id': trace_id,
            'timestamp': datetime.now().isoformat(),
            'processing_pipeline': 'langflow_v1'
        },
        'metadata': {
            'processed_at': datetime.now().isoformat(),
            'system_version': '1.0.0',
            'model_versions': {
                'phrase_analyzer': 'gpt-3.5-turbo',
                'language_analyzer': 'gpt-3.5-turbo',
                'commonsense_analyzer': 'gpt-3.5-turbo'
            }
        }
    }
    
    # Store in database (would implement actual database connection)
    # store_detection_result(formatted_result)
    
    return formatted_result
"""
                    },
                    "opik_config": {
                        "tags": ["output", "formatting"],
                        "log_outputs": True
                    }
                }
            ],
            "connections": [
                {"from": "input_handler", "to": "content_extractor"},
                {"from": "content_extractor", "to": "phrase_analyzer"},
                {"from": "content_extractor", "to": "language_analyzer"},
                {"from": "content_extractor", "to": "commonsense_analyzer"},
                {"from": "content_extractor", "to": "source_credibility_analyzer"},
                {"from": "phrase_analyzer", "to": "evidence_aggregator"},
                {"from": "language_analyzer", "to": "evidence_aggregator"},
                {"from": "commonsense_analyzer", "to": "evidence_aggregator"},
                {"from": "source_credibility_analyzer", "to": "evidence_aggregator"},
                {"from": "evidence_aggregator", "to": "result_formatter"},
                {"from": "content_extractor", "to": "result_formatter"}
            ],
            "global_config": {
                "opik_integration": {
                    "project_name": "fake-news-detection-langflow",
                    "auto_trace": True,
                    "log_inputs": True,
                    "log_outputs": True,
                    "log_metadata": True
                }
            }
        }
        
        return flow_config

# Usage example
def deploy_traced_langflow():
    """Deploy Langflow with Opik tracing"""
    
    integration = OpikLangflowIntegration()
    flow_config = integration.create_traced_flow()
    
    # Save configuration
    with open('fake_news_detection_flow.json', 'w') as f:
        json.dump(flow_config, f, indent=2)
    
    print("Langflow configuration with Opik tracing saved to fake_news_detection_flow.json")
    print("Load this configuration in Langflow UI or via API")
    
    return flow_config
```

### 6. Testing and Validation with Opik

#### A. Comprehensive Test Suite
```python
# test_with_opik.py
import pytest
import opik
from opik.decorators import track
import json

class TestFakeNewsDetectionWithTracing:
    def __init__(self):
        self.opik_client = opik.Opik()
        self.test_project = "fake-news-detection-testing"
    
    @track(project_name="fake-news-detection-testing", tags=["testing", "validation"])
    def test_phrase_analysis_agent(self):
        """Test phrase analysis with various input types"""
        
        test_cases = [
            {
                "input": "SHOCKING: You Won't Believe What Happened Next!!!",
                "expected_risk": "high",
                "test_name": "sensational_clickbait"
            },
            {
                "input": "Federal Reserve announces interest rate decision following economic data review.",
                "expected_risk": "low", 
                "test_name": "neutral_news"
            },
            {
                "input": "URGENT!!! BREAKING!!! Celebrity CAUGHT in SCANDAL!!!",
                "expected_risk": "very_high",
                "test_name": "excessive_sensationalism"
            }
        ]
        
        results = []
        
        for case in test_cases:
            with self.opik_client.trace(
                name=f"test_phrase_analysis_{case['test_name']}",
                input=case,
                tags=["test", "phrase-analysis"]
            ) as trace:
                
                # Run phrase analysis
                agent = PhrasAnalysisAgent()
                result = agent.analyze_phrases(case["input"])
                
                # Validate result
                assert "severity_score" in result
                assert "confidence" in result
                assert 0 <= result["severity_score"] <= 5
                assert 0 <= result["confidence"] <= 1
                
                # Check expected risk level
                severity = result["severity_score"]
                if case["expected_risk"] == "high" and severity < 3:
                    pytest.fail(f"Expected high risk but got severity {severity}")
                elif case["expected_risk"] == "low" and severity > 2:
                    pytest.fail(f"Expected low risk but got severity {severity}")
                
                # Log test results
                trace.log_output(result)
                trace.log_feedback_score("test_passed", 1.0)
                
                results.append({
                    "test_case": case["test_name"],
                    "result": result,
                    "trace_id": trace.id
                })
        
        return results
    
    @track(project_name="fake-news-detection-testing", tags=["integration-test"])
    def test_end_to_end_pipeline(self):
        """Test complete pipeline with known fake and real news examples"""
        
        test_articles = [
            {
                "title": "Local Scientists Discover Aliens Living Among Us",
                "content": "SHOCKING revelation!!! You WON'T BELIEVE what researchers found...",
                "source_domain": "fakenews-site.com",
                "expected_outcome": "fake"
            },
            {
                "title": "Federal Budget Proposal Includes Infrastructure Investment",
                "content": "The administration's budget proposal includes significant infrastructure investments...",
                "source_domain": "reuters.com",
                "expected_outcome": "real"
            }
        ]
        
        detector = TracedFakeNewsDetector()
        test_results = []
        
        for article in test_articles:
            with self.opik_client.trace(
                name=f"e2e_test_{article['expected_outcome']}",
                input=article,
                tags=["e2e-test", "integration"]
            ) as trace:
                
                result = detector.detect_fake_news(article)
                
                # Validate structure
                assert "credibility_score" in result
                assert "risk_level" in result
                assert "trace_id" in result
                
                # Check expected outcome
                risk_level = result["risk_level"]
                if article["expected_outcome"] == "fake":
                    assert risk_level in ["HIGH_RISK", "VERY_HIGH_RISK"]
                else:
                    assert risk_level in ["LOW_RISK", "MEDIUM_RISK"]
                
                trace.log_output(result)
                trace.log_feedback_score("test_accuracy", 1.0)
                
                test_results.append({
                    "article": article,
                    "result": result,
                    "test_passed": True
                })
        
        return test_results
    
    def run_performance_tests(self):
        """Run performance tests and log metrics to Opik"""
        
        import time
        
        with self.opik_client.trace(
            name="performance_test_suite",
            tags=["performance", "benchmarking"]
        ) as trace:
            
            # Test processing speed
            start_time = time.time()
            
            sample_article = {
                "title": "Sample news article for performance testing",
                "content": "This is a sample article used for performance testing. " * 50,
                "source_domain": "test.com"
            }
            
            detector = TracedFakeNewsDetector()
            
            # Run multiple iterations
            iterations = 10
            processing_times = []
            
            for i in range(iterations):
                iteration_start = time.time()
                result = detector.detect_fake_news(sample_article)
                iteration_end = time.time()
                
                processing_times.append(iteration_end - iteration_start)
            
            end_time = time.time()
            
            # Calculate metrics
            avg_processing_time = sum(processing_times) / len(processing_times)
            total_time = end_time - start_time
            
            performance_metrics = {
                "total_iterations": iterations,
                "total_time_seconds": total_time,
                "avg_processing_time": avg_processing_time,
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "throughput_per_second": iterations / total_time
            }
            
            trace.log_output(performance_metrics)
            
            # Log individual metrics
            for metric_name, value in performance_metrics.items():
                trace.log_feedback_score(metric_name, float(value))
            
            return performance_metrics

# Run tests
def run_all_tests():
    """Execute all tests with Opik tracing"""
    
    tester = TestFakeNewsDetectionWithTracing()
    
    print("Running phrase analysis tests...")
    phrase_results = tester.test_phrase_analysis_agent()
    
    print("Running end-to-end pipeline tests...")
    e2e_results = tester.test_end_to_end_pipeline()
    
    print("Running performance tests...")
    perf_results = tester.run_performance_tests()
    
    print("\n=== Test Summary ===")
    print(f"Phrase analysis tests: {len(phrase_results)} passed")
    print(f"E2E tests: {len(e2e_results)} passed")
    print(f"Performance metrics: {perf_results}")
    print("\nAll test traces are available in Opik dashboard")

if __name__ == "__main__":
    run_all_tests()
```

This completes Phase 1 implementation with comprehensive Opik integration providing:

## **Key Opik Benefits Added:**

1. **Complete Traceability**: Every LLM call, processing step, and decision is traced
2. **Performance Monitoring**: Real-time metrics on processing times, token usage, and success rates
3. **Error Tracking**: Detailed error logs with context for debugging
4. **Human Feedback Loop**: Integration of manual reviews with model performance data
5. **Analytics Dashboard**: Rich insights into model behavior and performance trends

## **Getting Started:**

1. **Set up Opik account** and get API keys
2. **Deploy with Docker Compose** using the provided configuration
3. **Load Langflow configuration** from the JSON file
4. **Run test suite** to validate everything works
5. **Access dashboards** for monitoring and manual review

The system now provides production-ready observability while maintaining the core fake news detection capabilities. You can monitor model performance, track user feedback, and continuously improve the system based on real usage data.

Would you like me to detail any specific aspect further, such as the Opik dashboard configuration or advanced analytics queries? requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Environment Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/fakenews
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=fakenews
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:alpine
    
volumes:
  postgres_data:
```

### 3. Langflow Configuration
```python
# langflow_config.py
LANGFLOW_CONFIG = {
    "flows": {
        "fake_news_detection": {
            "nodes": [
                {
                    "id": "input_handler",
                    "type": "TextInput",
                    "config": {"multiline": True}
                },
                {
                    "id": "phrase_analyzer",
                    "type": "LLMChain",
                    "config": {
                        "prompt": PHRASE_ANALYSIS_PROMPT,
                        "llm": "gpt-3.5-turbo"
                    }
                },
                {
                    "id": "language_analyzer", 
                    "type": "LLMChain",
                    "config": {
                        "prompt": LANGUAGE_QUALITY_PROMPT,
                        "llm": "gpt-3.5-turbo"
                    }
                },
                {
                    "id": "commonsense_analyzer",
                    "type": "LLMChain", 
                    "config": {
                        "prompt": COMMONSENSE_PROMPT,
                        "llm": "gpt-3.5-turbo"
                    }
                },
                {
                    "id": "evidence_aggregator",
                    "type": "PythonFunction",
                    "config": {"function": "aggregate_evidence"}
                }
            ],
            "connections": [
                {"from": "input_handler", "to": "phrase_analyzer"},
                {"from": "input_handler", "to": "language_analyzer"},
                {"from": "input_handler", "to": "commonsense_analyzer"},
                {"from": "phrase_analyzer", "to": "evidence_aggregator"},
                {"from": "language_analyzer", "to": "evidence_aggregator"},
                {"from": "commonsense_analyzer", "to": "evidence_aggregator"}
            ]
        }
    }
}
```

## Testing and Validation

### 1. Test Dataset Preparation
```python
# Create test cases covering different scenarios
TEST_CASES = [
    {
        "title": "Local Man Discovers Shocking Truth About His Neighbor!!!",
        "content": "YOU WON'T BELIEVE what happened next...",
        "expected_risk": "VERY_HIGH_RISK"
    },
    {
        "title": "Federal Reserve Announces Interest Rate Decision",
        "content": "The Federal Reserve announced today...",
        "expected_risk": "LOW_RISK"
    }
    # Add more test cases...
]
```

### 2. Performance Monitoring
```python
# Basic performance tracking
def track_performance():
    metrics = {
        "processing_time": measure_average_processing_time(),
        "accuracy": calculate_accuracy_against_manual_reviews(),
        "false_positive_rate": calculate_false_positive_rate(),
        "system_uptime": get_system_uptime()
    }
    return metrics
```

This Phase 1 implementation provides a solid foundation for fake news detection that can be built incrementally. The modular design allows for easy expansion and improvement in subsequent phases.
