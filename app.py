"""
Multi-Agent Orchestration Playground - Streamlit App

Interactive playground to explore and experiment with 5 different 
multi-agent orchestration patterns for AI systems.
"""
import streamlit as st
import asyncio
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import orchestration patterns
from patterns.sequential_orchestration import (
    demo_sequential_orchestration, 
    ReportGenerationPipeline
)
from patterns.mapreduce_orchestration import (
    demo_mapreduce_orchestration,
    DocumentSummarizationPipeline,
    SentimentAnalysisPipeline
)
from patterns.consensus_orchestration import (
    demo_consensus_orchestration,
    SentimentConsensusSystem,
    DecisionMakingConsensus
)
from patterns.hierarchical_orchestration import (
    demo_hierarchical_orchestration,
    TravelPlanningSystem,
    ProjectManagementSystem
)
from patterns.producer_reviewer_orchestration import (
    demo_producer_reviewer_orchestration,
    DocumentSummarizationSystem,
    ContentCreationSystem,
    CodeReviewSystem
)

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Orchestration Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .pattern-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'execution_results' not in st.session_state:
        st.session_state.execution_results = {}
    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = []

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    return api_key.startswith('sk-') and len(api_key) > 20

def display_pattern_overview():
    """Display overview of all orchestration patterns."""
    st.markdown('<h1 class="main-header">🤖 Multi-Agent Orchestration Playground</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore 5 powerful orchestration patterns for multi-agent AI systems. Each pattern solves different types of 
    complex problems through intelligent agent coordination and collaboration.
    """)
    
    # Pattern overview cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="pattern-card">
            <h3>🔄 Sequential Orchestration</h3>
            <p><strong>Pipeline Processing</strong></p>
            <p>Agents work in sequence, each building on the previous agent's output. Perfect for step-by-step processes like report generation.</p>
            <p><em>Example: Data Collection → Formatting → Analysis → Optimization → Delivery</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pattern-card">
            <h3>🗳️ Consensus Orchestration</h3>
            <p><strong>Redundant Verification</strong></p>
            <p>Multiple agents independently solve the same problem, then vote on the best solution. Improves reliability through "wisdom of crowds".</p>
            <p><em>Example: Multiple sentiment analyzers voting on final sentiment</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pattern-card">
            <h3>🔄 Producer-Reviewer</h3>
            <p><strong>Iterative Quality Assurance</strong></p>
            <p>Producer creates content, reviewer provides feedback, and the cycle continues until quality standards are met.</p>
            <p><em>Example: Document creation with legal review and revision cycles</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pattern-card">
            <h3>⚡ MapReduce Orchestration</h3>
            <p><strong>Parallel Computing Intelligence</strong></p>
            <p>Breaks large tasks into independent chunks, processes them in parallel, then combines results. Excellent for scalability.</p>
            <p><em>Example: Document summarization with parallel chunk processing</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="pattern-card">
            <h3>🏗️ Hierarchical Orchestration</h3>
            <p><strong>Specialized Division of Labor</strong></p>
            <p>Master orchestrator decomposes complex tasks and coordinates specialized agents. Handles cross-domain problems effectively.</p>
            <p><em>Example: Travel planning with flight, hotel, and activity specialists</em></p>
        </div>
        """, unsafe_allow_html=True)

def display_api_key_input():
    """Display API key input section."""
    st.sidebar.markdown("## 🔑 Configuration")
    
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        help="Enter your OpenAI API key to enable the playground"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    if api_key and not validate_api_key(api_key):
        st.sidebar.error("⚠️ Invalid API key format. Please check your key.")
        return False
    elif api_key:
        st.sidebar.success("✅ API key configured")
        return True
    else:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key to continue")
        return False

def display_execution_metrics(results: Dict[str, Any]):
    """Display execution metrics in a nice format."""
    if not results:
        return
    
    st.markdown("### 📊 Execution Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Extract metrics based on pattern type
    if 'total_execution_time' in results:
        execution_time = results['total_execution_time']
    elif 'execution_summary' in results:
        execution_time = results['execution_summary'].get('total_execution_time', 0)
    elif 'process_summary' in results:
        execution_time = results['process_summary'].get('total_execution_time', 0)
    else:
        execution_time = 0
    
    if 'total_tokens' in results:
        total_tokens = results['total_tokens'].get('total_tokens', 0)
    elif 'execution_summary' in results:
        total_tokens = results['execution_summary'].get('total_tokens', {}).get('total_tokens', 0)
    elif 'process_summary' in results:
        total_tokens = results['process_summary'].get('total_tokens', {}).get('total_tokens', 0)
    else:
        total_tokens = 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{execution_time:.2f}s</h3>
            <p>Execution Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_tokens:,}</h3>
            <p>Total Tokens</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'task_summary' in results:
            success_rate = results['task_summary'].get('success_rate', 0) * 100
        elif 'execution_summary' in results:
            success_rate = 100 if results['execution_summary'].get('successful_maps', 0) > 0 else 0
        elif 'meets_threshold' in results:
            success_rate = 100 if results['meets_threshold'] else 0
        else:
            success_rate = 100
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{success_rate:.1f}%</h3>
            <p>Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'intermediate_results' in results:
            agent_count = len(results['intermediate_results'])
        elif 'individual_opinions' in results:
            agent_count = len(results['individual_opinions'])
        elif 'task_details' in results:
            agent_count = len(results['task_details'])
        elif 'iteration_history' in results:
            agent_count = 2  # Producer + Reviewer
        else:
            agent_count = 1
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{agent_count}</h3>
            <p>Agents Used</p>
        </div>
        """, unsafe_allow_html=True)

def display_results_visualization(results: Dict[str, Any], pattern_type: str):
    """Display results with appropriate visualizations."""
    if not results:
        return
    
    st.markdown("### 📈 Results Visualization")
    
    if pattern_type == "sequential":
        # Sequential execution timeline
        if 'execution_log' in results:
            df = pd.DataFrame(results['execution_log'])
            if not df.empty:
                fig = px.bar(df, x='agent', y='execution_time', 
                           title="Agent Execution Timeline",
                           labels={'execution_time': 'Execution Time (s)', 'agent': 'Agent'})
                st.plotly_chart(fig, use_container_width=True)
    
    elif pattern_type == "mapreduce":
        # Parallel processing efficiency
        if 'execution_summary' in results:
            summary = results['execution_summary']
            efficiency_data = {
                'Metric': ['Total Chunks', 'Successful Maps', 'Parallel Efficiency'],
                'Value': [
                    summary.get('total_chunks', 0),
                    summary.get('successful_maps', 0),
                    summary.get('parallel_efficiency', 0)
                ]
            }
            df = pd.DataFrame(efficiency_data)
            fig = px.bar(df, x='Metric', y='Value', title="MapReduce Efficiency Metrics")
            st.plotly_chart(fig, use_container_width=True)
    
    elif pattern_type == "consensus":
        # Consensus voting results
        if 'individual_opinions' in results:
            agents = [op['agent'] for op in results['individual_opinions']]
            opinions = [op['opinion'][:50] + "..." if len(op['opinion']) > 50 else op['opinion'] 
                       for op in results['individual_opinions']]
            
            df = pd.DataFrame({'Agent': agents, 'Opinion': opinions})
            st.dataframe(df, use_container_width=True)
            
            # Confidence visualization
            confidence = results.get('confidence_score', 0)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Consensus Confidence"},
                gauge = {'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}))
            st.plotly_chart(fig, use_container_width=True)
    
    elif pattern_type == "hierarchical":
        # Task execution hierarchy
        if 'task_details' in results:
            tasks = []
            for task_id, task_info in results['task_details'].items():
                tasks.append({
                    'Task ID': task_id,
                    'Status': task_info['status'],
                    'Agent': task_info.get('agent', 'N/A'),
                    'Execution Time': task_info.get('execution_time', 0)
                })
            
            df = pd.DataFrame(tasks)
            if not df.empty:
                fig = px.sunburst(df, path=['Status', 'Agent'], values='Execution Time',
                                title="Task Execution Hierarchy")
                st.plotly_chart(fig, use_container_width=True)
    
    elif pattern_type == "producer_reviewer":
        # Iteration improvement tracking
        if 'iteration_history' in results:
            df = pd.DataFrame(results['iteration_history'])
            if not df.empty:
                fig = px.line(df, x='iteration', y='quality_score',
                            title="Quality Score Improvement Over Iterations",
                            labels={'quality_score': 'Quality Score', 'iteration': 'Iteration'})
                fig.add_hline(y=80, line_dash="dash", line_color="green", 
                            annotation_text="Quality Threshold")
                st.plotly_chart(fig, use_container_width=True)

async def run_orchestration_demo(pattern_type: str, demo_config: Dict[str, Any]):
    """Run the selected orchestration demo."""
    api_key = st.session_state.api_key
    
    try:
        if pattern_type == "sequential":
            topic = demo_config.get('topic', 'Artificial Intelligence in Healthcare')
            result = await demo_sequential_orchestration(api_key, topic)
        
        elif pattern_type == "mapreduce":
            demo_subtype = demo_config.get('subtype', 'summarization')
            result = await demo_mapreduce_orchestration(api_key, demo_subtype)
        
        elif pattern_type == "consensus":
            demo_subtype = demo_config.get('subtype', 'sentiment')
            result = await demo_consensus_orchestration(api_key, demo_subtype)
        
        elif pattern_type == "hierarchical":
            demo_subtype = demo_config.get('subtype', 'travel')
            result = await demo_hierarchical_orchestration(api_key, demo_subtype)
        
        elif pattern_type == "producer_reviewer":
            demo_subtype = demo_config.get('subtype', 'legal')
            result = await demo_producer_reviewer_orchestration(api_key, demo_subtype)
        
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        return result
    
    except Exception as e:
        st.error(f"Execution failed: {str(e)}")
        return None

def main():
    """Main application function."""
    initialize_session_state()
    
    # Display overview
    display_pattern_overview()
    
    # API key configuration
    api_key_valid = display_api_key_input()
    
    if not api_key_valid:
        st.info("👆 Please configure your OpenAI API key in the sidebar to start exploring the orchestration patterns.")
        return
    
    # Pattern selection
    st.sidebar.markdown("## 🎯 Select Pattern")
    
    pattern_options = {
        "Sequential Orchestration": "sequential",
        "MapReduce Orchestration": "mapreduce", 
        "Consensus Orchestration": "consensus",
        "Hierarchical Orchestration": "hierarchical",
        "Producer-Reviewer Orchestration": "producer_reviewer"
    }
    
    selected_pattern = st.sidebar.selectbox(
        "Choose an orchestration pattern to explore:",
        list(pattern_options.keys())
    )
    
    pattern_type = pattern_options[selected_pattern]
    
    # Pattern-specific configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    demo_config = {}
    
    if pattern_type == "sequential":
        demo_config['topic'] = st.sidebar.text_input(
            "Report Topic",
            value="Artificial Intelligence in Healthcare",
            help="Topic for the report generation pipeline"
        )
    
    elif pattern_type == "mapreduce":
        demo_config['subtype'] = st.sidebar.selectbox(
            "Demo Type",
            ["summarization", "sentiment"],
            help="Choose between document summarization or sentiment analysis"
        )
    
    elif pattern_type == "consensus":
        demo_config['subtype'] = st.sidebar.selectbox(
            "Demo Type", 
            ["sentiment", "decision"],
            help="Choose between sentiment analysis or business decision consensus"
        )
    
    elif pattern_type == "hierarchical":
        demo_config['subtype'] = st.sidebar.selectbox(
            "Demo Type",
            ["travel", "project"],
            help="Choose between travel planning or project management"
        )
    
    elif pattern_type == "producer_reviewer":
        demo_config['subtype'] = st.sidebar.selectbox(
            "Demo Type",
            ["legal", "marketing", "technical"],
            help="Choose the type of content creation and review"
        )
    
    # Main content area
    st.markdown(f"## {selected_pattern}")
    
    # Pattern description
    pattern_descriptions = {
        "sequential": "Execute agents in a fixed sequence where each agent's output becomes the next agent's input. Perfect for step-by-step processes that require building upon previous results.",
        "mapreduce": "Break large tasks into independent chunks, process them in parallel, then combine results. Excellent for scalable processing of large datasets.",
        "consensus": "Multiple agents independently solve the same problem and vote on the best solution. Improves reliability through collective intelligence.",
        "hierarchical": "A master orchestrator decomposes complex tasks and coordinates specialized agents. Ideal for complex, cross-domain problems.",
        "producer_reviewer": "Iterative content improvement through producer-reviewer cycles until quality standards are met. Ensures high-quality outputs through feedback loops."
    }
    
    st.info(pattern_descriptions[pattern_type])
    
    # Execution button
    if st.button(f"🚀 Run {selected_pattern} Demo", type="primary"):
        with st.spinner(f"Executing {selected_pattern}..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 20:
                    status_text.text("Initializing agents...")
                elif i < 40:
                    status_text.text("Processing tasks...")
                elif i < 70:
                    status_text.text("Coordinating agents...")
                elif i < 90:
                    status_text.text("Finalizing results...")
                else:
                    status_text.text("Complete!")
                time.sleep(0.02)
            
            # Run the actual demo
            result = asyncio.run(run_orchestration_demo(pattern_type, demo_config))
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if result:
                st.session_state.execution_results[pattern_type] = result
                st.session_state.execution_history.append({
                    'timestamp': time.time(),
                    'pattern': selected_pattern,
                    'config': demo_config,
                    'success': True
                })
                
                st.markdown('<div class="success-message">✅ Execution completed successfully!</div>', 
                          unsafe_allow_html=True)
                
                # Display metrics
                display_execution_metrics(result)
                
                # Display visualization
                display_results_visualization(result, pattern_type)
                
                # Display detailed results
                with st.expander("📋 Detailed Results", expanded=False):
                    if pattern_type == "sequential" and 'final_output' in result:
                        st.markdown("### Final Output")
                        st.text_area("Result", result['final_output'], height=200)
                        
                        if 'execution_log' in result:
                            st.markdown("### Execution Log")
                            for log_entry in result['execution_log']:
                                st.markdown(f"**{log_entry['agent']}** ({log_entry['execution_time']:.2f}s)")
                                st.text(log_entry['output'][:300] + "..." if len(log_entry['output']) > 300 else log_entry['output'])
                    
                    elif pattern_type == "consensus" and 'consensus_decision' in result:
                        st.markdown("### Consensus Decision")
                        st.text_area("Decision", result['consensus_decision'], height=100)
                        
                        st.markdown("### Individual Opinions")
                        for opinion in result['individual_opinions']:
                            st.markdown(f"**{opinion['agent']}**: {opinion['opinion']}")
                    
                    elif pattern_type == "producer_reviewer" and 'final_content' in result:
                        st.markdown("### Final Content")
                        st.text_area("Content", result['final_content'], height=300)
                        
                        st.markdown("### Final Review")
                        review = result['final_review']
                        st.markdown(f"**Status**: {review['status']}")
                        st.markdown(f"**Score**: {review['score']}/100")
                        st.markdown(f"**Feedback**: {review['feedback']}")
                    
                    else:
                        st.json(result)
            
            else:
                st.session_state.execution_history.append({
                    'timestamp': time.time(),
                    'pattern': selected_pattern,
                    'config': demo_config,
                    'success': False
                })
                
                st.markdown('<div class="error-message">❌ Execution failed. Please check your API key and try again.</div>', 
                          unsafe_allow_html=True)
    
    # Execution history
    if st.session_state.execution_history:
        st.sidebar.markdown("## 📈 Execution History")
        
        for i, entry in enumerate(reversed(st.session_state.execution_history[-5:])):
            status_icon = "✅" if entry['success'] else "❌"
            timestamp = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
            st.sidebar.markdown(f"{status_icon} {entry['pattern']} - {timestamp}")

if __name__ == "__main__":
    main()
