"""
Sequential Orchestration Pattern - Pipeline Processing

Agents perform tasks sequentially in a fixed order. The output of one agent 
serves as the input for the next, forming a clear data flow pipeline.

Example: Report Generation System
1. Data Collection Agent -> Raw Information
2. Formatting Agent -> Structured Data  
3. Analysis Agent -> Key Insights
4. Optimization Agent -> Improved Presentation
5. Delivery Agent -> Final Output
"""
import asyncio
from typing import List, Any, Dict, Optional
from agents.base_agent import BaseAgent, SimpleAgent


class SequentialOrchestrator:
    """Orchestrates agents in a sequential pipeline."""
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.execution_log = []
        self.total_execution_time = 0
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    async def execute(self, initial_input: Any, context: Optional[Dict] = None) -> Dict:
        """Execute the sequential pipeline."""
        current_input = initial_input
        results = []
        self.execution_log = []
        
        for i, agent in enumerate(self.agents):
            try:
                print(f"🔄 Executing {agent.name} (Step {i+1}/{len(self.agents)})")
                
                # Process current input
                output = await agent.process(current_input, context)
                
                # Log execution details
                metrics = agent.get_metrics()
                self.execution_log.append({
                    "step": i + 1,
                    "agent": agent.name,
                    "role": agent.role,
                    "input": str(current_input)[:200] + "..." if len(str(current_input)) > 200 else str(current_input),
                    "output": str(output)[:200] + "..." if len(str(output)) > 200 else str(output),
                    "execution_time": metrics["execution_time"],
                    "token_usage": metrics["token_usage"]
                })
                
                # Update totals
                self.total_execution_time += metrics["execution_time"]
                for key in self.total_tokens:
                    self.total_tokens[key] += metrics["token_usage"][key]
                
                results.append({
                    "agent": agent.name,
                    "output": output
                })
                
                # Output becomes input for next agent
                current_input = output
                
            except Exception as e:
                error_msg = f"Error in {agent.name}: {str(e)}"
                print(f"❌ {error_msg}")
                self.execution_log.append({
                    "step": i + 1,
                    "agent": agent.name,
                    "error": error_msg
                })
                raise Exception(f"Pipeline failed at step {i+1}: {error_msg}")
        
        return {
            "final_output": current_input,
            "intermediate_results": results,
            "execution_log": self.execution_log,
            "total_execution_time": self.total_execution_time,
            "total_tokens": self.total_tokens
        }


class ReportGenerationPipeline:
    """Example implementation: Report Generation System."""
    
    @staticmethod
    def create_pipeline(api_key: str) -> SequentialOrchestrator:
        """Create a report generation pipeline."""
        
        agents = [
            SimpleAgent(
                name="Data Collector",
                role="Information Gathering",
                api_key=api_key,
                system_prompt="""You are a data collection specialist. Your job is to gather and organize raw information about the given topic. 
                
                Instructions:
                - Identify key data points and facts
                - Organize information logically
                - Include relevant statistics, dates, and figures
                - Provide comprehensive coverage of the topic
                - Output should be factual and well-structured raw data"""
            ),
            
            SimpleAgent(
                name="Data Formatter",
                role="Structure & Organization",
                api_key=api_key,
                system_prompt="""You are a data formatting specialist. Transform raw information into a well-structured format.
                
                Instructions:
                - Organize data into clear sections and categories
                - Create proper headings and subheadings
                - Format lists, tables, and hierarchical information
                - Ensure logical flow and readability
                - Maintain all important information while improving structure"""
            ),
            
            SimpleAgent(
                name="Data Analyst",
                role="Insight Generation",
                api_key=api_key,
                system_prompt="""You are a data analyst. Extract key insights, patterns, and conclusions from structured data.
                
                Instructions:
                - Identify trends, patterns, and correlations
                - Generate actionable insights
                - Highlight key findings and implications
                - Provide analysis and interpretation
                - Focus on what the data means and why it matters"""
            ),
            
            SimpleAgent(
                name="Content Optimizer",
                role="Quality Enhancement",
                api_key=api_key,
                system_prompt="""You are a content optimization specialist. Improve the presentation quality and readability.
                
                Instructions:
                - Enhance clarity and readability
                - Improve flow and transitions
                - Optimize language and tone
                - Ensure professional presentation
                - Add executive summary if appropriate
                - Make content engaging and accessible"""
            ),
            
            SimpleAgent(
                name="Report Finalizer",
                role="Final Delivery",
                api_key=api_key,
                system_prompt="""You are a report finalization specialist. Create the final polished report ready for delivery.
                
                Instructions:
                - Add professional formatting and structure
                - Include executive summary at the top
                - Ensure consistent style and tone
                - Add conclusions and recommendations
                - Create a complete, publication-ready document
                - Include proper sections: Summary, Analysis, Insights, Recommendations"""
            )
        ]
        
        return SequentialOrchestrator(agents)


# Example usage and demonstration
async def demo_sequential_orchestration(api_key: str, topic: str = "Artificial Intelligence in Healthcare"):
    """Demonstrate sequential orchestration with report generation."""
    
    print("🚀 Sequential Orchestration Demo: Report Generation Pipeline")
    print("=" * 60)
    
    # Create pipeline
    pipeline = ReportGenerationPipeline.create_pipeline(api_key)
    
    # Execute pipeline
    try:
        result = await pipeline.execute(
            initial_input=f"Generate a comprehensive report about: {topic}",
            context={"report_type": "business_analysis", "target_audience": "executives"}
        )
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Total execution time: {result['total_execution_time']:.2f} seconds")
        print(f"🔤 Total tokens used: {result['total_tokens']['total_tokens']}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        return None
