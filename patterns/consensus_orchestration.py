"""
Consensus Orchestration Pattern - Redundant Verification for Reliability

Multiple agents independently approach the same problem, then compare and aggregate 
results. Leverages the "wisdom of crowds" statistical principle to improve 
decision-making quality and reduce false positives.

Example: Sentiment Analysis with Multiple Perspectives
1. Multiple agents analyze the same text independently
2. Each agent uses different approaches/training backgrounds
3. Results are aggregated through voting or weighted averaging
4. Final decision based on consensus with confidence metrics
"""
import asyncio
from typing import List, Any, Dict, Optional, Union
import statistics
import json
from collections import Counter
from agents.base_agent import BaseAgent, SimpleAgent


class ConsensusOrchestrator:
    """Orchestrates agents using consensus-based decision making."""
    
    def __init__(self, agents: List[BaseAgent], consensus_threshold: float = 0.6):
        self.agents = agents
        self.consensus_threshold = consensus_threshold  # Minimum agreement required
        self.execution_log = []
        self.total_execution_time = 0
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    async def _collect_opinions(self, input_data: Any, context: Optional[Dict] = None) -> List[Dict]:
        """Collect independent opinions from all agents."""
        print(f"🗳️  Collecting opinions from {len(self.agents)} agents")
        
        async def get_agent_opinion(agent, data):
            try:
                result = await agent.process(data, context)
                metrics = agent.get_metrics()
                
                return {
                    "agent": agent.name,
                    "role": agent.role,
                    "opinion": result,
                    "metrics": metrics,
                    "success": True
                }
            except Exception as e:
                return {
                    "agent": agent.name,
                    "role": agent.role,
                    "error": str(e),
                    "success": False
                }
        
        # Collect all opinions concurrently
        tasks = [get_agent_opinion(agent, input_data) for agent in self.agents]
        opinions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and update metrics
        valid_opinions = []
        for opinion in opinions:
            if isinstance(opinion, dict) and opinion.get("success"):
                valid_opinions.append(opinion)
                metrics = opinion["metrics"]
                self.total_execution_time += metrics["execution_time"]
                for key in self.total_tokens:
                    self.total_tokens[key] += metrics["token_usage"][key]
            else:
                print(f"❌ Agent failed: {opinion}")
        
        return valid_opinions
    
    def _calculate_consensus(self, opinions: List[Dict], consensus_method: str = "majority") -> Dict:
        """Calculate consensus from collected opinions."""
        if not opinions:
            return {"consensus": None, "confidence": 0, "method": consensus_method}
        
        if consensus_method == "majority":
            return self._majority_consensus(opinions)
        elif consensus_method == "weighted":
            return self._weighted_consensus(opinions)
        elif consensus_method == "average":
            return self._average_consensus(opinions)
        else:
            raise ValueError(f"Unknown consensus method: {consensus_method}")
    
    def _majority_consensus(self, opinions: List[Dict]) -> Dict:
        """Simple majority voting consensus."""
        opinion_texts = [op["opinion"] for op in opinions]
        
        # Count occurrences of each opinion
        opinion_counter = Counter(opinion_texts)
        most_common = opinion_counter.most_common(1)[0]
        
        consensus_opinion = most_common[0]
        vote_count = most_common[1]
        confidence = vote_count / len(opinions)
        
        return {
            "consensus": consensus_opinion,
            "confidence": confidence,
            "vote_distribution": dict(opinion_counter),
            "method": "majority",
            "meets_threshold": confidence >= self.consensus_threshold
        }
    
    def _weighted_consensus(self, opinions: List[Dict]) -> Dict:
        """Weighted consensus based on agent confidence or performance."""
        # For simplicity, using equal weights. In practice, you might weight by:
        # - Agent historical accuracy
        # - Response confidence scores
        # - Agent specialization relevance
        
        weights = [1.0] * len(opinions)  # Equal weights for now
        total_weight = sum(weights)
        
        # This is a simplified version - in practice, you'd need more sophisticated
        # methods for combining different types of opinions
        opinion_texts = [op["opinion"] for op in opinions]
        
        # For demonstration, we'll use the most common opinion with weight consideration
        weighted_counter = Counter()
        for i, opinion in enumerate(opinion_texts):
            weighted_counter[opinion] += weights[i]
        
        most_common = weighted_counter.most_common(1)[0]
        consensus_opinion = most_common[0]
        weighted_confidence = most_common[1] / total_weight
        
        return {
            "consensus": consensus_opinion,
            "confidence": weighted_confidence,
            "weighted_distribution": dict(weighted_counter),
            "method": "weighted",
            "meets_threshold": weighted_confidence >= self.consensus_threshold
        }
    
    def _average_consensus(self, opinions: List[Dict]) -> Dict:
        """Average-based consensus for numerical opinions."""
        try:
            # Try to extract numerical values from opinions
            numerical_values = []
            for op in opinions:
                opinion = op["opinion"]
                # Try to extract numbers from the opinion text
                import re
                numbers = re.findall(r'-?\d+\.?\d*', str(opinion))
                if numbers:
                    numerical_values.append(float(numbers[0]))
            
            if numerical_values:
                avg_value = statistics.mean(numerical_values)
                std_dev = statistics.stdev(numerical_values) if len(numerical_values) > 1 else 0
                confidence = 1.0 - (std_dev / avg_value) if avg_value != 0 else 0
                
                return {
                    "consensus": avg_value,
                    "confidence": max(0, min(1, confidence)),
                    "average": avg_value,
                    "std_deviation": std_dev,
                    "method": "average",
                    "meets_threshold": confidence >= self.consensus_threshold
                }
        except:
            pass
        
        # Fallback to majority if numerical averaging fails
        return self._majority_consensus(opinions)
    
    async def execute(self, input_data: Any, consensus_method: str = "majority", 
                     context: Optional[Dict] = None) -> Dict:
        """Execute consensus-based decision making."""
        
        try:
            # Collect independent opinions
            opinions = await self._collect_opinions(input_data, context)
            
            if not opinions:
                raise Exception("No valid opinions collected from agents")
            
            # Calculate consensus
            consensus_result = self._calculate_consensus(opinions, consensus_method)
            
            # Prepare detailed results
            return {
                "consensus_decision": consensus_result["consensus"],
                "confidence_score": consensus_result["confidence"],
                "meets_threshold": consensus_result.get("meets_threshold", False),
                "consensus_method": consensus_method,
                "individual_opinions": opinions,
                "consensus_details": consensus_result,
                "execution_summary": {
                    "total_agents": len(self.agents),
                    "successful_agents": len(opinions),
                    "total_execution_time": self.total_execution_time,
                    "total_tokens": self.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }


class SentimentConsensusSystem:
    """Example implementation: Multi-perspective Sentiment Analysis."""
    
    @staticmethod
    def create_system(api_key: str) -> ConsensusOrchestrator:
        """Create a sentiment analysis consensus system."""
        
        agents = [
            SimpleAgent(
                name="Optimistic Analyzer",
                role="Positive-leaning Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are a sentiment analyzer with a slight optimistic bias. You tend to notice positive aspects and give benefit of the doubt.
                
                Instructions:
                - Analyze the sentiment of the given text
                - Look for positive indicators and constructive elements
                - Consider context that might make negative statements less harsh
                - Rate sentiment on scale: Very Negative (-2), Negative (-1), Neutral (0), Positive (1), Very Positive (2)
                - Provide your rating and brief reasoning
                
                Output format: "Rating: [number] - [brief explanation]" """
            ),
            
            SimpleAgent(
                name="Critical Analyzer",
                role="Critical Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are a sentiment analyzer with a critical perspective. You're good at detecting subtle negativity and sarcasm.
                
                Instructions:
                - Analyze the sentiment of the given text critically
                - Look for hidden negativity, sarcasm, and passive-aggressive tones
                - Consider implications and subtext
                - Rate sentiment on scale: Very Negative (-2), Negative (-1), Neutral (0), Positive (1), Very Positive (2)
                - Provide your rating and brief reasoning
                
                Output format: "Rating: [number] - [brief explanation]" """
            ),
            
            SimpleAgent(
                name="Balanced Analyzer",
                role="Neutral Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are a balanced sentiment analyzer. You provide objective, unbiased sentiment analysis.
                
                Instructions:
                - Analyze the sentiment objectively without bias
                - Consider both positive and negative aspects equally
                - Focus on explicit sentiment indicators
                - Rate sentiment on scale: Very Negative (-2), Negative (-1), Neutral (0), Positive (1), Very Positive (2)
                - Provide your rating and brief reasoning
                
                Output format: "Rating: [number] - [brief explanation]" """
            ),
            
            SimpleAgent(
                name="Context Analyzer",
                role="Context-aware Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are a context-aware sentiment analyzer. You excel at understanding cultural context and implied meanings.
                
                Instructions:
                - Analyze sentiment considering cultural and social context
                - Look for implied meanings and cultural references
                - Consider the broader context and situation
                - Rate sentiment on scale: Very Negative (-2), Negative (-1), Neutral (0), Positive (1), Very Positive (2)
                - Provide your rating and brief reasoning
                
                Output format: "Rating: [number] - [brief explanation]" """
            ),
            
            SimpleAgent(
                name="Linguistic Analyzer",
                role="Language-focused Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are a linguistic sentiment analyzer. You focus on word choice, grammar, and language patterns.
                
                Instructions:
                - Analyze sentiment based on linguistic patterns and word choice
                - Consider grammatical structures that indicate sentiment
                - Look at word connotations and linguistic markers
                - Rate sentiment on scale: Very Negative (-2), Negative (-1), Neutral (0), Positive (1), Very Positive (2)
                - Provide your rating and brief reasoning
                
                Output format: "Rating: [number] - [brief explanation]" """
            )
        ]
        
        return ConsensusOrchestrator(agents, consensus_threshold=0.6)


class DecisionMakingConsensus:
    """Example implementation: Business Decision Consensus."""
    
    @staticmethod
    def create_system(api_key: str) -> ConsensusOrchestrator:
        """Create a business decision consensus system."""
        
        agents = [
            SimpleAgent(
                name="Financial Advisor",
                role="Financial Analysis",
                api_key=api_key,
                system_prompt="""You are a financial advisor analyzing business decisions from a financial perspective.
                
                Instructions:
                - Evaluate the financial implications of the proposed decision
                - Consider costs, revenues, ROI, and financial risks
                - Provide recommendation: Strongly Recommend, Recommend, Neutral, Not Recommend, Strongly Against
                - Include brief financial reasoning
                
                Output format: "[Recommendation] - [financial reasoning]" """
            ),
            
            SimpleAgent(
                name="Risk Manager",
                role="Risk Assessment",
                api_key=api_key,
                system_prompt="""You are a risk management specialist analyzing potential risks and mitigation strategies.
                
                Instructions:
                - Identify and assess risks associated with the decision
                - Consider operational, strategic, and market risks
                - Evaluate risk mitigation possibilities
                - Provide recommendation: Strongly Recommend, Recommend, Neutral, Not Recommend, Strongly Against
                - Include brief risk assessment
                
                Output format: "[Recommendation] - [risk assessment]" """
            ),
            
            SimpleAgent(
                name="Market Analyst",
                role="Market Analysis",
                api_key=api_key,
                system_prompt="""You are a market analyst evaluating decisions from a market and competitive perspective.
                
                Instructions:
                - Analyze market conditions and competitive implications
                - Consider customer impact and market positioning
                - Evaluate timing and market readiness
                - Provide recommendation: Strongly Recommend, Recommend, Neutral, Not Recommend, Strongly Against
                - Include brief market analysis
                
                Output format: "[Recommendation] - [market analysis]" """
            ),
            
            SimpleAgent(
                name="Operations Expert",
                role="Operational Feasibility",
                api_key=api_key,
                system_prompt="""You are an operations expert evaluating the operational feasibility and implementation aspects.
                
                Instructions:
                - Assess operational requirements and capabilities
                - Consider resource needs and implementation challenges
                - Evaluate scalability and operational efficiency
                - Provide recommendation: Strongly Recommend, Recommend, Neutral, Not Recommend, Strongly Against
                - Include brief operational assessment
                
                Output format: "[Recommendation] - [operational assessment]" """
            )
        ]
        
        return ConsensusOrchestrator(agents, consensus_threshold=0.75)


# Example usage and demonstration
async def demo_consensus_orchestration(api_key: str, demo_type: str = "sentiment"):
    """Demonstrate consensus orchestration."""
    
    print("🚀 Consensus Orchestration Demo")
    print("=" * 45)
    
    if demo_type == "sentiment":
        # Sentiment analysis consensus demo
        sample_texts = [
            "I absolutely love this new product! It's amazing and works perfectly.",
            "This is okay, I guess. Not great, not terrible. Just average.",
            "What a disaster! This is the worst experience I've ever had. Completely useless.",
            "It's pretty good overall, though there are some minor issues to work out.",
            "I'm not sure how I feel about this. It has potential but needs improvement."
        ]
        
        system = SentimentConsensusSystem.create_system(api_key)
        
        for i, text in enumerate(sample_texts):
            print(f"\n📝 Analyzing text {i+1}: '{text[:50]}...'")
            
            try:
                result = await system.execute(
                    input_data=text,
                    consensus_method="majority",
                    context={"analysis_type": "product_review"}
                )
                
                if result.get("error"):
                    print(f"❌ Analysis failed: {result['error']}")
                    continue
                
                print(f"✅ Consensus Decision: {result['consensus_decision']}")
                print(f"📊 Confidence Score: {result['confidence_score']:.2f}")
                print(f"🎯 Meets Threshold: {result['meets_threshold']}")
                print(f"👥 Agents Participated: {result['execution_summary']['successful_agents']}/{result['execution_summary']['total_agents']}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        return result if 'result' in locals() else None
    
    elif demo_type == "decision":
        # Business decision consensus demo
        decision_scenario = """
        Our company is considering launching a new AI-powered customer service chatbot. 
        The initial investment would be $500,000, with expected annual savings of $200,000 
        in customer service costs. The project would take 6 months to implement and 
        requires training our support team on new processes. Market research shows 
        70% of our competitors already use similar technology.
        """
        
        system = DecisionMakingConsensus.create_system(api_key)
        
        print(f"\n📋 Business Decision Scenario:")
        print(decision_scenario)
        
        try:
            result = await system.execute(
                input_data=decision_scenario,
                consensus_method="weighted",
                context={"decision_type": "technology_investment", "urgency": "medium"}
            )
            
            if result.get("error"):
                print(f"❌ Decision analysis failed: {result['error']}")
                return None
            
            print(f"\n✅ Consensus Recommendation: {result['consensus_decision']}")
            print(f"📊 Confidence Score: {result['confidence_score']:.2f}")
            print(f"🎯 Meets Threshold: {result['meets_threshold']}")
            
            print(f"\n👥 Individual Expert Opinions:")
            for opinion in result['individual_opinions']:
                print(f"   {opinion['agent']}: {opinion['opinion'][:100]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
