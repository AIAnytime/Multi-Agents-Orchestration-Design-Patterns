"""
MapReduce Orchestration Pattern - Parallel Computing Intelligence

Breaks large tasks into multiple independent subtasks and processes them in parallel,
then aggregates results. Based on distributed computing principles.

Example: Large-scale Text Processing
1. Map Phase: Divide document collection into segments
2. Process Phase: Each agent processes a segment in parallel  
3. Reduce Phase: Aggregate all results into final output
"""
import asyncio
from typing import List, Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import time
from agents.base_agent import BaseAgent, SimpleAgent


class MapReduceOrchestrator:
    """Orchestrates agents using MapReduce pattern."""
    
    def __init__(self, mapper_agents: List[BaseAgent], reducer_agent: BaseAgent, max_workers: int = 4):
        self.mapper_agents = mapper_agents
        self.reducer_agent = reducer_agent
        self.max_workers = max_workers
        self.execution_log = []
        self.total_execution_time = 0
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def _split_data(self, data: Any, split_function: Callable) -> List[Any]:
        """Split input data into chunks for parallel processing."""
        return split_function(data)
    
    async def _map_phase(self, data_chunks: List[Any], context: Optional[Dict] = None) -> List[Dict]:
        """Execute map phase - process chunks in parallel."""
        print(f"🗺️  Starting Map Phase with {len(data_chunks)} chunks")
        
        async def process_chunk(agent, chunk, chunk_id):
            try:
                print(f"   Processing chunk {chunk_id + 1} with {agent.name}")
                result = await agent.process(chunk, context)
                metrics = agent.get_metrics()
                
                return {
                    "chunk_id": chunk_id,
                    "agent": agent.name,
                    "result": result,
                    "metrics": metrics,
                    "success": True
                }
            except Exception as e:
                return {
                    "chunk_id": chunk_id,
                    "agent": agent.name,
                    "error": str(e),
                    "success": False
                }
        
        # Create tasks for parallel execution
        tasks = []
        for i, chunk in enumerate(data_chunks):
            agent = self.mapper_agents[i % len(self.mapper_agents)]  # Round-robin assignment
            tasks.append(process_chunk(agent, chunk, i))
        
        # Execute all tasks concurrently
        map_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and update metrics
        successful_results = []
        for result in map_results:
            if isinstance(result, dict) and result.get("success"):
                successful_results.append(result)
                metrics = result["metrics"]
                self.total_execution_time += metrics["execution_time"]
                for key in self.total_tokens:
                    self.total_tokens[key] += metrics["token_usage"][key]
            else:
                print(f"❌ Map task failed: {result}")
        
        return successful_results
    
    async def _reduce_phase(self, map_results: List[Dict], context: Optional[Dict] = None) -> Dict:
        """Execute reduce phase - aggregate results."""
        print(f"🔄 Starting Reduce Phase with {len(map_results)} results")
        
        # Prepare input for reducer
        aggregated_input = {
            "map_results": [r["result"] for r in map_results],
            "metadata": {
                "total_chunks": len(map_results),
                "processing_agents": list(set(r["agent"] for r in map_results))
            }
        }
        
        try:
            final_result = await self.reducer_agent.process(aggregated_input, context)
            metrics = self.reducer_agent.get_metrics()
            
            # Update metrics
            self.total_execution_time += metrics["execution_time"]
            for key in self.total_tokens:
                self.total_tokens[key] += metrics["token_usage"][key]
            
            return {
                "final_result": final_result,
                "reducer_metrics": metrics,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    async def execute(self, input_data: Any, split_function: Callable, context: Optional[Dict] = None) -> Dict:
        """Execute the complete MapReduce pipeline."""
        start_time = time.time()
        
        try:
            # Split data into chunks
            print("📊 Splitting data into chunks...")
            data_chunks = self._split_data(input_data, split_function)
            print(f"   Created {len(data_chunks)} chunks")
            
            # Map phase - parallel processing
            map_results = await self._map_phase(data_chunks, context)
            
            if not map_results:
                raise Exception("All map tasks failed")
            
            # Reduce phase - aggregate results
            reduce_result = await self._reduce_phase(map_results, context)
            
            if not reduce_result.get("success"):
                raise Exception(f"Reduce phase failed: {reduce_result.get('error')}")
            
            total_time = time.time() - start_time
            
            return {
                "final_output": reduce_result["final_result"],
                "map_results": map_results,
                "execution_summary": {
                    "total_chunks": len(data_chunks),
                    "successful_maps": len(map_results),
                    "total_execution_time": total_time,
                    "parallel_efficiency": len(data_chunks) / total_time if total_time > 0 else 0,
                    "total_tokens": self.total_tokens
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "success": False
            }


class DocumentSummarizationPipeline:
    """Example implementation: Large-scale Document Summarization."""
    
    @staticmethod
    def create_pipeline(api_key: str, num_mappers: int = 3) -> MapReduceOrchestrator:
        """Create a document summarization MapReduce pipeline."""
        
        # Create mapper agents for parallel processing
        mapper_agents = []
        for i in range(num_mappers):
            mapper_agents.append(
                SimpleAgent(
                    name=f"Summarizer-{i+1}",
                    role="Document Chunk Summarizer",
                    api_key=api_key,
                    system_prompt="""You are a document summarization specialist. Your job is to create concise, accurate summaries of text chunks.
                    
                    Instructions:
                    - Read the provided text chunk carefully
                    - Extract the main points and key information
                    - Create a concise summary (2-3 paragraphs max)
                    - Preserve important details, names, dates, and figures
                    - Maintain the original context and meaning
                    - Focus on the most significant content"""
                )
            )
        
        # Create reducer agent for final aggregation
        reducer_agent = SimpleAgent(
            name="Master Summarizer",
            role="Summary Aggregator",
            api_key=api_key,
            system_prompt="""You are a master document summarizer. Your job is to combine multiple chunk summaries into a comprehensive final summary.
            
            Instructions:
            - Review all the provided chunk summaries
            - Identify common themes and key points across summaries
            - Create a coherent, comprehensive final summary
            - Eliminate redundancy while preserving important information
            - Organize content logically with clear structure
            - Ensure the final summary captures the essence of the entire document
            - Include an executive summary at the beginning
            - Structure: Executive Summary, Key Points, Detailed Analysis, Conclusions"""
        )
        
        return MapReduceOrchestrator(mapper_agents, reducer_agent)
    
    @staticmethod
    def split_text_by_paragraphs(text: str, max_chunks: int = 6) -> List[str]:
        """Split text into chunks by paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= max_chunks:
            return paragraphs
        
        # Group paragraphs into chunks
        chunk_size = len(paragraphs) // max_chunks
        chunks = []
        
        for i in range(0, len(paragraphs), chunk_size):
            chunk = '\n\n'.join(paragraphs[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks[:max_chunks]  # Ensure we don't exceed max_chunks


class SentimentAnalysisPipeline:
    """Example implementation: Parallel Sentiment Analysis."""
    
    @staticmethod
    def create_pipeline(api_key: str, num_mappers: int = 4) -> MapReduceOrchestrator:
        """Create a sentiment analysis MapReduce pipeline."""
        
        # Create mapper agents with different analysis approaches
        mapper_agents = [
            SimpleAgent(
                name="Emotion Analyzer",
                role="Emotional Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are an emotion-focused sentiment analyzer. Analyze the emotional tone and sentiment of the given text.
                
                Focus on:
                - Overall emotional tone (positive, negative, neutral)
                - Specific emotions detected (joy, anger, sadness, fear, etc.)
                - Emotional intensity (1-10 scale)
                - Key phrases that indicate sentiment
                
                Output format: JSON with sentiment, emotions, intensity, and key_phrases"""
            ),
            
            SimpleAgent(
                name="Context Analyzer",
                role="Contextual Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are a context-aware sentiment analyzer. Analyze sentiment considering context, sarcasm, and implied meanings.
                
                Focus on:
                - Contextual sentiment (considering implied meanings)
                - Sarcasm and irony detection
                - Cultural and social context
                - Confidence level in analysis
                
                Output format: JSON with contextual_sentiment, sarcasm_detected, confidence, and reasoning"""
            ),
            
            SimpleAgent(
                name="Linguistic Analyzer",
                role="Linguistic Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are a linguistic sentiment analyzer. Focus on language patterns, word choice, and grammatical structures.
                
                Focus on:
                - Word choice and connotations
                - Sentence structure impact on sentiment
                - Linguistic patterns and markers
                - Formal vs informal tone
                
                Output format: JSON with linguistic_sentiment, tone_formality, word_patterns, and linguistic_markers"""
            ),
            
            SimpleAgent(
                name="Topic Analyzer",
                role="Topic-based Sentiment Analysis",
                api_key=api_key,
                system_prompt="""You are a topic-focused sentiment analyzer. Analyze sentiment in relation to specific topics and themes.
                
                Focus on:
                - Topic identification
                - Sentiment toward specific topics
                - Theme-based emotional responses
                - Subject-specific sentiment patterns
                
                Output format: JSON with topics, topic_sentiments, themes, and subject_analysis"""
            )
        ]
        
        # Create reducer for final sentiment aggregation
        reducer_agent = SimpleAgent(
            name="Sentiment Aggregator",
            role="Final Sentiment Analysis",
            api_key=api_key,
            system_prompt="""You are a master sentiment analyzer. Combine multiple sentiment analyses into a comprehensive final assessment.
            
            Instructions:
            - Review all sentiment analyses from different perspectives
            - Identify consensus and disagreements
            - Create a weighted final sentiment score
            - Provide confidence intervals and uncertainty measures
            - Explain the reasoning behind the final assessment
            - Include recommendations for interpretation
            
            Output comprehensive sentiment report with final scores, confidence levels, and detailed analysis."""
        )
        
        return MapReduceOrchestrator(mapper_agents, reducer_agent)
    
    @staticmethod
    def split_text_by_sentences(text: str, max_chunks: int = 4) -> List[str]:
        """Split text into sentence-based chunks."""
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        if len(sentences) <= max_chunks:
            return sentences
        
        chunk_size = len(sentences) // max_chunks
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks[:max_chunks]


# Example usage and demonstration
async def demo_mapreduce_orchestration(api_key: str, demo_type: str = "summarization"):
    """Demonstrate MapReduce orchestration."""
    
    print("🚀 MapReduce Orchestration Demo")
    print("=" * 50)
    
    if demo_type == "summarization":
        # Document summarization demo
        sample_document = """
        Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, revolutionizing industries and reshaping the way we live and work. From healthcare to finance, transportation to entertainment, AI applications are becoming increasingly sophisticated and widespread.

        In healthcare, AI is being used to diagnose diseases more accurately and quickly than ever before. Machine learning algorithms can analyze medical images, predict patient outcomes, and even assist in drug discovery. For example, AI systems can now detect certain types of cancer in medical scans with accuracy rates that match or exceed those of human radiologists.

        The financial sector has also embraced AI technology extensively. Banks and financial institutions use AI for fraud detection, algorithmic trading, risk assessment, and customer service automation. These applications have not only improved efficiency but also enhanced security and customer experience.

        Transportation is another area where AI is making significant strides. Autonomous vehicles, powered by AI, promise to reduce traffic accidents, improve fuel efficiency, and provide mobility solutions for people who cannot drive. Companies like Tesla, Google, and Uber are investing billions of dollars in developing self-driving car technology.

        However, the rapid advancement of AI also raises important ethical and societal questions. Issues such as job displacement, privacy concerns, algorithmic bias, and the need for AI governance are becoming increasingly important. As AI systems become more powerful and autonomous, ensuring they are developed and deployed responsibly becomes crucial.

        The future of AI holds immense promise, but it also requires careful consideration of its implications. As we continue to develop and integrate AI technologies into our daily lives, we must balance innovation with responsibility, ensuring that AI serves humanity's best interests while minimizing potential risks and negative consequences.
        """
        
        pipeline = DocumentSummarizationPipeline.create_pipeline(api_key)
        
        try:
            result = await pipeline.execute(
                input_data=sample_document,
                split_function=DocumentSummarizationPipeline.split_text_by_paragraphs,
                context={"document_type": "technology_overview", "target_length": "comprehensive"}
            )
            
            print("\n✅ MapReduce pipeline completed successfully!")
            print(f"📊 Execution Summary:")
            print(f"   - Total chunks processed: {result['execution_summary']['total_chunks']}")
            print(f"   - Successful map operations: {result['execution_summary']['successful_maps']}")
            print(f"   - Total execution time: {result['execution_summary']['total_execution_time']:.2f} seconds")
            print(f"   - Parallel efficiency: {result['execution_summary']['parallel_efficiency']:.2f} chunks/second")
            print(f"   - Total tokens used: {result['execution_summary']['total_tokens']['total_tokens']}")
            
            return result
            
        except Exception as e:
            print(f"\n❌ MapReduce pipeline failed: {str(e)}")
            return None
    
    elif demo_type == "sentiment":
        # Sentiment analysis demo
        sample_text = "I absolutely love this new AI technology! It's incredibly innovative and will definitely change how we work. However, I'm a bit concerned about the potential job losses and privacy issues. Overall, I'm cautiously optimistic about the future."
        
        pipeline = SentimentAnalysisPipeline.create_pipeline(api_key)
        
        try:
            result = await pipeline.execute(
                input_data=sample_text,
                split_function=lambda x: [x],  # Single text for multiple analysis approaches
                context={"analysis_depth": "comprehensive", "include_confidence": True}
            )
            
            print("\n✅ Sentiment Analysis pipeline completed!")
            return result
            
        except Exception as e:
            print(f"\n❌ Sentiment Analysis pipeline failed: {str(e)}")
            return None
