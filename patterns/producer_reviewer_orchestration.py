"""
Producer-Reviewer Orchestration Pattern - Iterative Quality Assurance

Establishes a closed-loop feedback system between content generation and quality control.
The Producer Agent focuses on content creation, while the Reviewer Agent handles quality 
assessment and error detection. Through multiple iterations, content quality is optimized.

Example: Legal Document Summarization
1. Producer Agent: Creates initial summary
2. Reviewer Agent: Checks accuracy, terminology, identifies issues
3. Iteration: If issues found, producer creates improved version
4. Convergence: Process continues until quality standards are met
"""
import asyncio
from typing import List, Any, Dict, Optional, Tuple
from enum import Enum
from agents.base_agent import BaseAgent, SimpleAgent


class ReviewStatus(Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class ReviewResult:
    """Represents the result of a review process."""
    
    def __init__(self, status: ReviewStatus, score: float, feedback: str, 
                 suggestions: List[str] = None, issues: List[str] = None):
        self.status = status
        self.score = score  # Quality score 0-100
        self.feedback = feedback
        self.suggestions = suggestions or []
        self.issues = issues or []


class ProducerReviewerOrchestrator:
    """Orchestrates iterative content improvement through producer-reviewer cycles."""
    
    def __init__(self, producer_agent: BaseAgent, reviewer_agent: BaseAgent, 
                 max_iterations: int = 5, quality_threshold: float = 80.0):
        self.producer_agent = producer_agent
        self.reviewer_agent = reviewer_agent
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.iteration_history = []
        self.total_execution_time = 0
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def _parse_review_result(self, review_text: str) -> ReviewResult:
        """Parse review agent output into structured ReviewResult."""
        try:
            import re
            import json
            
            # Try to extract JSON from review
            json_match = re.search(r'\{.*\}', review_text, re.DOTALL)
            if json_match:
                review_data = json.loads(json_match.group())
                
                status_map = {
                    "approved": ReviewStatus.APPROVED,
                    "needs_revision": ReviewStatus.NEEDS_REVISION,
                    "rejected": ReviewStatus.REJECTED
                }
                
                return ReviewResult(
                    status=status_map.get(review_data.get("status", "needs_revision"), ReviewStatus.NEEDS_REVISION),
                    score=float(review_data.get("score", 0)),
                    feedback=review_data.get("feedback", review_text),
                    suggestions=review_data.get("suggestions", []),
                    issues=review_data.get("issues", [])
                )
            
            # Fallback: parse text-based review
            score_match = re.search(r'score[:\s]*(\d+(?:\.\d+)?)', review_text.lower())
            score = float(score_match.group(1)) if score_match else 50.0
            
            if "approved" in review_text.lower() or "accept" in review_text.lower():
                status = ReviewStatus.APPROVED
            elif "reject" in review_text.lower():
                status = ReviewStatus.REJECTED
            else:
                status = ReviewStatus.NEEDS_REVISION
            
            return ReviewResult(
                status=status,
                score=score,
                feedback=review_text,
                suggestions=[],
                issues=[]
            )
            
        except Exception as e:
            # Default fallback
            return ReviewResult(
                status=ReviewStatus.NEEDS_REVISION,
                score=50.0,
                feedback=review_text,
                suggestions=[],
                issues=[f"Review parsing error: {str(e)}"]
            )
    
    async def _produce_content(self, task: str, context: Optional[Dict] = None, 
                              previous_feedback: Optional[ReviewResult] = None) -> str:
        """Generate or revise content using the producer agent."""
        
        if previous_feedback is None:
            # Initial production
            prompt = f"Task: {task}\nContext: {context or {}}\n\nPlease create high-quality content for this task."
        else:
            # Revision based on feedback
            prompt = f"""
            Original Task: {task}
            Context: {context or {}}
            
            Previous Review Feedback:
            - Status: {previous_feedback.status.value}
            - Score: {previous_feedback.score}/100
            - Feedback: {previous_feedback.feedback}
            - Issues to Address: {', '.join(previous_feedback.issues)}
            - Suggestions: {', '.join(previous_feedback.suggestions)}
            
            Please revise and improve the content based on this feedback. Address all identified issues and implement the suggestions where appropriate.
            """
        
        try:
            content = await self.producer_agent.process(prompt, context)
            
            # Update metrics
            metrics = self.producer_agent.get_metrics()
            self.total_execution_time += metrics["execution_time"]
            for key in self.total_tokens:
                self.total_tokens[key] += metrics["token_usage"][key]
            
            return content
            
        except Exception as e:
            raise Exception(f"Content production failed: {str(e)}")
    
    async def _review_content(self, content: str, task: str, context: Optional[Dict] = None) -> ReviewResult:
        """Review content using the reviewer agent."""
        
        review_prompt = f"""
        You are a quality reviewer. Please review the following content and provide detailed feedback.
        
        Original Task: {task}
        Context: {context or {}}
        
        Content to Review:
        {content}
        
        Please evaluate the content and provide your review in the following JSON format:
        {{
            "status": "approved|needs_revision|rejected",
            "score": [0-100 quality score],
            "feedback": "detailed feedback explanation",
            "issues": ["list", "of", "specific", "issues"],
            "suggestions": ["list", "of", "improvement", "suggestions"]
        }}
        
        Evaluation Criteria:
        - Accuracy and correctness
        - Completeness and thoroughness
        - Clarity and readability
        - Relevance to the task
        - Professional quality
        - Adherence to requirements
        """
        
        try:
            review_text = await self.reviewer_agent.process(review_prompt, context)
            
            # Update metrics
            metrics = self.reviewer_agent.get_metrics()
            self.total_execution_time += metrics["execution_time"]
            for key in self.total_tokens:
                self.total_tokens[key] += metrics["token_usage"][key]
            
            return self._parse_review_result(review_text)
            
        except Exception as e:
            return ReviewResult(
                status=ReviewStatus.NEEDS_REVISION,
                score=0.0,
                feedback=f"Review failed: {str(e)}",
                issues=[f"Review process error: {str(e)}"]
            )
    
    async def execute(self, task: str, context: Optional[Dict] = None) -> Dict:
        """Execute the producer-reviewer iterative process."""
        
        print("🚀 Starting Producer-Reviewer Orchestration")
        print("=" * 50)
        
        current_content = None
        previous_review = None
        
        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Production phase
            print("📝 Producer creating/revising content...")
            try:
                current_content = await self._produce_content(task, context, previous_review)
                print(f"   Content length: {len(current_content)} characters")
            except Exception as e:
                print(f"❌ Production failed: {str(e)}")
                break
            
            # Review phase
            print("🔍 Reviewer evaluating content...")
            current_review = await self._review_content(current_content, task, context)
            
            # Log iteration
            iteration_log = {
                "iteration": iteration + 1,
                "content_length": len(current_content),
                "review_status": current_review.status.value,
                "quality_score": current_review.score,
                "feedback": current_review.feedback[:200] + "..." if len(current_review.feedback) > 200 else current_review.feedback,
                "issues_count": len(current_review.issues),
                "suggestions_count": len(current_review.suggestions)
            }
            self.iteration_history.append(iteration_log)
            
            print(f"   Review Status: {current_review.status.value}")
            print(f"   Quality Score: {current_review.score}/100")
            print(f"   Issues Found: {len(current_review.issues)}")
            
            # Check termination conditions
            if current_review.status == ReviewStatus.APPROVED:
                print("✅ Content approved! Process completed successfully.")
                break
            elif current_review.status == ReviewStatus.REJECTED:
                print("❌ Content rejected. Process terminated.")
                break
            elif current_review.score >= self.quality_threshold:
                print(f"✅ Quality threshold ({self.quality_threshold}) reached! Process completed.")
                break
            
            # Prepare for next iteration
            previous_review = current_review
            
            if iteration == self.max_iterations - 1:
                print(f"⚠️  Maximum iterations ({self.max_iterations}) reached.")
        
        # Prepare final results
        final_status = "completed" if current_review.status == ReviewStatus.APPROVED or current_review.score >= self.quality_threshold else "terminated"
        
        return {
            "final_content": current_content,
            "final_review": {
                "status": current_review.status.value,
                "score": current_review.score,
                "feedback": current_review.feedback,
                "issues": current_review.issues,
                "suggestions": current_review.suggestions
            },
            "process_summary": {
                "total_iterations": len(self.iteration_history),
                "final_status": final_status,
                "quality_improvement": self.iteration_history[-1]["quality_score"] - self.iteration_history[0]["quality_score"] if len(self.iteration_history) > 1 else 0,
                "total_execution_time": self.total_execution_time,
                "total_tokens": self.total_tokens
            },
            "iteration_history": self.iteration_history
        }


class DocumentSummarizationSystem:
    """Example implementation: Legal Document Summarization with Quality Control."""
    
    @staticmethod
    def create_system(api_key: str) -> ProducerReviewerOrchestrator:
        """Create a document summarization system with quality review."""
        
        producer = SimpleAgent(
            name="Document Summarizer",
            role="Content Producer - Document Summarization",
            api_key=api_key,
            system_prompt="""You are an expert document summarizer specializing in legal and business documents.
            
            Your responsibilities:
            - Create clear, concise, and accurate summaries
            - Preserve key legal terms and important details
            - Maintain professional tone and structure
            - Include executive summary and key points
            - Ensure completeness while being concise
            
            When revising based on feedback:
            - Address all identified issues specifically
            - Implement suggested improvements
            - Maintain document integrity
            - Improve clarity and accuracy"""
        )
        
        reviewer = SimpleAgent(
            name="Legal Document Reviewer",
            role="Quality Reviewer - Legal Accuracy",
            api_key=api_key,
            system_prompt="""You are a legal document review specialist. Your job is to ensure accuracy, completeness, and quality.
            
            Review Criteria:
            - Legal accuracy and terminology correctness
            - Completeness of key information
            - Clarity and readability
            - Professional presentation
            - Adherence to legal document standards
            
            Scoring Guidelines:
            - 90-100: Excellent, ready for publication
            - 80-89: Good, minor improvements needed
            - 70-79: Acceptable, moderate revisions required
            - 60-69: Below standard, significant improvements needed
            - Below 60: Poor quality, major revisions or rejection
            
            Always provide specific, actionable feedback and suggestions."""
        )
        
        return ProducerReviewerOrchestrator(producer, reviewer, max_iterations=4, quality_threshold=85.0)


class ContentCreationSystem:
    """Example implementation: Marketing Content Creation with Editorial Review."""
    
    @staticmethod
    def create_system(api_key: str) -> ProducerReviewerOrchestrator:
        """Create a marketing content creation system with editorial review."""
        
        producer = SimpleAgent(
            name="Content Creator",
            role="Marketing Content Producer",
            api_key=api_key,
            system_prompt="""You are a creative marketing content writer specializing in engaging, persuasive content.
            
            Your expertise includes:
            - Compelling headlines and copy
            - Brand voice consistency
            - Audience-targeted messaging
            - SEO-friendly content
            - Call-to-action optimization
            
            When creating content:
            - Focus on audience engagement
            - Maintain brand consistency
            - Include clear value propositions
            - Use persuasive language techniques
            - Ensure content is actionable"""
        )
        
        reviewer = SimpleAgent(
            name="Editorial Reviewer",
            role="Content Quality and Brand Reviewer",
            api_key=api_key,
            system_prompt="""You are an editorial reviewer specializing in marketing content quality assurance.
            
            Review Focus Areas:
            - Brand voice and tone consistency
            - Message clarity and impact
            - Grammar and style correctness
            - Audience appropriateness
            - Marketing effectiveness
            - Legal and compliance considerations
            
            Scoring Guidelines:
            - 90-100: Publication ready, excellent quality
            - 80-89: Minor edits needed, good quality
            - 70-79: Moderate revisions required
            - 60-69: Significant improvements needed
            - Below 60: Major revisions or complete rewrite
            
            Provide specific, actionable feedback for improvements."""
        )
        
        return ProducerReviewerOrchestrator(producer, reviewer, max_iterations=3, quality_threshold=80.0)


class CodeReviewSystem:
    """Example implementation: Code Documentation with Technical Review."""
    
    @staticmethod
    def create_system(api_key: str) -> ProducerReviewerOrchestrator:
        """Create a code documentation system with technical review."""
        
        producer = SimpleAgent(
            name="Technical Writer",
            role="Code Documentation Producer",
            api_key=api_key,
            system_prompt="""You are a technical documentation specialist focusing on code documentation and API guides.
            
            Your expertise:
            - Clear technical explanations
            - Code examples and usage patterns
            - API documentation standards
            - Developer-friendly formatting
            - Comprehensive coverage of functionality
            
            Documentation standards:
            - Include code examples for all features
            - Provide clear parameter descriptions
            - Add error handling examples
            - Include performance considerations
            - Maintain consistent formatting"""
        )
        
        reviewer = SimpleAgent(
            name="Senior Developer",
            role="Technical Accuracy Reviewer",
            api_key=api_key,
            system_prompt="""You are a senior software developer reviewing technical documentation for accuracy and completeness.
            
            Review Criteria:
            - Technical accuracy and correctness
            - Code example validity
            - Completeness of coverage
            - Clarity for target audience
            - Best practices adherence
            - Security considerations
            
            Scoring Guidelines:
            - 90-100: Excellent, comprehensive documentation
            - 80-89: Good, minor technical corrections needed
            - 70-79: Adequate, some gaps to address
            - 60-69: Incomplete, significant improvements needed
            - Below 60: Poor quality, major revisions required
            
            Focus on technical accuracy and developer usability."""
        )
        
        return ProducerReviewerOrchestrator(producer, reviewer, max_iterations=4, quality_threshold=85.0)


# Example usage and demonstration
async def demo_producer_reviewer_orchestration(api_key: str, demo_type: str = "legal"):
    """Demonstrate producer-reviewer orchestration."""
    
    print("🚀 Producer-Reviewer Orchestration Demo")
    print("=" * 50)
    
    if demo_type == "legal":
        # Legal document summarization demo
        legal_document = """
        SOFTWARE LICENSE AGREEMENT
        
        This Software License Agreement ("Agreement") is entered into on [DATE] between TechCorp Inc., 
        a Delaware corporation ("Licensor"), and the entity or individual accepting this agreement ("Licensee").
        
        1. GRANT OF LICENSE
        Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee a 
        non-exclusive, non-transferable, revocable license to use the Software solely for Licensee's 
        internal business purposes.
        
        2. RESTRICTIONS
        Licensee shall not: (a) modify, adapt, or create derivative works of the Software; 
        (b) reverse engineer, decompile, or disassemble the Software; (c) distribute, sublicense, 
        or transfer the Software to any third party; (d) use the Software for any unlawful purpose.
        
        3. TERM AND TERMINATION
        This Agreement shall commence on the date of acceptance and continue until terminated. 
        Either party may terminate this Agreement at any time with 30 days written notice. 
        Upon termination, Licensee must cease all use of the Software and destroy all copies.
        
        4. WARRANTY DISCLAIMER
        THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. LICENSOR DISCLAIMS ALL 
        WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY 
        AND FITNESS FOR A PARTICULAR PURPOSE.
        
        5. LIMITATION OF LIABILITY
        IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL 
        DAMAGES ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT OR THE USE OF THE SOFTWARE.
        """
        
        system = DocumentSummarizationSystem.create_system(api_key)
        
        try:
            result = await system.execute(
                task=f"Create a comprehensive summary of this software license agreement: {legal_document}",
                context={
                    "document_type": "software_license",
                    "target_audience": "business_executives",
                    "summary_length": "detailed"
                }
            )
            
            print(f"\n✅ Document summarization process completed!")
            print(f"📊 Process Summary:")
            print(f"   - Total iterations: {result['process_summary']['total_iterations']}")
            print(f"   - Final status: {result['process_summary']['final_status']}")
            print(f"   - Quality improvement: +{result['process_summary']['quality_improvement']:.1f} points")
            print(f"   - Final score: {result['final_review']['score']}/100")
            print(f"   - Total execution time: {result['process_summary']['total_execution_time']:.2f} seconds")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    elif demo_type == "marketing":
        # Marketing content creation demo
        content_brief = """
        Create a landing page copy for a new AI-powered project management tool called "TaskFlow AI".
        
        Key Features:
        - Intelligent task prioritization
        - Automated deadline tracking
        - Team collaboration tools
        - Progress analytics and insights
        - Integration with popular tools
        
        Target Audience: Small to medium business owners and project managers
        Tone: Professional yet approachable, emphasizing efficiency and results
        Goal: Drive sign-ups for free trial
        """
        
        system = ContentCreationSystem.create_system(api_key)
        
        try:
            result = await system.execute(
                task=content_brief,
                context={
                    "content_type": "landing_page",
                    "brand_voice": "professional_approachable",
                    "cta_goal": "free_trial_signup"
                }
            )
            
            print(f"\n✅ Marketing content creation completed!")
            print(f"📊 Process Summary:")
            print(f"   - Total iterations: {result['process_summary']['total_iterations']}")
            print(f"   - Final status: {result['process_summary']['final_status']}")
            print(f"   - Final score: {result['final_review']['score']}/100")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    elif demo_type == "technical":
        # Technical documentation demo
        code_documentation_task = """
        Create comprehensive API documentation for a user authentication endpoint:
        
        Endpoint: POST /api/auth/login
        Purpose: Authenticate user credentials and return access token
        
        Parameters:
        - email (string, required): User's email address
        - password (string, required): User's password
        - remember_me (boolean, optional): Whether to extend session duration
        
        Returns:
        - Success: 200 OK with access_token, refresh_token, user_info
        - Error: 401 Unauthorized for invalid credentials
        - Error: 400 Bad Request for missing parameters
        
        Include code examples in JavaScript and Python.
        """
        
        system = CodeReviewSystem.create_system(api_key)
        
        try:
            result = await system.execute(
                task=code_documentation_task,
                context={
                    "documentation_type": "api_reference",
                    "target_developers": "full_stack",
                    "include_examples": True
                }
            )
            
            print(f"\n✅ Technical documentation completed!")
            print(f"📊 Process Summary:")
            print(f"   - Total iterations: {result['process_summary']['total_iterations']}")
            print(f"   - Final status: {result['process_summary']['final_status']}")
            print(f"   - Final score: {result['final_review']['score']}/100")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
