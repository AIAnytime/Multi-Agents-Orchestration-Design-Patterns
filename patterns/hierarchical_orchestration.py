"""
Hierarchical Orchestration Pattern - Specialized Division of Labor System

Establishes a clear management hierarchy with an orchestration agent responsible for 
task understanding, decomposition, and scheduling, while specialized agents handle 
specific execution. Can handle complex, cross-domain problems.

Example: Intelligent Itinerary Planning System
1. Main Orchestrator: Analyzes user needs and decomposes into sub-tasks
2. Specialized Agents: Flight search, hotel reservations, activity recommendations
3. Coordination: Main orchestrator manages scheduling and integration
4. Final Integration: Combines all results into comprehensive plan
"""
import asyncio
from typing import List, Any, Dict, Optional, Tuple
from enum import Enum
from agents.base_agent import BaseAgent, SimpleAgent


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    """Represents a task in the hierarchical system."""
    
    def __init__(self, task_id: str, task_type: str, description: str, 
                 input_data: Any, priority: int = 1, dependencies: List[str] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.input_data = input_data
        self.priority = priority
        self.dependencies = dependencies or []
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.assigned_agent = None
        self.execution_time = 0


class HierarchicalOrchestrator:
    """Main orchestrator that manages task decomposition and agent coordination."""
    
    def __init__(self, orchestrator_agent: BaseAgent, specialized_agents: Dict[str, BaseAgent]):
        self.orchestrator_agent = orchestrator_agent
        self.specialized_agents = specialized_agents
        self.tasks = {}
        self.execution_log = []
        self.total_execution_time = 0
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    async def _decompose_task(self, main_task: str, context: Optional[Dict] = None) -> List[Task]:
        """Use orchestrator agent to decompose main task into subtasks."""
        print("🧠 Orchestrator analyzing and decomposing task...")
        
        decomposition_prompt = f"""
        Main Task: {main_task}
        Context: {context or {}}
        
        Available Specialized Agents: {list(self.specialized_agents.keys())}
        
        Please decompose this main task into specific subtasks that can be handled by the available specialized agents.
        For each subtask, specify:
        1. Task ID (unique identifier)
        2. Task Type (which specialized agent should handle it)
        3. Description (what needs to be done)
        4. Priority (1-5, where 5 is highest)
        5. Dependencies (which other tasks must complete first)
        
        Format your response as JSON with this structure:
        {{
            "subtasks": [
                {{
                    "task_id": "task_1",
                    "task_type": "agent_type",
                    "description": "detailed description",
                    "priority": 3,
                    "dependencies": ["task_0"]
                }}
            ]
        }}
        """
        
        try:
            response = await self.orchestrator_agent.process(decomposition_prompt, context)
            
            # Update metrics
            metrics = self.orchestrator_agent.get_metrics()
            self.total_execution_time += metrics["execution_time"]
            for key in self.total_tokens:
                self.total_tokens[key] += metrics["token_usage"][key]
            
            # Parse response (simplified - in production, use proper JSON parsing with error handling)
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                task_data = json.loads(json_match.group())
                
                tasks = []
                for subtask in task_data.get("subtasks", []):
                    task = Task(
                        task_id=subtask["task_id"],
                        task_type=subtask["task_type"],
                        description=subtask["description"],
                        input_data=subtask["description"],  # Using description as input for simplicity
                        priority=subtask.get("priority", 1),
                        dependencies=subtask.get("dependencies", [])
                    )
                    tasks.append(task)
                    self.tasks[task.task_id] = task
                
                return tasks
            else:
                # Fallback: create simple tasks based on available agents
                tasks = []
                for i, (agent_type, agent) in enumerate(self.specialized_agents.items()):
                    task = Task(
                        task_id=f"task_{i}",
                        task_type=agent_type,
                        description=f"Handle {agent_type} aspects of: {main_task}",
                        input_data=main_task,
                        priority=1
                    )
                    tasks.append(task)
                    self.tasks[task.task_id] = task
                
                return tasks
                
        except Exception as e:
            print(f"❌ Task decomposition failed: {str(e)}")
            # Create fallback tasks
            tasks = []
            for i, (agent_type, agent) in enumerate(self.specialized_agents.items()):
                task = Task(
                    task_id=f"fallback_task_{i}",
                    task_type=agent_type,
                    description=f"Handle {agent_type} for: {main_task}",
                    input_data=main_task,
                    priority=1
                )
                tasks.append(task)
                self.tasks[task.task_id] = task
            
            return tasks
    
    def _get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_met = all(
                    self.tasks.get(dep_id, Task("", "", "", "")).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                
                if dependencies_met:
                    ready_tasks.append(task)
        
        # Sort by priority (highest first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ready_tasks
    
    async def _execute_task(self, task: Task, context: Optional[Dict] = None) -> bool:
        """Execute a single task using the appropriate specialized agent."""
        if task.task_type not in self.specialized_agents:
            task.status = TaskStatus.FAILED
            task.error = f"No agent available for task type: {task.task_type}"
            return False
        
        agent = self.specialized_agents[task.task_type]
        task.assigned_agent = agent.name
        task.status = TaskStatus.IN_PROGRESS
        
        print(f"🔧 Executing {task.task_id} with {agent.name}")
        
        try:
            # Prepare context with completed task results
            enhanced_context = context.copy() if context else {}
            enhanced_context["completed_tasks"] = {
                tid: t.result for tid, t in self.tasks.items() 
                if t.status == TaskStatus.COMPLETED
            }
            enhanced_context["task_description"] = task.description
            
            result = await agent.process(task.input_data, enhanced_context)
            
            # Update metrics
            metrics = agent.get_metrics()
            task.execution_time = metrics["execution_time"]
            self.total_execution_time += metrics["execution_time"]
            for key in self.total_tokens:
                self.total_tokens[key] += metrics["token_usage"][key]
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            self.execution_log.append({
                "task_id": task.task_id,
                "agent": agent.name,
                "status": "completed",
                "execution_time": task.execution_time,
                "result_preview": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
            })
            
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            self.execution_log.append({
                "task_id": task.task_id,
                "agent": agent.name,
                "status": "failed",
                "error": str(e)
            })
            
            print(f"❌ Task {task.task_id} failed: {str(e)}")
            return False
    
    async def _integrate_results(self, context: Optional[Dict] = None) -> str:
        """Use orchestrator to integrate all completed task results."""
        print("🔗 Integrating results from all completed tasks...")
        
        completed_tasks = {
            tid: task for tid, task in self.tasks.items() 
            if task.status == TaskStatus.COMPLETED
        }
        
        if not completed_tasks:
            return "No tasks completed successfully."
        
        integration_prompt = f"""
        You are the main orchestrator. Please integrate the results from all completed subtasks into a comprehensive final result.
        
        Completed Tasks and Results:
        {chr(10).join([f"- {task.description}: {task.result}" for task in completed_tasks.values()])}
        
        Context: {context or {}}
        
        Please provide a well-structured, comprehensive integration of all these results that addresses the original main task.
        """
        
        try:
            integrated_result = await self.orchestrator_agent.process(integration_prompt, context)
            
            # Update metrics
            metrics = self.orchestrator_agent.get_metrics()
            self.total_execution_time += metrics["execution_time"]
            for key in self.total_tokens:
                self.total_tokens[key] += metrics["token_usage"][key]
            
            return integrated_result
            
        except Exception as e:
            return f"Integration failed: {str(e)}"
    
    async def execute(self, main_task: str, context: Optional[Dict] = None) -> Dict:
        """Execute the complete hierarchical orchestration."""
        print("🚀 Starting Hierarchical Orchestration")
        print("=" * 50)
        
        try:
            # Step 1: Decompose main task
            subtasks = await self._decompose_task(main_task, context)
            print(f"📋 Decomposed into {len(subtasks)} subtasks")
            
            # Step 2: Execute tasks based on dependencies and priorities
            max_iterations = len(subtasks) * 2  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                ready_tasks = self._get_ready_tasks()
                
                if not ready_tasks:
                    # Check if all tasks are completed or failed
                    pending_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
                    if not pending_tasks:
                        break
                    else:
                        # Handle circular dependencies or other issues
                        print("⚠️  No ready tasks but pending tasks exist. Executing remaining tasks.")
                        for task in pending_tasks:
                            await self._execute_task(task, context)
                        break
                
                # Execute ready tasks (could be done in parallel for better performance)
                for task in ready_tasks:
                    await self._execute_task(task, context)
                
                iteration += 1
            
            # Step 3: Integrate results
            final_result = await self._integrate_results(context)
            
            # Prepare summary
            completed_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
            
            return {
                "final_result": final_result,
                "task_summary": {
                    "total_tasks": len(self.tasks),
                    "completed": completed_count,
                    "failed": failed_count,
                    "success_rate": completed_count / len(self.tasks) if self.tasks else 0
                },
                "execution_log": self.execution_log,
                "total_execution_time": self.total_execution_time,
                "total_tokens": self.total_tokens,
                "task_details": {tid: {
                    "description": task.description,
                    "status": task.status.value,
                    "agent": task.assigned_agent,
                    "execution_time": task.execution_time,
                    "result": task.result
                } for tid, task in self.tasks.items()}
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }


class TravelPlanningSystem:
    """Example implementation: Intelligent Itinerary Planning System."""
    
    @staticmethod
    def create_system(api_key: str) -> HierarchicalOrchestrator:
        """Create a travel planning hierarchical system."""
        
        # Main orchestrator agent
        orchestrator = SimpleAgent(
            name="Travel Coordinator",
            role="Main Travel Planning Orchestrator",
            api_key=api_key,
            system_prompt="""You are a master travel coordinator. Your job is to analyze travel requests and coordinate with specialized agents.
            
            Capabilities:
            - Decompose complex travel requests into specific subtasks
            - Coordinate between different travel service specialists
            - Integrate all travel components into comprehensive itineraries
            - Handle scheduling conflicts and optimize travel plans
            
            When decomposing tasks, consider: transportation, accommodation, activities, dining, and logistics."""
        )
        
        # Specialized agents
        specialized_agents = {
            "flight_search": SimpleAgent(
                name="Flight Specialist",
                role="Flight Search and Booking",
                api_key=api_key,
                system_prompt="""You are a flight search specialist. Find and recommend the best flight options.
                
                Consider:
                - Flight times and connections
                - Price ranges and value
                - Airline preferences and reliability
                - Baggage policies
                - Seat availability and upgrades
                
                Provide detailed flight recommendations with alternatives."""
            ),
            
            "hotel_booking": SimpleAgent(
                name="Accommodation Expert",
                role="Hotel and Accommodation Booking",
                api_key=api_key,
                system_prompt="""You are an accommodation specialist. Find and recommend hotels and lodging options.
                
                Consider:
                - Location and proximity to attractions
                - Price range and value for money
                - Amenities and services
                - Guest reviews and ratings
                - Cancellation policies
                
                Provide detailed accommodation recommendations with alternatives."""
            ),
            
            "activity_planning": SimpleAgent(
                name="Activity Coordinator",
                role="Activities and Attractions Planning",
                api_key=api_key,
                system_prompt="""You are an activities and attractions specialist. Plan engaging activities and sightseeing.
                
                Consider:
                - Popular attractions and hidden gems
                - Activity duration and scheduling
                - Age-appropriate and interest-based activities
                - Seasonal availability
                - Booking requirements and costs
                
                Create detailed activity itineraries with timing and logistics."""
            ),
            
            "dining_recommendations": SimpleAgent(
                name="Culinary Guide",
                role="Dining and Restaurant Recommendations",
                api_key=api_key,
                system_prompt="""You are a culinary specialist. Recommend restaurants and dining experiences.
                
                Consider:
                - Local cuisine and specialties
                - Dietary restrictions and preferences
                - Price ranges and value
                - Restaurant atmosphere and style
                - Reservation requirements
                
                Provide diverse dining recommendations for different meals and occasions."""
            ),
            
            "transportation": SimpleAgent(
                name="Transport Coordinator",
                role="Local Transportation Planning",
                api_key=api_key,
                system_prompt="""You are a local transportation specialist. Plan efficient local transportation.
                
                Consider:
                - Public transportation options
                - Car rental and ride-sharing
                - Walking distances and routes
                - Transportation costs
                - Accessibility requirements
                
                Create comprehensive transportation plans connecting all activities."""
            )
        }
        
        return HierarchicalOrchestrator(orchestrator, specialized_agents)


class ProjectManagementSystem:
    """Example implementation: Software Project Management System."""
    
    @staticmethod
    def create_system(api_key: str) -> HierarchicalOrchestrator:
        """Create a project management hierarchical system."""
        
        orchestrator = SimpleAgent(
            name="Project Manager",
            role="Software Project Orchestrator",
            api_key=api_key,
            system_prompt="""You are a senior project manager for software development projects.
            
            Responsibilities:
            - Break down project requirements into manageable tasks
            - Coordinate between different development specialists
            - Manage dependencies and timelines
            - Integrate deliverables into final project plan
            - Risk assessment and mitigation planning"""
        )
        
        specialized_agents = {
            "requirements_analysis": SimpleAgent(
                name="Business Analyst",
                role="Requirements Analysis",
                api_key=api_key,
                system_prompt="""You are a business analyst specializing in requirements gathering and analysis.
                
                Focus on:
                - Functional and non-functional requirements
                - User stories and acceptance criteria
                - Stakeholder needs analysis
                - Requirements prioritization
                - Risk identification"""
            ),
            
            "technical_design": SimpleAgent(
                name="Solution Architect",
                role="Technical Design and Architecture",
                api_key=api_key,
                system_prompt="""You are a solution architect responsible for technical design.
                
                Focus on:
                - System architecture and design patterns
                - Technology stack recommendations
                - Scalability and performance considerations
                - Integration requirements
                - Technical risk assessment"""
            ),
            
            "development_planning": SimpleAgent(
                name="Development Lead",
                role="Development Planning and Estimation",
                api_key=api_key,
                system_prompt="""You are a development lead responsible for planning and estimation.
                
                Focus on:
                - Development task breakdown
                - Effort estimation and timeline
                - Resource allocation
                - Development methodology recommendations
                - Quality assurance planning"""
            ),
            
            "testing_strategy": SimpleAgent(
                name="QA Manager",
                role="Testing Strategy and Quality Assurance",
                api_key=api_key,
                system_prompt="""You are a QA manager responsible for testing strategy.
                
                Focus on:
                - Test planning and strategy
                - Test case design and coverage
                - Automation opportunities
                - Quality metrics and KPIs
                - Risk-based testing approach"""
            )
        }
        
        return HierarchicalOrchestrator(orchestrator, specialized_agents)


# Example usage and demonstration
async def demo_hierarchical_orchestration(api_key: str, demo_type: str = "travel"):
    """Demonstrate hierarchical orchestration."""
    
    print("🚀 Hierarchical Orchestration Demo")
    print("=" * 45)
    
    if demo_type == "travel":
        # Travel planning demo
        travel_request = """
        Plan a 5-day vacation to Tokyo, Japan for 2 adults in March 2024.
        Budget: $3000 per person. Interests: technology, traditional culture, food.
        Preferences: Modern hotels, mix of popular and off-the-beaten-path experiences.
        Flying from New York City.
        """
        
        system = TravelPlanningSystem.create_system(api_key)
        
        try:
            result = await system.execute(
                main_task=travel_request,
                context={
                    "travel_type": "leisure",
                    "group_size": 2,
                    "duration": "5 days",
                    "season": "spring"
                }
            )
            
            if result.get("error"):
                print(f"❌ Travel planning failed: {result['error']}")
                return None
            
            print(f"\n✅ Travel planning completed!")
            print(f"📊 Task Summary:")
            print(f"   - Total tasks: {result['task_summary']['total_tasks']}")
            print(f"   - Completed: {result['task_summary']['completed']}")
            print(f"   - Success rate: {result['task_summary']['success_rate']:.1%}")
            print(f"   - Total execution time: {result['total_execution_time']:.2f} seconds")
            print(f"   - Total tokens: {result['total_tokens']['total_tokens']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    elif demo_type == "project":
        # Project management demo
        project_request = """
        Plan the development of a mobile e-commerce application with the following requirements:
        - iOS and Android native apps
        - User authentication and profiles
        - Product catalog and search
        - Shopping cart and checkout
        - Payment integration
        - Order tracking
        - Admin dashboard
        Timeline: 6 months, Team: 8 developers, Budget: $500,000
        """
        
        system = ProjectManagementSystem.create_system(api_key)
        
        try:
            result = await system.execute(
                main_task=project_request,
                context={
                    "project_type": "mobile_development",
                    "timeline": "6 months",
                    "team_size": 8,
                    "methodology": "agile"
                }
            )
            
            if result.get("error"):
                print(f"❌ Project planning failed: {result['error']}")
                return None
            
            print(f"\n✅ Project planning completed!")
            print(f"📊 Task Summary:")
            print(f"   - Total tasks: {result['task_summary']['total_tasks']}")
            print(f"   - Completed: {result['task_summary']['completed']}")
            print(f"   - Success rate: {result['task_summary']['success_rate']:.1%}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
