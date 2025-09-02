# Multi-Agent Orchestration Playground

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/openai-1.0+-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive playground to explore and experiment with **5 different multi-agent orchestration patterns** for AI systems. Built with Streamlit and OpenAI GPT models.

> **Perfect for**: AI engineers, developers learning multi-agent systems, and anyone interested in agent coordination patterns

![Demo Screenshot]("Multi-Agent-Orchestration-Playground-09-02-2025_08_25_PM.png")

## Features

### 5 Orchestration Patterns Implemented

1. **🔄 Sequential Orchestration** - Pipeline Processing
   - Agents work in sequence, each building on previous output
   - Example: Report Generation System (Data Collection → Formatting → Analysis → Optimization → Delivery)

2. **⚡ MapReduce Orchestration** - Parallel Computing Intelligence
   - Breaks large tasks into independent chunks, processes in parallel, then combines results
   - Example: Document Summarization with parallel chunk processing

3. **🗳️ Consensus Orchestration** - Redundant Verification for Reliability
   - Multiple agents independently solve the same problem and vote on best solution
   - Example: Multi-perspective sentiment analysis with voting

4. **🏗️ Hierarchical Orchestration** - Specialized Division of Labor
   - Master orchestrator decomposes tasks and coordinates specialized agents
   - Example: Travel Planning System with flight, hotel, and activity specialists

5. **🔄 Producer-Reviewer Orchestration** - Iterative Quality Assurance
   - Producer creates content, reviewer provides feedback, iterates until quality standards met
   - Example: Legal document creation with review and revision cycles

### Interactive Features

- **Real-time Execution**: Watch agents collaborate in real-time
- **Metrics Dashboard**: Track execution time, token usage, success rates
- **Visualization**: Interactive charts showing agent performance and coordination
- **Configurable Examples**: Customize inputs and parameters for each pattern
- **Execution History**: Track your experimentation sessions

## Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- uv package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AIAnytime/Multi-Agents-Orchestration-Design-Patterns
   cd Multi-Agents-Orchestration-Design-Patterns
   ```

2. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

4. **Run the Streamlit app:**
   ```bash
   uv run streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

6. **Enter your OpenAI API key** in the sidebar

7. **Start exploring** the orchestration patterns!

### 🔑 API Key Setup

Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys) and enter it in the app's sidebar.

## Usage Guide

### Getting Started

1. **Configure API Key**: Enter your OpenAI API key in the sidebar
2. **Select Pattern**: Choose one of the 5 orchestration patterns
3. **Configure Parameters**: Adjust settings specific to each pattern
4. **Run Demo**: Click the "Run Demo" button to execute
5. **Analyze Results**: Review metrics, visualizations, and detailed outputs

### Pattern-Specific Examples

#### Sequential Orchestration
- **Use Case**: Report generation, content creation pipelines
- **Configuration**: Specify report topic
- **Output**: Comprehensive report with step-by-step processing log

#### MapReduce Orchestration
- **Use Cases**: Document summarization, sentiment analysis at scale
- **Configuration**: Choose between summarization or sentiment analysis
- **Output**: Parallel processing metrics and aggregated results

#### Consensus Orchestration
- **Use Cases**: Decision making, sentiment analysis with multiple perspectives
- **Configuration**: Choose between sentiment analysis or business decisions
- **Output**: Consensus decision with confidence scores and individual opinions

#### Hierarchical Orchestration
- **Use Cases**: Complex planning, project management
- **Configuration**: Choose between travel planning or project management
- **Output**: Coordinated results from specialized agents with task breakdown

#### Producer-Reviewer Orchestration
- **Use Cases**: Content quality assurance, iterative improvement
- **Configuration**: Choose between legal, marketing, or technical content
- **Output**: Final content with quality scores and iteration history

## Architecture

### Project Structure

```
multi-agent-playground/
├── agents/
│   └── base_agent.py          # Base agent class and utilities
├── patterns/
│   ├── sequential_orchestration.py      # Sequential pattern implementation
│   ├── mapreduce_orchestration.py       # MapReduce pattern implementation
│   ├── consensus_orchestration.py       # Consensus pattern implementation
│   ├── hierarchical_orchestration.py    # Hierarchical pattern implementation
│   └── producer_reviewer_orchestration.py # Producer-Reviewer pattern
├── app.py                     # Main Streamlit application
├── pyproject.toml            # Project dependencies and configuration
└── README.md                 # This file
```

### Key Components

- **BaseAgent**: Abstract base class for all agents with OpenAI integration
- **SimpleAgent**: Concrete implementation for basic agent functionality
- **Orchestrators**: Pattern-specific coordination logic
- **Streamlit App**: Interactive web interface with visualizations

## Technical Details

### Dependencies

- **streamlit**: Web application framework
- **openai**: OpenAI API client
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **asyncio**: Asynchronous programming support

### Agent Architecture

Each agent includes:
- **Role-based prompting**: Specialized system prompts for different functions
- **Metrics tracking**: Execution time and token usage monitoring
- **Error handling**: Robust error management and recovery
- **Context awareness**: Ability to use context from previous agents

### Orchestration Patterns

Each pattern implements:
- **Task decomposition**: Breaking complex tasks into manageable parts
- **Agent coordination**: Managing dependencies and execution order
- **Result aggregation**: Combining outputs into final results
- **Performance monitoring**: Tracking metrics and success rates

## Use Cases

### Business Applications

- **Content Creation**: Automated report generation with quality assurance
- **Decision Support**: Multi-perspective analysis for business decisions
- **Document Processing**: Large-scale document analysis and summarization
- **Project Planning**: Complex project decomposition and resource allocation

### Educational Applications

- **AI Learning**: Understanding multi-agent system design patterns
- **Experimentation**: Testing different coordination strategies
- **Research**: Exploring agent collaboration effectiveness
- **Prototyping**: Rapid development of multi-agent solutions

## Experimentation Tips

### Performance Optimization

1. **Token Management**: Monitor token usage across patterns
2. **Parallel Efficiency**: Compare sequential vs parallel processing
3. **Quality vs Speed**: Balance iteration count with execution time
4. **Agent Specialization**: Test different agent role configurations

### Pattern Selection Guidelines

- **Sequential**: Use for step-by-step processes with dependencies
- **MapReduce**: Use for large-scale, parallelizable tasks
- **Consensus**: Use when reliability and accuracy are critical
- **Hierarchical**: Use for complex, multi-domain problems
- **Producer-Reviewer**: Use when quality assurance is essential

## Customization

### Adding New Patterns

1. Create new pattern file in `patterns/` directory
2. Implement orchestrator class with `execute()` method
3. Add pattern to Streamlit app configuration
4. Include pattern-specific visualizations

### Extending Agents

1. Inherit from `BaseAgent` class
2. Implement `process()` method
3. Add specialized prompts and logic
4. Include metrics tracking

### Custom Examples

1. Modify existing example systems
2. Create new agent configurations
3. Add domain-specific prompts
4. Implement custom evaluation metrics

## Metrics and Analytics

### Execution Metrics

- **Execution Time**: Total time for pattern completion
- **Token Usage**: OpenAI API token consumption
- **Success Rate**: Percentage of successful agent executions
- **Agent Count**: Number of agents involved in execution

### Quality Metrics

- **Consensus Confidence**: Agreement level in consensus patterns
- **Quality Scores**: Iterative improvement in producer-reviewer patterns
- **Task Completion**: Success rate of hierarchical task execution
- **Parallel Efficiency**: Speedup achieved through parallel processing

## Security and Best Practices

### API Key Management

- Store API keys securely (use environment variables in production)
- Never commit API keys to version control
- Use API key rotation for production deployments
- Monitor API usage and costs

### Error Handling

- Implement graceful degradation for API failures
- Add retry logic for transient errors
- Provide meaningful error messages to users
- Log errors for debugging and monitoring

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork:**
   ```bash
   git clone https://github.com/yourusername/multi-agent-playground.git
   cd multi-agent-playground
   ```
3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies:**
   ```bash
   uv sync
   ```
5. **Make your changes and test thoroughly**
6. **Submit a pull request** with detailed description

### Code Style

- Follow Python PEP 8 guidelines
- Use type hints for better code clarity
- Add docstrings for all classes and methods
- Include unit tests for new functionality

### Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue!
- **Feature Requests**: Have an idea? We'd love to hear it!
- **Documentation**: Help improve our docs
- **Code**: Add new patterns or improve existing ones
- **UI/UX**: Enhance the Streamlit interface

## License

This project is open source and available under the MIT License.

## Support

### Common Issues

1. **API Key Errors**: Ensure valid OpenAI API key with sufficient credits
2. **Import Errors**: Run `uv sync` to install all dependencies
3. **Performance Issues**: Monitor token usage and adjust parameters
4. **Visualization Problems**: Check browser compatibility with Plotly

### Getting Help

- Check the execution logs in the Streamlit interface
- Review the detailed results in the expandable sections
- Monitor the execution history for patterns
- Adjust configuration parameters based on your use case
- [Open an issue](https://github.com/yourusername/multi-agent-playground/issues) for bugs or questions
- [Start a discussion](https://github.com/yourusername/multi-agent-playground/discussions) for general questions

### Planned Features

- **Custom Agent Templates**: Pre-built agents for common use cases
- **Pattern Comparison**: Side-by-side pattern performance analysis
- **Export Functionality**: Save results and configurations
- **Advanced Visualizations**: More detailed performance analytics
- **Integration Examples**: Real-world application templates

### Research Opportunities

- **Hybrid Patterns**: Combining multiple orchestration approaches
- **Dynamic Orchestration**: Adaptive pattern selection based on task characteristics
- **Performance Optimization**: Automated parameter tuning
- **Scalability Studies**: Large-scale multi-agent system analysis

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multi_agent_orchestration_playground,
  title={Multi-Agent Orchestration Playground},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/multi-agent-playground}
}
```

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

**Happy Experimenting!** 🎉

*Explore the fascinating world of multi-agent orchestration and discover how different coordination patterns can solve complex AI challenges.*

</div>
