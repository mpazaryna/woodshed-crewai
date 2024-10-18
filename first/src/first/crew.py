from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper

from .tools.custom_serper_news_tool import CustomSerperNewsTool

# Initialize tools
search = GoogleSerperAPIWrapper()
serper_tool = Tool(
    name="Intermediate Answer",
    func=search.run,
    description="Useful for search-based queries",
)
custom_news_tool = CustomSerperNewsTool()


@CrewBase
class FirstCrew:
    """First crew for managing research and reporting tasks."""

    def __init__(self, use_custom_news=True):
        """
        Initialize the FirstCrew instance with the specified research tool.

        This constructor allows for easy switching between two different research tools:
        the CustomSerperNewsTool and the default GoogleSerperAPIWrapper tool.

        Parameters:
        -----------
        use_custom_news : bool, optional (default=True)
            A flag to determine which research tool to use.
            If True, the CustomSerperNewsTool will be used.
            If False, the default GoogleSerperAPIWrapper tool will be used.

        Attributes:
        -----------
        research_tool : Tool
            The selected research tool that will be used by the researcher agent.
            This is either the custom_news_tool or the serper_tool, depending on
            the value of use_custom_news.

        Notes:
        ------
        - The custom_news_tool is an instance of CustomSerperNewsTool, which is
          assumed to be a custom implementation for news-related searches.
        - The serper_tool is an instance of the Tool class, initialized with
          GoogleSerperAPIWrapper, which is a general-purpose search tool.
        - This initialization allows for easy swapping between tools without
          modifying the code of the researcher agent or other parts of the class.

        Example:
        --------
        # To use the CustomSerperNewsTool (default behavior):
        crew = FirstCrew()

        # To use the default GoogleSerperAPIWrapper tool:
        crew = FirstCrew(use_custom_news=False)
        """
        self.research_tool = custom_news_tool if use_custom_news else serper_tool

    @agent
    def researcher(self) -> Agent:
        """
        Creates a researcher agent with the specified tool.

        Returns:
            Agent: An instance of the Agent configured for research tasks.
        """
        return Agent(
            config=self.agents_config["researcher"],
            tools=[self.research_tool],
            verbose=True,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        """
        Creates a reporting analyst agent.

        This agent is responsible for analyzing data and generating reports.
        It is configured based on the settings defined in the agents_config.

        Returns:
            Agent: An instance of the Agent configured for reporting analysis.
        """
        return Agent(config=self.agents_config["reporting_analyst"], verbose=True)

    @task
    def research_task(self) -> Task:
        """
        Defines a research task.

        This task is configured to perform research activities as specified in the tasks_config.

        Returns:
            Task: An instance of the Task configured for research activities.
        """
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def reporting_task(self) -> Task:
        """
        Defines a reporting task.

        This task is responsible for generating reports based on the analysis performed by the reporting analyst.
        The output is saved to a specified file.

        Returns:
            Task: An instance of the Task configured for reporting activities.
        """
        return Task(config=self.tasks_config["reporting_task"], output_file="report.md")

    @crew
    def crew(self) -> Crew:
        """
        Creates the First crew.

        This method initializes the crew with the defined agents and tasks.
        The process can be set to sequential or hierarchical based on the requirements.

        Returns:
            Crew: An instance of the Crew containing the defined agents and tasks.
        """
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
