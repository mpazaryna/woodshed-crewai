from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain.agents import Tool
from langchain.utilities import GoogleSerperAPIWrapper

# Check our tools documentations for more information on how to use them
# from .tools.custom_serper_news_tool import CustomSerperNewsTool

search = GoogleSerperAPIWrapper()

# Create and assign the search tool to an agent
serper_tool = Tool(
    name="Intermediate Answer",
    func=search.run,
    description="Useful for search-based queries",
)


@CrewBase
class FirstCrew:
    """First crew for managing research and reporting tasks."""

    @agent
    def researcher(self) -> Agent:
        """
        Creates a researcher agent.

        This agent is configured to perform research tasks using the assigned tools.
        The tools include a search tool for retrieving information based on queries.

        Returns:
            Agent: An instance of the Agent configured for research tasks.
        """
        return Agent(
            config=self.agents_config["researcher"],
            tools=[serper_tool],
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
