from crewai import Task
from tools import yt_tool
from agents import blog_researcher,blog_writer

#Research Task
research_task=Task(
    description=(
        "Identify the video {topic}."
        "Get detailed information about the video from the channel"
    ),
    expected_output="A comprehensive 3 paragraphs long report based on the {topic} of video of content",
    tools=[yt_tool],
    agent=blog_researcher

)

#writing task with language model configuration
write_task=Task(
    description=(
        "get the info from the youtube channel on the topic {topic} and create the content for the blog"
    ),
    expected_output="Summarize the info from the youtube channel video on the topic{topic}.",
    tool=[yt_tool],
    agent=blog_writer,
    async_execution=False,
    output_Files='new-blog-post.md'    #example of output customization
)
