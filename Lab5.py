import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
API_KEY = os.getenv('NGU_API_KEY')
BASE_URL = os.getenv('NGU_BASE_URL')
LLM_MODEL = os.getenv('NGU_MODEL')
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(messages, tools=None, tool_choice=None):
    kwargs = {"model": LLM_MODEL, "messages": messages}
    if tools: kwargs["tools"] = tools
    if tool_choice: kwargs["tool_choice"] = tool_choice
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        print("LLM Error:", e)
        return None

def get_sample_blog_post():
    try:
        with open('sample_blog_post.json') as f:
            return json.load(f)
    except Exception as e:
        print("File error:", e)
        return None

# Tool schemas
extract_key_points_schema = {
    "type": "function",
    "function": {
        "name": "extract_key_points",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["key_points"]
        }
    }
}

generate_summary_schema = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"]
        }
    }
}

create_social_media_posts_schema = {
    "type": "function",
    "function": {
        "name": "create_social_media_posts",
        "parameters": {
            "type": "object",
            "properties": {
                "twitter": {"type": "string"},
                "linkedin": {"type": "string"},
                "facebook": {"type": "string"}
            },
            "required": ["twitter", "linkedin", "facebook"]
        }
    }
}

create_email_newsletter_schema = {
    "type": "function",
    "function": {
        "name": "create_email_newsletter",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["subject", "body"]
        }
    }
}

# === Task Functions ===
def task_extract_key_points(blog_post):
    messages = [
        {"role": "system", "content": "Extract key points from blog posts."},
        {"role": "user", "content": f"Title: {blog_post['title']}\nContent: {blog_post['content']}"}
    ]
    res = call_llm(messages, [extract_key_points_schema], {"type": "function", "function": {"name": "extract_key_points"}})
    if res and res.choices[0].message.tool_calls:
        return json.loads(res.choices[0].message.tool_calls[0].function.arguments).get("key_points", [])
    return []

def task_generate_summary(key_points):
    messages = [
        {"role": "system", "content": "Summarize given points."},
        {"role": "user", "content": "Summarize:\n" + "\n".join(f"- {kp}" for kp in key_points)}
    ]
    res = call_llm(messages, [generate_summary_schema], {"type": "function", "function": {"name": "generate_summary"}})
    if res and res.choices[0].message.tool_calls:
        return json.loads(res.choices[0].message.tool_calls[0].function.arguments).get("summary", "")
    return ""

def task_create_social_media_posts(key_points, blog_title):
    messages = [
        {"role": "system", "content": "Create platform-specific social media posts."},
        {"role": "user", "content": f"Title: {blog_title}\n" + "\n".join(f"- {kp}" for kp in key_points)}
    ]
    res = call_llm(messages, [create_social_media_posts_schema], {"type": "function", "function": {"name": "create_social_media_posts"}})
    if res and res.choices[0].message.tool_calls:
        return json.loads(res.choices[0].message.tool_calls[0].function.arguments)
    return {}

def task_create_email_newsletter(blog_post, summary, key_points):
    messages = [
        {"role": "system", "content": "Write a newsletter email."},
        {"role": "user", "content": f"Title: {blog_post['title']}\nSummary: {summary}\nKey Points:\n" + "\n".join(f"- {kp}" for kp in key_points)}
    ]
    res = call_llm(messages, [create_email_newsletter_schema], {"type": "function", "function": {"name": "create_email_newsletter"}})
    if res and res.choices[0].message.tool_calls:
        return json.loads(res.choices[0].message.tool_calls[0].function.arguments)
    return {}

# === Reflexion System ===
def evaluate_content(content, content_type):
    messages = [
        {"role": "system", "content": "Evaluate quality and give feedback."},
        {"role": "user", "content": f"Evaluate this {content_type}:\n{content}"}
    ]
    res = call_llm(messages)
    if res:
        feedback = res.choices[0].message.content
        return {"quality_score": 0.9 if "good" in feedback else 0.6, "feedback": feedback}
    return {"quality_score": 0.5, "feedback": "No evaluation"}

def improve_content(content, feedback, content_type):
    messages = [
        {"role": "system", "content": "Improve content based on feedback."},
        {"role": "user", "content": f"Feedback: {feedback}\nContent: {content}"}
    ]
    res = call_llm(messages)
    return res.choices[0].message.content if res else content

def generate_with_reflexion(generator_func, max_attempts=3):
    def wrapped(*args, **kwargs):
        content_type = kwargs.pop("content_type", "content")
        content = generator_func(*args, **kwargs)
        for _ in range(max_attempts):
            eval_result = evaluate_content(content, content_type)
            if eval_result["quality_score"] >= 0.8:
                return content
            content = improve_content(content, eval_result["feedback"], content_type)
        return content
    return wrapped

# === Workflows ===
def run_workflow_with_reflexion(blog_post):
    key_points = task_extract_key_points(blog_post)
    summary = generate_with_reflexion(task_generate_summary)(key_points, content_type="summary")
    sm_posts = generate_with_reflexion(lambda k, t: task_create_social_media_posts(k, t))(key_points, blog_post["title"], content_type="social_media_post")
    email = generate_with_reflexion(lambda bp, s, k: task_create_email_newsletter(bp, s, k))(blog_post, summary, key_points, content_type="email")
    return {"summary": summary, "social_media": sm_posts, "email": email}

# === Agent Workflow ===
def define_agent_tools():
    return [
        extract_key_points_schema,
        generate_summary_schema,
        create_social_media_posts_schema,
        create_email_newsletter_schema,
        {
            "type": "function",
            "function": {
                "name": "finish",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "social_posts": {"type": "object"},
                        "email": {"type": "object"}
                    },
                    "required": ["summary", "social_posts", "email"]
                }
            }
        }
    ]

def execute_agent_tool(tool_name, arguments):
    blog_post = get_sample_blog_post()

    if tool_name == "extract_key_points":
        return {"key_points": task_extract_key_points(blog_post)}

    elif tool_name == "generate_summary":
        key_points = arguments.get("key_points")
        if not key_points:
            print("⚠️ Warning: key_points not provided, extracting again.")
            key_points = task_extract_key_points(blog_post)
        return {"summary": task_generate_summary(key_points)}

    elif tool_name == "create_social_media_posts":
        key_points = arguments.get("key_points")
        if not key_points:
            print("⚠️ Warning: key_points not provided for social posts, extracting again.")
            key_points = task_extract_key_points(blog_post)
        return task_create_social_media_posts(key_points, blog_post["title"])

    elif tool_name == "create_email_newsletter":
        key_points = arguments.get("key_points")
        summary = arguments.get("summary")
        if not key_points:
            print("⚠️ Warning: key_points missing, re-extracting.")
            key_points = task_extract_key_points(blog_post)
        if not summary:
            print("⚠️ Warning: summary missing, regenerating.")
            summary = task_generate_summary(key_points)
        return task_create_email_newsletter(blog_post, summary, key_points)

    return {}

def run_agent_workflow(blog_post):
    tools = define_agent_tools()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Content Repurposing Agent. Your job is to take a blog post and repurpose it into:\n"
                "1. Extracted key points\n"
                "2. A concise summary\n"
                "3. Social media posts\n"
                "4. An email newsletter\n"
                "Use the tools provided, and when you're done, call the 'finish' tool with all the final results."
            )
        },
        {
            "role": "user",
            "content": f"Blog:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"
        }
    ]

    results = {"summary": None, "social_posts": None, "email": None}
    max_iterations = 20

    for i in range(max_iterations):
        print(f"\n🔁 Agent Step {i+1}")

        # Print conversation so far
        print("\n🧠 Current Conversation:")
        for msg in messages:
            try:
                if isinstance(msg, dict):
                    role = msg.get("role", "UNKNOWN").upper()
                    name = msg.get("name", "")
                    content = msg.get("content", "")
                    if role == "TOOL":
                        print(f"TOOL ({name}): {content[:150]}...\n")
                    else:
                        print(f"{role}: {content[:150]}...\n")
                else:
                    print(f"{msg.role.upper()}: {msg.content[:150]}...\n")
            except Exception as e:
                print(f"⚠️ Error printing message: {e}")

        # Call the LLM agent
        response = call_llm(messages, tools)
        if not response:
            print("❌ LLM call failed.")
            return {"error": "LLM call failed"}

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print("ℹ️ No tool call from agent. Exiting.")
            break

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"🛠️ Agent is calling tool: {tool_name}")
            print(f"🔧 Arguments: {json.dumps(arguments, indent=2)}")

            if tool_name == "finish":
                print("✅ Agent called finish. Workflow complete.")
                return arguments

            tool_result = execute_agent_tool(tool_name, arguments)

            # Save partial results
            if tool_name == "generate_summary":
                results["summary"] = tool_result.get("summary")
            elif tool_name == "create_social_media_posts":
                results["social_posts"] = tool_result
            elif tool_name == "create_email_newsletter":
                results["email"] = tool_result

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
            })

    print("⚠️ Agent did not call 'finish'. Returning collected outputs.")
    return results  # fallback



# === Bonus Comparison ===
def compare_workflows(blog_post):
    reflexion_output = run_workflow_with_reflexion(blog_post)
    agent_output = run_agent_workflow(blog_post)
    return {
        "reflexion_eval": {
            "summary": evaluate_content(reflexion_output["summary"], "summary"),
            "social": evaluate_content(json.dumps(reflexion_output["social_media"]), "social_media_post"),
            "email": evaluate_content(json.dumps(reflexion_output["email"]), "email")
        },
        "agent_eval": {
            "summary": evaluate_content(agent_output.get("summary", ""), "summary"),
            "social": evaluate_content(json.dumps(agent_output.get("social_posts", {})), "social_media_post"),
            "email": evaluate_content(json.dumps(agent_output.get("email", {})), "email")
        }
    }
if __name__ == "__main__":
    import pprint

    print("🔄 Loading sample blog post...")
    blog_post = get_sample_blog_post()

    if not blog_post:
        print("❌ Blog post data not found. Make sure 'sample_blog_post.json' is in the same directory.")
        exit(1)

    print("\n🚀 Running Reflexion Workflow...")
    reflexion_output = run_workflow_with_reflexion(blog_post)
    print("\n📋 Reflexion Output:")
    pprint.pprint(reflexion_output)

    print("\n🤖 Running Agent Workflow...")
    agent_output = run_agent_workflow(blog_post)
    print("\n📋 Agent Output:")
    pprint.pprint(agent_output)

    print("\n📊 Running Comparative Evaluation...")
    comparison = compare_workflows(blog_post)
    print("\n📈 Evaluation Results:")
    pprint.pprint(comparison)
