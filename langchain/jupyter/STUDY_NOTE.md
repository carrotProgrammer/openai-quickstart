# LangChain学习笔记
该笔记分为两个部分，分别是理论部分与应用部分

### LangChain理论部分
## Modules
一个 LangChain 应用是通过很多个组件实现的，LangChain 主要支持 6 种组件：
[结构图]()

---
## Model I/O
[Model]()

### 模型抽象
各种类型的模型和模型集成，比如 GPT-4 等大语言模型，LangChain 将各家公司的大模型进行了抽象，封装了通用的 API，我们只要使用对应的 API 就可以完成对各个公司大模型的调用
- 语言模型[LLMS]
- 聊天模型[Chat Models]
#### 语言模型[LLMS]
类继承关系：
```
BaseLanguageModel --> BaseLLM --> LLM --> <name>  # Examples: AI21, HuggingFaceHub, OpenAI
```

使用 LangChain 调用 OpenAI GPT Completion API：
```
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003")
print(llm("Tell me a Joke"))
```
Q. Why did the scarecrow win an award?

A. Because he was outstanding in his field!

主要参数：

[点击这里查看主要参数](https://github.com/carrotProgrammer/openai-quickstart/blob/main/langchain/jupyter/model_io/BaseOpenAI.md)

可以调用变量的方式修改最大token数
```
llm.max_tokens
```
可以调采样温度，为0时最稳定
```
llm.temperature=0
```
#### 聊天模型[Chat Models]
类继承关系：

```
BaseLanguageModel --> BaseChatModel --> <name>  # Examples: ChatOpenAI, ChatGooglePalm
```

主要抽象：

```
AIMessage, BaseMessage, HumanMessage
```

使用 LangChain 调用 OpenAI GPT ChatCompletion API：
```
from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# 以消息列表的方式传入
messages = [SystemMessage(content="You are a helpful assistant."),
 HumanMessage(content="Who won the world series in 2020?"),
 AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."), 
 HumanMessage(content="Where was it played?")]

chat_model(messages)

```
生成一个AIMessage结果：

AIMessage(content='The 2020 World Series was played at Globe Life Field in Arlington, Texas.', additional_kwargs={}, example=False)

---
### 模型输入Prompt
一个语言模型的提示是用户提供的一组指令或输入，用于引导模型的响应，帮助它理解上下文并生成相关和连贯的基于语言的输出，例如回答问题、完成句子或进行对话。

- 提示模板（Prompt Templates）：参数化的模型输入
- 示例选择器（Example Selectors）：动态选择要包含在提示中的示例

#### 提示模板 Prompt Templates
**Prompt Templates 提供了一种预定义、动态注入、模型无关和参数化的提示词生成方式，以便在不同的语言模型之间重用模板。**

一个模板可能包括指令、少量示例以及适用于特定任务的具体背景和问题。

通常，提示要么是一个字符串（LLMs），要么是一组聊天消息（Chat Model）。

使用 PromptTemplate 类生成提示词：
- 使用 from_template 方法实例化 PromptTemplate
```
from langchain import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)

# 使用 format 生成提示
prompt = prompt_template.format(adjective="funny", content="chickens")
print(prompt)

```
- 使用构造函数（Initializer）实例化 PromptTemplate
```
valid_prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template="Tell me a {adjective} joke about {content}."
)

valid_prompt.format(adjective="funny", content="chickens")
```

#### 使用 ChatPromptTemplate 类生成适用于聊天模型的聊天记录
**`ChatPromptTemplate` 类的实例，使用`format_messages`方法生成适用于聊天模型的提示。**
-使用 from_messages 方法实例化 ChatPromptTemplate
```
summary_template = ChatPromptTemplate.from_messages([
    ("system", "你将获得关于同一主题的{num}篇文章（用-----------标签分隔）。首先总结每篇文章的论点。然后指出哪篇文章提出了更好的论点，并解释原因。"),
    ("human", "{user_input}"),
])

messages = summary_template.format_messages(
    num=3,
    user_input='''1. [PHP是世界上最好的语言]
PHP是世界上最好的情感派编程语言，无需逻辑和算法，只要情绪。它能被蛰伏在冰箱里的PHP大神轻易驾驭，会话结束后的感叹号也能传达对代码的热情。写PHP就像是在做披萨，不需要想那么多，只需把配料全部扔进一个碗，然后放到服务器上，热乎乎出炉的网页就好了。
-----------
2. [Python是世界上最好的语言]
Python是世界上最好的拜金主义者语言。它坚信：美丽就是力量，简洁就是灵魂。Python就像是那个永远在你皱眉的那一刻扔给你言情小说的好友。只有Python，你才能够在两行代码之间感受到飘逸的花香和清新的微风。记住，这世上只有一种语言可以使用空格来领导全世界的进步，那就是Python。
-----------
3. [Java是世界上最好的语言]
Java是世界上最好的德育课编程语言，它始终坚守了严谨、安全的编程信条。Java就像一个严格的老师，他不会对你怀柔，不会让你偷懒，也不会让你走捷径，但他教会你规范和自律。Java就像是那个喝咖啡也算加班费的上司，拥有对邪恶的深度厌恶和对善良的深度拥护。
'''
)
```

#### 使用 FewShotPromptTemplate 类生成 Few-shot Prompt 
构造 few-shot prompt 的方法通常有两种：
- 从示例集（set of examples）中手动选择；
- 通过示例选择器（Example Selector）自动选择.

---
### Data connection
### Chains
### Memory
### Agents
### Callbacks
