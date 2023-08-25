```python
class BaseOpenAI(BaseLLM):
    """OpenAI 大语言模型的基类。"""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"openai_api_key": "OPENAI_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    client: Any  #: :meta private:
    model_name: str = Field("text-davinci-003", alias="model")
    """使用的模型名。"""
    temperature: float = 0.7
    """要使用的采样温度。"""
    max_tokens: int = 256
    """完成中生成的最大令牌数。 
    -1表示根据提示和模型的最大上下文大小返回尽可能多的令牌。"""
    top_p: float = 1
    """在每一步考虑的令牌的总概率质量。"""
    frequency_penalty: float = 0
    """根据频率惩罚重复的令牌。"""
    presence_penalty: float = 0
    """惩罚重复的令牌。"""
    n: int = 1
    """为每个提示生成多少完成。"""
    best_of: int = 1
    """在服务器端生成best_of完成并返回“最佳”。"""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """保存任何未明确指定的`create`调用的有效模型参数。"""
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_organization: Optional[str] = None
    # 支持OpenAI的显式代理
    openai_proxy: Optional[str] = None
    batch_size: int = 20
    """传递多个文档以生成时使用的批处理大小。"""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """向OpenAI完成API的请求超时。 默认为600秒。"""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """调整生成特定令牌的概率。"""
    max_retries: int = 6
    """生成时尝试的最大次数。"""
    streaming: bool = False
    """是否流式传输结果。"""
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    """允许的特殊令牌集。"""
    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    """不允许的特殊令牌集。"""
    tiktoken_model_name: Optional[str] = None
    """使用此类时传递给tiktoken的模型名。
    Tiktoken用于计算文档中的令牌数量以限制它们在某个限制以下。
    默认情况下，设置为None时，这将与嵌入模型名称相同。
    但是，在某些情况下，您可能希望使用此嵌入类与tiktoken不支持的模型名称。
    这可以包括使用Azure嵌入或使用多个模型提供商的情况，这些提供商公开了类似OpenAI的API但模型不同。
    在这些情况下，为了避免在调用tiktoken时出错，您可以在此处指定要使用的模型名称。"""
```
