from openai import OpenAI
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
)
from typing import Union, Dict, Callable, Any
import json
import tiktoken
import ast


class ChatGPT:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        concept: str = None,
        **kwargs,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        self.concept = concept
        self.gpt_opt = {}
        if kwargs:
            self.gpt_opt.update(kwargs)

    def get_gpt_res(self, gpt_args: dict):
        """
        OpenAI에 실제로 요청 보내고 응답 받는 함수. choices의 result list와 token 사용 정보를 반환합니다.
        - Attributes
            - `gpt_args`(dict, req): GPT 요청 보낼 때 쓰는 인자들. model, messages, temperature... 이런 거
        - Return
            - list(OpenAI Obj): completion 결과값(들), arg에 있던 n의 개수만큼의 길이
            - int: chat tokens와 completion tokens를 모두 합한 total tokens만을 가져옵니다. stateful한 놈을 만들 때 토큰 제어용으로 사용하세용.
        """
        gpt_response = self.client.chat.completions.create(**gpt_args)
        res_msg = gpt_response.choices
        token_info = gpt_response.usage.total_tokens

        return res_msg, token_info

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(),
        reraise=True,
    )
    def chat(self, messages: list, n: int = 1) -> str:
        prompt = (
            [{"role": "system", "content": self.concept}] + messages
            if self.concept is not None
            else messages
        )
        gpt_args = {
            "model": self.model,
            "messages": prompt,
            "n": n,
        }
        gpt_args.update(self.gpt_opt)

        gpt_response, token_info = self.get_gpt_res(gpt_args)
        res = []
        for gpt_res in gpt_response:
            msg = gpt_res.message.content
            if isinstance(msg, str):
                msg = msg.strip()
            res.append({"res": msg, "used_func": False})

        return {"res": res, "total_tokens": token_info}

    __call__ = chat


class ChatGPT_func(ChatGPT):
    """
    `OpenAI GPT API`의 [function_call](https://platform.openai.com/docs/guides/gpt/function-calling) 기능을 활용한 API 활용 class.
    - Attributes
        - `api_key`(str, req): OpenAI api key
        - `concept`(str, default=None): GPT's concept(on "system" role)
        - `temp`(float, default=1.): GPT's temparature
        - `func_desc`(list[dict], default=None): descriptions of functions for GPT. note that [here](https://platform.openai.com/docs/guides/gpt/function-calling)
        - `functions`(dict[str, callable], default=None): functions for GPT. note that [here](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)
    - Note
        - `__call__ = chat`
        - [여기](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)는 한 번쯤 읽어 보시는 걸 추천합니당
        - init 시 GPT의 function calling을 쓰고자 한다면, `func_desc`와 `functions` 인자를 모두 넣으세용. 아니면 에러 뱉어용
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        concept: str = None,
        func_desc: list = None,
        functions: Dict[str, Callable] = None,
        **kwargs,
    ) -> None:
        """
        initializer.\n
        """
        assert (func_desc is None) == (
            functions is None
        ), "function calling 쓸 거면 func_desc랑 functions 다 넣든가 안 쓸 거면 다 넣지 마세여ㅡㅡ 왜 헷갈리게 해"
        self.func_desc = func_desc
        self.functions = functions
        super().__init__(api_key, model_name, concept, **kwargs)

    @staticmethod
    def _chat_res_parser(chat_res: list, return_args_only: bool):
        """
        GPT result parser. \n
        - Attributes
            - `chat_res`(list, req): { ChatCompletion result }["choices"]
            - `return_args_only`(bool, req): GPT 응답 결과에서 argument로 오는 놈을 쓰는 건지 아닌지 판단. 이게 True면 GPT에 요청했던 functions의 args 중에 used_func가 있어야 합니다.
        """
        gpt_func_lst = []
        for res in chat_res:
            if func_call := res.message.function_call:
                if return_args_only:
                    func = json.loads(func_call.arguments).get("func_name")
                    gpt_func_lst.append(
                        {"func_name": func, "args": None, "used_func": True}
                    )
                else:
                    func_info = {}
                    try:
                        func = func_call.name
                        func_info["func_name"] = func
                    except:
                        func_info["func_name"] = None
                    try:
                        args = json.loads(func_call["arguments"])
                        func_info["args"] = args
                    except:
                        func_info["args"] = None

                    func_info["used_func"] = True
                    gpt_func_lst.append(func_info)

            else:
                msg = res.message.content
                if isinstance(msg, str):
                    msg = msg.strip()
                gpt_func_lst.append({"res": msg, "used_func": False})

        return gpt_func_lst

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(),
        reraise=True,
    )
    def chat(
        self,
        messages: list,
        n: int = 1,
        args: Dict[str, Any] = None,
        ensure_func_name: str = None,
        use_gpt_args: bool = False,
        return_args_only: bool = False,
    ) -> dict:
        """
        [OpenAI cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)의 API call 함수를 약간 수정.\n
        - Attributes
            - `messages`(list[dict], req): conversation history to send to the API (except "system")
                - e.g. [{"role": "user", "content": "who are you? explain in 10 words."}, {"role": "assistant", "content": "I'm an advanced artificial intelligence model developed by OpenAI for communication."}]
            - `n`(int, default=1): GPT한테 몇 개의 completion res를 달라고 할지 전달하는 인자.
            - `args`(dict[str, any], default=None): arguments to be written to functions inferred from GPT. ignored if the class's user_gpt_args is True.
            - `ensure_func_name`(str, default=None): the name of the function that must be inferred from the GPT. this must be the name of the function present in the func_desc argument in the __init__ of the class.
            - `use_gpt_args`(bool, default=False): determine if you want to use the arguments derived by GPT in your function.
            - `return_args_only`(bool, default=False): GPT 응답 결과에서 argument로 오는 놈을 쓰는 건지 아닌지 판단. 이게 True면 GPT에 요청했던 functions의 args 중에 used_func가 있어야 합니다.
        - Return
            - dict({`res`: list([{`content`: str, `used_func`: bool, `used_tokens`: int}]), `total_tokens`: int})
        - Note
            - `tenacity`의 `@retry` 데코레이터를 사용했습니다. 실행 중 exception이 발생하면 1초에서 40초 사이의 랜덤한 간격을 두고 함수를 재실행합니다. 최대 3번까지 시도하고, 아니면 `RetryError`를 raise.
            - 단, `function_call` 기능 사용 시 실행한 함수의 결과값이 str이 아니라면 재시도 없이 바로 `RetryError`를 raise합니당
            - `return_args_only를` 쓸 거면 `ensure_func_name을` 반드시 집어넣으세용.
            - 당연하게도, `ensure_func_name의` 값은 `self.functions`의 key 중에 존재해야 합니다.
            - `use_gpt_args`와 `return_args_only`가 둘 다 True일 수는 없어용. `return_args_only`를 사용하면 'args로 반환되는 func'를 호출하기 때무니에용.
        """
        assert not (
            return_args_only and ensure_func_name is None
        ), "return_args_only를 쓸 거면 ensure_func_name을 반드시 집어넣으세용"
        assert not (
            use_gpt_args and return_args_only
        ), "use_gpt_args와 return_args_only가 둘 다 True일 수는 없어용. return_args_only를 사용하면 'args로 반환되는 func'를 호출하기 때무니에용."

        prompt = (
            [{"role": "system", "content": self.concept}] + messages
            if self.concept is not None
            else messages
        )
        gpt_args = {
            "model": self.model,
            "messages": prompt,
            "n": n,
        }
        gpt_args.update(self.gpt_opt)

        if self.func_desc is not None and self.functions is not None:
            gpt_args.update({"functions": self.func_desc})

        if ensure_func_name is not None:
            gpt_args.update({"function_call": {"name": ensure_func_name}})

        res_msg, token_info = self.get_gpt_res(gpt_args)

        parsed_res = self._chat_res_parser(res_msg, return_args_only)

        res = []
        for parsed_item in parsed_res:
            if parsed_item["used_func"]:
                func_name = parsed_item["func_name"]
                func_args = None

                if use_gpt_args and parsed_item["args"] is not None:
                    func_args = parsed_item["args"]
                    func_res = self.functions[func_name](**func_args)
                elif not use_gpt_args and args is not None:
                    func_args = args
                    func_res = self.functions[func_name](**func_args)
                else:
                    func_res = self.functions[func_name]()

                if not isinstance(func_res, str):
                    raise RetryError

                prompt.append(
                    {"role": "function", "name": func_name, "content": func_res}
                )
                gpt_func_args = {
                    "model": self.model,
                    "messages": prompt,
                    "temperature": self.temp,
                }

                gpt_func_res, token_func_info = self.get_gpt_res(gpt_func_args)
                token_info += token_func_info

                res.append(
                    {
                        "content": gpt_func_res[0].message.content,
                        "used_func": func_name,
                        "used_tokens": token_func_info,
                        "args": func_args,
                    }
                )
            else:
                res.append({"content": parsed_item["res"], "used_func": None})

        return {"res": res, "total_tokens": token_info}

    __call__ = chat
