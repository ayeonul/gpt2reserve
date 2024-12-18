from openai import OpenAI
import time
import json


class GPTAssistant(object):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        functions: dict[str, callable] = None,
        return_args_only: bool = False,
        **kwargs,
    ) -> None:

        assert model_name in (
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ), "모델 이름 확인하든가 assert문을 고치든가 - assistant api 지원되는 녀석만 넣으세여"

        assert (
            not return_args_only or "tools" in kwargs
        ), "function calling으로 의도추론하라면서 tools를 안 넣음"

        if return_args_only:
            functions = functions.copy() if functions is not None else {}
            for tool in kwargs["tools"]:
                if (func := tool.get("function", None)) is not None:
                    functions.update({func["name"]: self.__return_args})

        else:
            assert (
                any(item.get("type") == "function" for item in kwargs.get("tools", {}))
            ) == (functions is not None), (
                "tool에 function이 빠졌든가 functions 인자를 안 줬든가"
            )

        self.return_args_only = return_args_only

        # OpenAI 클라이언트 초기화
        self.__client = OpenAI(api_key=api_key)
        self.__assistant_needs_file = kwargs.get("tools", None)

        # Assistant 생성
        self.assistant = self.__client.beta.assistants.create(
            model=model_name, **kwargs
        )
        self.functions = functions

    @staticmethod
    def __return_args(**kwargs):
        return kwargs

    def _get_file_id(self, file: bytes) -> str:
        uploaded_file = self.__client.files.create(file=file, purpose="assistants")

        while True:
            file_status = self.__client.files.retrieve(uploaded_file.id)
            if file_status.status == "processed":
                return uploaded_file.id
            if file_status.status == "error":
                raise
            time.sleep(0.5)

    def __submit_tool_outputs(self, thread_id: str, run_id: str, tool_calls: list):
        tool_outputs = []
        tool_info = []
        for call in tool_calls:
            func_name = call.function.name
            func_args = call.function.arguments
            if not func_name:
                raise
            tool_info.append({"name": func_name, "args": func_args})
            if func_args != "":
                args = json.loads(func_args)
                func_res = self.functions[func_name](**args)
            else:
                func_res = self.functions[func_name]()

            tool_outputs.append({"tool_call_id": call.id, "output": str(func_res)})

        if self.return_args_only:
            return tool_outputs

        if len(tool_outputs) != 0:
            run = self.__client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id, run_id=run_id, tool_outputs=tool_outputs
            )
            while True:
                run_status = self.__client.beta.threads.runs.retrieve(
                    run.id, thread_id=thread_id
                )
                if run_status.status == "completed":
                    return tool_info
                if run_status.status in (
                    "requires_action",
                    "cancelling",
                    "cancelled",
                    "failed",
                    "expired",
                ):
                    raise
                time.sleep(0.5)

    def chat(
        self,
        thread_id: str,
        user_input: str,
        upload_files: list[bytes] = None,
        return_tool_info: bool = False,
    ) -> dict:
        assert (upload_files is None) or (
            self.__assistant_needs_file is not None
        ), "files는 code_interpreter 혹은 retrieval에서만 작동합니다."

        create_opt = {
            "thread_id": thread_id,
            "role": "user",
            "content": user_input,
        }
        if upload_files is not None and len(upload_files) != 0:
            create_opt["file_ids"] = [self._get_file_id(file) for file in upload_files]

        # 기존 Thread에 User의 새로운 Message 추가
        message = self.__client.beta.threads.messages.create(**create_opt)

        # Assistant의 응답 대기를 위한 새 Run 생성
        run = self.__client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=self.assistant.id
        )

        tool_calls = None
        # Run 상태가 완료될 때까지 체크
        while True:
            run_status = self.__client.beta.threads.runs.retrieve(
                run.id, thread_id=thread_id
            )
            if run_status.status == "completed":
                break
            if run_status.status == "requires_action":
                # return run_status, run.id
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                break

            if run_status.status in (
                "cancelling",
                "cancelled",
                "failed",
                "expired",
            ):
                raise
            time.sleep(0.5)
        if tool_calls is not None:
            tool_info = self.__submit_tool_outputs(thread_id, run.id, tool_calls)
            if self.return_args_only:
                return {"results": tool_info, "thread_id": thread_id}

        # Assistant의 응답 반환
        responses = self.__client.beta.threads.messages.list(
            thread_id=thread_id, before=message.id
        )
        data = [data.content for data in responses.data]
        results = {
            "results": [d[0].text.value for d in data][::-1],
            "thread_id": thread_id,
        }
        if tool_calls is not None and return_tool_info:
            results["tools"] = tool_info
        return results

    def first_chat(self, user_input: str, upload_files: list[bytes] = None) -> dict:
        assert (upload_files is None) or (
            self.__assistant_needs_file is not None
        ), "files는 code_interpreter 혹은 retrieval에서만 작동합니다."

        # 새 Thread 생성
        thread = self.__client.beta.threads.create()

        return self.chat(thread.id, user_input, upload_files)


class OnlyFunc(GPTAssistant):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        tools: list = None,
        **kwargs,
    ) -> None:
        super().__init__(
            api_key, model_name, tools=tools, tool_choice="required", **kwargs
        )

    def run(self, user_input: str, upload_files: list[bytes] = None):
        thread = self.__client.beta.threads.create()
