# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.engine.protocol import StreamingInput
from vllm.outputs import STREAM_FINISHED, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector


@pytest.fixture
def streaming_llm(mock_async_llm):
    llm = mock_async_llm
    llm.get_supported_tasks = AsyncMock(return_value=("generate",))
    llm._add_streaming_input_request = AsyncLLM._add_streaming_input_request.__get__(
        llm, AsyncLLM
    )
    llm.input_processor = MagicMock()
    llm.input_processor.process_inputs.side_effect = lambda **kwargs: MagicMock(
        request_id="request",
        prompt_embeds=None,
        resumable=kwargs.get("resumable", False),
    )
    queues = []
    tasks = []

    async def add_request(request_id, prompt, params, **kwargs):
        queue = await llm._add_streaming_input_request(request_id, prompt, params)
        queues.append(queue)
        tasks.append(queue._input_stream_task)
        return queue

    llm.add_request = add_request
    llm._add_request = AsyncMock()
    return llm, queues, tasks


async def consume_stream(llm, stream):
    async for _ in llm.generate(stream, SamplingParams(max_tokens=4), "request"):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("num_chunks", [0, 1, 3])
async def test_input_stream_error_does_not_send_final_sentinel(
    streaming_llm, num_chunks
):
    llm, queues, tasks = streaming_llm
    error = ValueError("input failed")

    async def stream():
        for _ in range(num_chunks):
            yield StreamingInput(prompt="hello")
        raise error

    with pytest.raises(ValueError, match="input failed") as caught:
        await asyncio.wait_for(consume_stream(llm, stream()), timeout=2)
    assert caught.value is error
    await asyncio.wait_for(tasks[0], timeout=2)
    assert llm._add_request.await_count == num_chunks
    assert all(call.args[0].resumable for call in llm._add_request.await_args_list)
    llm.abort.assert_awaited_once_with("request", internal=True)
    assert queues[0].get_nowait() is None
    assert queues[0]._input_stream_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_dead", [False, True])
async def test_final_request_failure_does_not_deadlock_generate(
    streaming_llm, engine_dead
):
    llm, queues, tasks = streaming_llm
    error = EngineDeadError() if engine_dead else RuntimeError("final dispatch failed")

    async def submit(req, *args):
        if not req.resumable:
            raise error

    llm._add_request.side_effect = submit

    async def stream():
        yield StreamingInput(prompt="hello")

    expected = EngineDeadError if engine_dead else EngineGenerateError
    with pytest.raises(expected) as caught:
        await asyncio.wait_for(consume_stream(llm, stream()), timeout=2)
    if engine_dead:
        assert caught.value is error
        llm.abort.assert_not_awaited()
    else:
        assert caught.value.__cause__ is error
        llm.abort.assert_awaited_once_with("request", internal=True)
    await asyncio.wait_for(tasks[0], timeout=2)
    assert llm._add_request.await_count == 2
    assert queues[0].get_nowait() is None
    assert queues[0]._input_stream_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("num_chunks", [0, 3])
async def test_streaming_input_normal_finalization(streaming_llm, num_chunks):
    llm, queues, tasks = streaming_llm

    async def submit(req, prompt, parent, index, queue):
        if not req.resumable:
            queue.put(make_output("request", finished=True))

    llm._add_request.side_effect = submit

    async def stream():
        for _ in range(num_chunks):
            yield StreamingInput(prompt="hello")

    await asyncio.wait_for(consume_stream(llm, stream()), timeout=2)
    await asyncio.wait_for(tasks[0], timeout=2)
    assert llm._add_request.await_count == (num_chunks + 1 if num_chunks else 0)
    llm.abort.assert_not_awaited()
    assert queues[0]._input_stream_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("early_stop", [False, True])
async def test_streaming_input_consumer_cancellation_stops_input(
    streaming_llm, early_stop
):
    llm, queues, tasks = streaming_llm
    submitted = asyncio.Event()
    input_closed = asyncio.Event()

    async def submit(req, prompt, parent, index, queue):
        queue.put(make_output("request", finished=False))
        submitted.set()

    llm._add_request.side_effect = submit

    async def stream():
        try:
            yield StreamingInput(prompt="hello")
            await asyncio.Future()
        finally:
            input_closed.set()

    if early_stop:
        consumer = llm.generate(stream(), SamplingParams(max_tokens=4), "request")
        await asyncio.wait_for(anext(consumer), timeout=2)
        await asyncio.wait_for(consumer.aclose(), timeout=2)
    else:
        consumer = asyncio.create_task(consume_stream(llm, stream()))
        await asyncio.wait_for(submitted.wait(), timeout=2)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, timeout=2)
    await asyncio.wait_for(input_closed.wait(), timeout=2)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(tasks[0], timeout=2)
    assert llm._add_request.await_count == 1
    llm.abort.assert_awaited_once_with("request", internal=True)
    assert queues[0].get_nowait() is None
    assert queues[0]._input_stream_task is None


@pytest.mark.asyncio
async def test_final_dispatch_keeps_input_task_cancellable(streaming_llm):
    llm, queues, tasks = streaming_llm
    final_started = asyncio.Event()
    final_cancelled = asyncio.Event()

    async def submit(req, *args):
        if not req.resumable:
            final_started.set()
            try:
                await asyncio.Future()
            finally:
                final_cancelled.set()

    llm._add_request.side_effect = submit

    async def stream():
        yield StreamingInput(prompt="hello")

    consumer = asyncio.create_task(consume_stream(llm, stream()))
    try:
        await asyncio.wait_for(final_started.wait(), timeout=2)
        assert queues[0]._input_stream_task is tasks[0]
    finally:
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, timeout=2)
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(tasks[0], timeout=2)
    assert final_cancelled.is_set()
    assert queues[0]._input_stream_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_via", ["task", "shutdown", "threaded_shutdown"])
async def test_output_handler_cancellation_terminates_generate(
    mock_async_llm, cancel_via
):
    llm = mock_async_llm
    queue = RequestOutputCollector(RequestOutputKind.DELTA, "request")
    handler_started = asyncio.Event()
    consumer_started = asyncio.Event()

    async def get_output():
        handler_started.set()
        await asyncio.Future()

    async def add_request(*args, **kwargs):
        consumer_started.set()
        return queue

    processor = OutputProcessor.__new__(OutputProcessor)
    processor.request_states = {"request": MagicMock(queue=queue)}
    processor.propagate_error = MagicMock(wraps=processor.propagate_error)
    llm.output_processor = processor
    llm.engine_core = MagicMock()
    llm.engine_core.get_output_async = get_output
    llm.renderer = MagicMock()
    llm.log_stats = False
    llm.logger_manager = None
    llm.output_handler = None
    llm.add_request = add_request
    AsyncLLM._run_output_handler(llm)
    consumer = asyncio.create_task(consume_stream(llm, "hello"))
    try:
        await asyncio.wait_for(handler_started.wait(), timeout=2)
        await asyncio.wait_for(consumer_started.wait(), timeout=2)
        if cancel_via == "task":
            llm.output_handler.cancel()
        elif cancel_via == "shutdown":
            AsyncLLM.shutdown(llm)
        else:
            await asyncio.wait_for(asyncio.to_thread(AsyncLLM.shutdown, llm), timeout=2)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(llm.output_handler, timeout=2)
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(consumer, timeout=2)
        processor.propagate_error.assert_called_once()
        llm.abort.assert_not_awaited()
        assert queue.get_nowait() is None
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        llm.output_handler.cancel()
        await asyncio.gather(llm.output_handler, return_exceptions=True)


@pytest.fixture
def mock_async_llm():
    """Create a mock AsyncLLM with mocked dependencies."""
    # Create a minimal mock without initializing the full engine
    llm = MagicMock(spec=AsyncLLM)

    # Mock the essential attributes
    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MagicMock()
    llm.model_config.max_model_len = 2048
    llm.log_requests = False
    llm.errored = False
    llm._pause_cond = asyncio.Condition()
    llm._paused = False

    # Mock methods
    llm._run_output_handler = MagicMock()
    llm.abort = AsyncMock()

    # Use the real generate method from AsyncLLM
    llm.generate = AsyncLLM.generate.__get__(llm, AsyncLLM)

    return llm


@pytest.mark.asyncio
async def test_generate_normal_flow(mock_async_llm):
    """Test normal generation flow with streaming requests."""
    request_id = "test_request"
    prompt = "Tell me about Paris"
    sampling_params = SamplingParams(max_tokens=10)

    # Create a mock queue with outputs
    queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)
    output1 = RequestOutput(
        request_id=request_id,
        prompt="Tell me about Paris",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
    )
    output2 = RequestOutput(
        request_id=request_id,
        prompt="Tell me about Paris",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
    )

    # Feed outputs to queue as they're consumed to avoid aggregation
    async def feed_outputs():
        queue.put(output1)
        await asyncio.sleep(1)  # Let first output be consumed
        queue.put(output2)

    asyncio.create_task(feed_outputs())  # noqa

    # Mock add_request to return the queue
    async def mock_add_request(*args, **kwargs):
        return queue

    mock_async_llm.add_request = mock_add_request

    # Collect outputs from generate
    outputs = []
    async for output in mock_async_llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        outputs.append(output)

    assert len(outputs) == 2
    assert outputs[0].finished is False
    assert outputs[1].finished is True


def make_output(request_id: str, finished: bool) -> RequestOutput:
    """Helper to create a RequestOutput."""
    return RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=finished,
    )


@pytest.mark.asyncio
async def test_generate_with_async_generator():
    """Test generate with an async input generator.

    With the new streaming input API, completion is signaled by finishing
    the input generator (not via a resumable flag). Each input chunk
    produces intermediate outputs, and the final output has finished=True.
    """
    request_id = "test"
    sampling_params = SamplingParams(max_tokens=10)

    llm = MagicMock(spec=AsyncLLM)
    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MagicMock()
    llm.model_config.max_model_len = 2048
    llm.log_requests = False
    llm.errored = False
    llm._pause_cond = asyncio.Condition()
    llm._paused = False
    llm._run_output_handler = MagicMock()
    llm.abort = AsyncMock()

    # Bind the real generate method
    llm.generate = AsyncLLM.generate.__get__(llm, AsyncLLM)

    # Track inputs processed
    inputs_received = []
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id)

    async def mock_add_request(req_id, prompt, params, *args, **kwargs):
        # When prompt is an AsyncGenerator, process streaming inputs
        if isinstance(prompt, AsyncGenerator):
            # Process inputs in background, produce outputs
            async def handle_stream():
                async for input_chunk in prompt:
                    inputs_received.append(input_chunk.prompt)
                    # Each input produces an intermediate output
                    queue.put(make_output(req_id, finished=False))
                    await asyncio.sleep(0.01)
                # Final output when stream ends
                queue.put(make_output(req_id, finished=True))

            asyncio.create_task(handle_stream())
            return queue
        return queue

    llm.add_request = mock_add_request

    async def input_generator() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt="Hello", sampling_params=sampling_params)
        yield StreamingInput(prompt=" world", sampling_params=sampling_params)

    outputs = []
    async for output in llm.generate(input_generator(), sampling_params, request_id):
        outputs.append(output)

    # Two intermediate outputs + one final output
    assert len(outputs) == 3
    assert outputs[0].finished is False
    assert outputs[1].finished is False
    assert outputs[2].finished is True
    # Both inputs were processed
    assert inputs_received == ["Hello", " world"]


@pytest.mark.asyncio
async def test_empty_input_stream_finishes_without_engine_request():
    llm = AsyncLLM.__new__(AsyncLLM)
    llm._validate_streaming_input_sampling_params = MagicMock()
    llm.get_supported_tasks = AsyncMock(return_value=("generate",))
    llm._run_output_handler = MagicMock()
    llm._add_request = AsyncMock()

    final_req = MagicMock()
    final_req.request_id = "request-int"
    llm.input_processor = MagicMock()
    llm.input_processor.process_inputs.return_value = final_req

    async def empty_stream():
        return
        yield  # pragma: no cover

    queue = await llm._add_streaming_input_request(
        "request", empty_stream(), SamplingParams(max_tokens=4)
    )
    task = queue._input_stream_task
    assert task is not None
    await task

    assert queue.get_nowait() is STREAM_FINISHED
    llm._add_request.assert_not_awaited()
