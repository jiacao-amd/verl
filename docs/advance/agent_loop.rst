Agent Loop
==========

Last updated: 10/08/2026.

.. versionadded:: 0.4.2
   [status: alpha]

.. warning::
   Agent Loop is ready for use, but the API may change in future releaes.

Agent Loop is designed as general interface for multi-turn rollout and agentic reinforcement learning.

**Design goal**:

- Plugable user defined agent loop
- Provide standard request generate api with different inference frameworks
- Provide request level load balance between multiple inference servers

**Non-goal**:

- How tool is defined and how to call tool

In high level overview, agent loop is given a prompt, run user defined loop: call LLM generate api, call tools, ...
and return the final output. The final output is then calculated reward and used as trajectory for RL training.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_overview.svg?raw=true


API Design
----------

``AgentLoopBase`` class is the abstraction of agent loop, and ``run`` method is the only interface that user need to implement.
The run method, given prompt messages in format: [{"role": "user"}, {"content": "..."}], and additional sampling params,
could do whatever user wants, such as

- call LLM generate api
- call tools: web search, database query, code sandbox, ...
- environment interaction
- reflection
- ...

.. code:: python

   class AgentLoopBase(ABC):
       @abstractmethod
       async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
           """Run agent loop to interact with LLM server and environment.

           Args:
               sampling_params (Dict[str, Any]): LLM sampling params.
               **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

           Returns:
               AgentLoopOutput: Agent loop output.
           """
           raise NotImplementedError

After running user defined loop, run method should return ``AgentLoopOutput``, including prompt token ids,
response token ids, and response mask.

.. code:: python

   class AgentLoopOutput(BaseModel):
       """Agent loop output."""

       prompt_ids: list[int]
       """Prompt token ids."""
       response_ids: list[int]
       """Response token ids including LLM generated token, tool response token."""
       response_mask: list[int]
       """Response mask, 1 for LLM generated token, 0 for tool response token."""

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_output.svg?raw=true

.. note:: AgentLoopOutput only output one trajectory for a given prompt, multiple trajectories output is still under discussion.

Architecture Design
-------------------

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_architecture.png?raw=true

A single PPO step contain two phase: rollout and train. In rollout phase:

1. PPOTrainer sample a batch from dataset and call ``AgentLoopManager.generate_sequences``.
2. AgentLoopManager ``wake_up`` all async LLM server instances, which will sync weights between inference engine(vLLM/SGLang) and training engine(FSDP/Megatron-LM).
3. AgentLoopManager split batch into chunks and send each chunk to ``AgentLoopWorker``.
4. AgentLoopWorker receive chunk and for each prompt, spawn a user defined ``AgentLoopBase`` instance, run ``run`` coroutine until end and get ``AgentLoopOutput``.

.. tip::
   AgentLoopWorker schedules multiple coroutines concurrently. If number of AgentLoopWorker equals batch_size, then each worker is response for one prompt.

In agent loop, when user need LLM generate response:

5. Call ``LLMServerClient.generate`` with prompt_ids.
6. LLMServerClient select a server instance with least request in first turn and send request to it. (In following turns, the request will be sent to the same server instance).
7. AsyncLLMServer receive a request, issue ipc/rpc with model_runner, and generate response. (There's slight differences between vLLM and SGLang, see below).

When all prompts in all AgentLoopWorker finish, AgentLoopManager gather results and return to PPOTrainer.

8. AgentLoopManager ``sleep`` all server instances, which will free kv cache and offload weights to CPU memory.

AsyncLLMServer
~~~~~~~~~~~~~~

AsyncLLMServer is the abstraction of LLM server with two types of generation api:

- `OpenAI chat completion <https://platform.openai.com/docs/api-reference/chat>`_: generate response for the given chat conversation.
- Token in token out: generate response ids for the given token ids.

We have officially supported vLLM and SGLang AsyncLLMServer, both of them implement the two api and are well tested.
Other inference engine should be easy to plug-in by implement the ``AsyncServerBase`` class.

.. code:: python

   class AsyncServerBase(ABC):
       @abstractmethod
       async def chat_completion(self, raw_request: Request) -> JSONResponse:
           """OpenAI chat completion API.

           Args:
               raw_request (Request): raw json request
           
           Returns:
               JSONResponse: json response

           API reference: https://platform.openai.com/docs/api-reference/chat/create
           """
           raise NotImplementedError

       @abstractmethod
       async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
           """Generate response ids given prompt ids.

           Args:
               prompt_ids (List[int]): prompt ids
               sampling_params (Dict[str, Any]): sampling params
               request_id (str): request id

           Returns:
               List[int]: response ids
           """
           raise NotImplementedError


Chat completion vs Token in token out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   The following conclusion is based on our recent experience and is still open to investigation and discussion.

Almost all agent frameworks (LangGraph, CrewAI, LlamaIndex, etc) call LLM with OpenAI chat completion api, and 
keep chat history as messages. So user may expect that we should use the chat completion api in multi-turn rollout.

But based on our recent experience on single-turn training on DAPO and multi-turn training on `retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_,
we found the token_ids from apply the final messages may not equal to the token_ids by concat prompt_ids and response_ids in each turn.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/multi_turn.png?raw=true

**Where does this inconsistency happened?**

First, the tool parser may alter the content. For example

.. code:: json

   {"role": "assistant", "content": "Let me call a <tool_call>...</tool_call> and get the result"}

After tool_calls extraction, the messages is like this:

.. code:: json

   {"role": "assistant", "content": "Let me call a and get the result", "tool_calls": [{"name": "foo", "arguments": "{}"}]}

Encode the extracted message back is not equal to the original LLM generated response_ids.

Second,  the `decode-encode` may also lead to inconsistency: `Agent-R1 issue#30 <https://github.com/0russwest0/Agent-R1/issues/30#issuecomment-2826155367>`_.

**What is the impact of this inconsistency?**

This inconsistency is not a big problem for serving/agent system, but is critical to RL training.
It causes the trajectory deviate from the policy model distribution. We have observed that apply_chat_template
to the final chat history messages make PPO training not even converged in single-turn.

vLLM
^^^^

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/async_vllm.png?raw=true

For vLLM, the Async LLM Engine is running in same process as the server, and ModelRunner is running in same process as FSDP/Megatron-LM workers.
Async LLM Engine communicate with ModelRunner through ZeroMQ. When server receive a request, it directly call engine to generate response_ids.

SGLang
^^^^^^

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/async_sglang.png?raw=true

For SGLang, the Async LLM Engine is running in same process as FSDP/Megatron-LM worker-0, and it spawn multiple subprocesses as ModelRunner.
Also, Async LLM Engine communicate with ModelRunner through ZeroMQ. When server receive a request, it remote call the worker-0 and get response_ids.

LLMServerClient
~~~~~~~~~~~~~~~~~~~~~

LLMServerClient serve as proxy to multiple AsyncLLMServer instances, provides:

- load balance: select a server instance with least request in first turn and send request to it.
- sticky session: bind request_id to server instance, so that the same request_id will be sent to the same server instance in following turns.

LLMServerClient is passed to ``AgentLoopBase.__init__``, whenever user want to interact with LLM in agent loop,
they can call ``LLMServerClient.generate`` to generate response_ids.

.. code:: python

   class LLMServerClient:
       async def generate(
           self,
           request_id,
           *,
           prompt_ids: list[int],
           sampling_params: dict[str, Any],
       ) -> list[int]:
           """Generate tokens from prompt ids.

           Args:
               request_id (str): request id for sticky session.
               prompt_ids (List[int]): List of prompt token ids.
               sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

           Returns:
               List[int]: List of generated token ids.
           """
           ...

Multi-turn request scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ToolAgentLoop`` can attach backend-neutral scheduling metadata to every model
request. The first request is classified as ``fresh``; requests after a tool
call are ``continuation``; partial-rollout attempts are ``retry``. Scheduling is
disabled by default, so existing rollout behavior is unchanged.

For throughput-oriented vLLM rollouts, prefer FIFO backend scheduling with a
resume-only admission burst. The base admission capacity remains available to
all requests, while continuation and retry requests may temporarily use extra
slots after returning from tool execution. The burst size is calculated from
current fresh and resume pressure, capped by
``max_resume_burst_requests``, and returns to zero when no resume request is
present. It does not reserve idle capacity in advance or let resumed requests
strictly overtake fresh requests in the backend scheduler. An optional
``fresh_wave_max_resume_burst_requests`` cap can keep that burst smaller while
fresh requests remain queued, then allow it to expand after the fresh queue
drains.

For example, one tested vLLM configuration uses:

.. code:: yaml

   router_class: verl.workers.rollout.router.SoftAdmissionRequestLoadBalancer
   max_concurrent_requests: 40
   fresh_wave_max_resume_burst_requests: 1
   max_resume_burst_requests: 20
   fresh_max_wait_seconds: 60

Then configure rollout without a request priority policy:

.. code:: yaml

   actor_rollout_ref:
     rollout:
       name: vllm
       scheduling_policy: fcfs
       max_num_seqs: 32
       engine_kwargs:
         vllm:
           max_num_batched_tokens: 32768
       router_config_path: /path/to/request_router.yaml

These values are workload-specific starting points. Tune the backend
running-request limit, per-iteration token budget, base capacity, fresh-wave
burst cap, and post-fresh burst cap jointly for the model, prompt length, tool
latency distribution, and rollout batch size.

MiMo-V2.6-Flash on eight MI355X GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the fixed multi-turn workload below, prefer two independent TP4 rollout
replicas on an eight-GPU node. Both replicas serve simultaneously; assign each
trajectory to one replica and retain that assignment for its continuations.

The benchmark uses ``XiaomiMiMo/MiMo-V2.6-Flash-RL`` at revision
``5711b268169967567844e1e560e8a3966da959b1`` and the recovered vLLM runtime
``0.30.1rc1.dev143+g29468dde8`` with its ROCm/AITER attention overlays. The
workload contains 64 trajectories, four model turns per trajectory, 8188 measured
fresh prompt tokens, and 64 generated tokens per request. Generated text is
appended to continuation prompts; estimated uncached continuation work is
approximately 28 tokens. Exact tokenized continuation lengths vary slightly
with generated output. Arrivals are pipelined, with synthetic tool waits of
0.25, 0.5, 1, and 2 seconds, temperature zero, ``ignore_eos=true``, and streaming.
Wall time starts at the common trajectory launch and ends when all 64 complete.

All configurations use FCFS, prefix caching, FP8 KV cache, block size 16,
``max_model_len=16384``, ``gpu_memory_utilization=0.93``, and disabled async
scheduling. No backend request priority is sent. TP4 uses ``max_num_seqs=32``;
TP8 uses ``max_num_seqs=64``. Both use ``max_num_batched_tokens=32768``.

Each formal group follows at least five workload warmups, with the last three
having a coefficient of variation (CV) at most 1%, range at most 2%, and no new
FlyDSL kernel cache files. Warmup is excluded. Formal groups require three hot
repetitions, CV at most 2%, and no new kernel files. Sessions differ across
cases to avoid reusing trajectory prefixes. The selected TP8 configuration is
restarted and FIFO/admission cases are interleaved; TP4 references are repeated
at the end on the same node.

The final same-node results on ``crsuse2-m2m-212`` are:

.. list-table:: Full 64-trajectory rollout (warmup excluded)
   :header-rows: 1
   :widths: 23 32 22 12 11

   * - Configuration
     - Three wall times (s)
     - Mean +/- sample std (s)
     - CV
     - Trajectory/s
   * - Two TP4 replicas
     - 9.885591, 9.963323, 9.972187
     - 9.940367 +/- 0.047644
     - 0.479%
     - 6.4384
   * - Single TP8, FCFS
     - 12.159869, 12.218751, 12.176857
     - 12.185159 +/- 0.030306
     - 0.249%
     - 5.2523
   * - Single TP8 + EP (exploratory)
     - 11.813304, 11.823091, 11.826492
     - 11.820962 +/- 0.006846
     - 0.058%
     - 5.4141
   * - Single TP4 reference
     - 16.777641, 16.712037, 16.731432
     - 16.740370 +/- 0.033703
     - 0.201%
     - 3.8231

Two TP4 replicas reduce wall time by 18.42% and increase trajectory throughput
by 22.58% versus the validated single TP8. Versus the single TP4 reference, they
reduce wall time by 40.62% and increase throughput by 68.41%. The earlier and
final TP4 baseline means differ by 0.19% for two replicas and 0.44% for one.
EP has stable timing, but remains exploratory for the quality reasons below.

.. list-table:: Supporting metrics from the same three repetitions
   :header-rows: 1
   :widths: 26 19 22 18 15

   * - Configuration
     - Fresh TTFT p95 (ms)
     - Continuation TTFT p95 (ms)
     - Computed prefill token/s
     - Decode token/s
   * - Two TP4 replicas
     - 3717.84
     - 177.21
     - 55049.29
     - 1648.25
   * - Single TP8, FCFS
     - 5594.26
     - 341.70
     - 44650.24
     - 1344.59
   * - Single TP8 + EP (exploratory)
     - 5133.21
     - 340.02
     - 46027.51
     - 1386.01
   * - Single TP4 reference
     - 7841.56
     - 2453.83
     - 32689.65
     - 978.71

TTFT includes router admission wait; p95 pools requests across repetitions.
There are zero preemptions in these groups. Prefix-cache hit rates are 74.34%
for TP4 and 74.48% for TP8. Backend queue means are 0.335/0.339 seconds for the
two replicas, 0.559 seconds for TP8, 0.510 seconds for EP, and 0.684 seconds for
single TP4. Mean utilization across the eight devices ranges from 91.10% to
92.84% for two TP4 replicas and 86.18% to 87.85% for TP8. Single TP4 uses four
cards at 93.78%-94.25%; the remaining cards are idle. These sampled utilization
values describe this rollout interval, not a sustained-training measurement.

The dual-replica benchmark explicitly partitions the trajectories into equal
sticky halves of 32, starts both from a common clock, and uses the unmodified
verl router for global admission. It measures simultaneous backend serving;
it does not validate live-trainer routing balance. At global admission capacity
64 this workload records zero burst admissions, so the dual-replica improvement
comes from replica parallelism. The single-TP4 reference uses the capacity-40,
fresh-wave-cap-1, resume-cap-20 configuration above.

With eight rollout GPUs, these parallelism settings create two TP4 replicas:

.. code:: yaml

   actor_rollout_ref:
     rollout:
       name: vllm
       tensor_model_parallel_size: 4
       data_parallel_size: 1
       pipeline_model_parallel_size: 1
       scheduling_policy: fcfs
       max_model_len: 16384
       max_num_seqs: 32
       max_num_batched_tokens: 32768
       gpu_memory_utilization: 0.93
       enable_prefix_caching: true
       router_config_path: /path/to/two_replica_router.yaml
       engine_kwargs:
         vllm:
           attention_backend: ROCM_AITER_DIFFKV
           kv_cache_dtype: fp8
           block_size: 16
           async_scheduling: false

The global router starting point is:

.. code:: yaml

   router_class: verl.workers.rollout.router.SoftAdmissionRequestLoadBalancer
   max_concurrent_requests: 64
   fresh_wave_max_resume_burst_requests: 2
   max_resume_burst_requests: 20
   fresh_max_wait_seconds: 60

The runtime environment shared by the cases is:

.. code:: bash

   export VLLM_ROCM_USE_AITER=1
   export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
   export HSA_NO_SCRATCH_RECLAIM=1
   export HSA_ENABLE_IPC_MODE_LEGACY=1
   export HIP_FORCE_DEV_KERNARG=1

The six-case pure-TP8 matrix tested sequence limits 32, 48, and 64 with token
budgets 32768 and 65536. The limit-64/token-budget-32768 configuration was the
fastest stable candidate; budget 65536 did not improve it. Adding
``NCCL_MIN_NCHANNELS=112``, ``GPU_MAX_HW_QUEUES=2``, and
``TORCH_BLAS_PREFER_HIPBLASLT=1`` did not reproduce a lower mean. One three-run
AMD-tuned group exceeded the 2% CV threshold and was rejected in full before
re-warmup and repetition.

TP8 router capacity 64 produces zero burst admissions for 64 trajectories.
Its three interleaved repetitions average 12.186351 seconds versus FIFO's
12.185159 seconds, providing no demonstrated benefit. Capacity 40 and 48
screening runs take 13.977991 and 14.484646 seconds. A separate capacity-40
active-burst/FIFO cross-check records 13.897098, 13.909017, and 14.439482 seconds
for burst versus 12.179957, 12.193477, and 12.191508 seconds for FIFO. The burst
group is rejected in full because its CV is 2.20%; it supplies no stable gain
or production recommendation. Retain FIFO for single TP8.

``--enable-expert-parallel`` was measured separately after a limited correctness
smoke evaluation. Base TP8 and EP both scored 24/24 on fixed-answer questions,
but five of eight deterministic free-generation prompts differed, including a
factual direction change. This does not establish an implementation error or
broad accuracy equivalence, so EP is excluded from the production recommendation.
For TP8/DP1/PCP1 without sequence parallel, this build partitions experts but
does not activate specialized MoE all-to-all kernels. Logs select
``AITER_MXFP4_BF16`` and ``MoEPrepareAndFinalizeNoDPEPModular``; TP all-reduce
dispatch uses ``QUICK_REDUCE``, ``AITER_CUSTOM``, and ``PYNCCL``, with ``PYNCCL``
on the EP group.

Strict continuation priority is intended for latency-sensitive experiments, not
as the default wall-time optimization. It can reduce continuation TTFT while
delaying fresh requests that would otherwise unlock later tool and model turns.
To experiment with it, configure
``EffectiveCostAdmissionRequestLoadBalancer`` and
``EffectiveCostPriorityPolicy``. The policy estimates request work as::

   estimated_cost = (
       prefill_weight * estimated_uncached_tokens
       + decode_weight * expected_output_tokens
         * (1 + prompt_tokens / decode_context_scale)
   )
   effective_cost = estimated_cost / (1 + wait_seconds / target_wait_seconds)

Lower effective cost maps to higher backend scheduling priority. vLLM requires
``scheduling_policy: priority``. SGLang requires
``enable_priority_scheduling: true`` and uses ``fcfs`` or ``lof`` as its base
schedule policy. verl translates the priority direction when SGLang keeps its
default higher-value-first behavior. For SGLang, configure:

.. code:: yaml

   actor_rollout_ref:
     rollout:
       name: sglang
       engine_kwargs:
         sglang:
           enable_priority_scheduling: true
           schedule_policy: fcfs

``schedule_low_priority_values_first: true`` is also supported. When it is
unset, verl negates its lower-value-first priority before sending the request so
the ordering remains consistent across vLLM and SGLang.

Next
----

- :doc:`Agentic RL Training<../start/agentic_rl>`: Quick start agentic RL training with gsm8k dataset.
- `LangGraph MathExpression <https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent/example>`_: Demonstrate how to use LangGraph to build agent loop.
- `Retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_: End-to-end retool paper reproduction using tool agent.
