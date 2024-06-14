import requests
import json
import sseclient
import itertools
from time import time
from random import randint
from tqdm.auto import tqdm
from pathlib import Path


params_dict = {
    # not sorted on purpose, to order jobs in a custom way
    "cache_4bit": [False, True],
    "flash_attn": [True, False],
    "promptlen": [128] + [1024 * x for x in (1, 2, 3, 5, 8, 11, 15)],
}

# FORCE_APPEND = True
FORCE_APPEND = False

n_tries = 2
max_retries = 4
n_ctx = 9 * 1024
max_tokens = 256
min_tokens = max_tokens - 16
max_gen_time = 60  # s
min_gen_tokens = 11  # 1st isn't counted for gen time
results_file = Path(f"results/exl2_speed_results_{int(time()):d}.json")
previous_result_files = list(Path("results").glob("exl2_speed_results_*.json"))

headers = {"Content-Type": "application/json"}


def print_f(output, **kwargs):
    tqdm.write(str(output) if output is not None else "", **kwargs)


def url(suffix):
    return "http://0.0.0.0:5000/" + suffix


def postrequest_f(url_str, json_dict=None, **kwargs):
    return requests.post(
        url(url_str), json=json_dict, headers=headers, verify=False, **kwargs
    )


def params_to_key(p):
    return tuple(v for k, v in sorted(p.items()) if k in params_dict)


results = []
if len(previous_result_files) > 0:
    with open(previous_result_files[-1], "r") as f:
        results = json.load(f)
        print_f(f"Loaded: {previous_result_files[-1].stem}")

already_computed = set(params_to_key(p) for p in results)

to_compute = []
for combination in itertools.product(*params_dict.values()):
    params = dict(zip(params_dict.keys(), combination))
    key = params_to_key(params)
    if not FORCE_APPEND and key not in already_computed:
        to_compute.append(params)
    elif FORCE_APPEND:
        to_compute.append(params)
        if key in already_computed:
            print_f(f"!!! WARNING: will append a duplicate of {key}")

if len(to_compute) == 0:
    raise RuntimeError("Nothing to do.")


def gen_random_prompt(promptlen, vocabsize=128256):
    prompt_tokens = [randint(0, vocabsize - 1) for _ in range(promptlen)]
    output = postrequest_f("v1/internal/decode", {"tokens": prompt_tokens})
    return output.json()["text"]


def load_model(model_name, flash_attn, n_ctx, cache_4bit=False):
    print_f(f"(Re)Loading model: 4 bit cache: {cache_4bit}, fa: {flash_attn}")
    model_data = {
        "model_name": model_name,
        "args": {
            "no_flash_attn": not flash_attn,
            "loader": "ExLlamav2_HF",
            "cache_4bit": cache_4bit,
            "max_seq_len": n_ctx,
        },
        "settings": {},
    }
    model_load_response = postrequest_f("v1/internal/model/load", model_data)
    if model_load_response.status_code != 200:
        print_f("")
        raise RuntimeError(
            f"Can't load the model! Response: {model_load_response.status_code}"
        )
    print_f("   ...Done!")


def gen_json_f(prompt):
    return {
        "add_bos_token": False,
        "ban_eos_token": True,
        "do_sample": False,
        "max_tokens": max_tokens,
        "prompt": prompt,
        "seed": 123,
        "skip_special_tokens": False,
        "stop": [],
        "stream": False,
    }


current_model_params = None
print_f(to_compute[0].keys())
progressbar = tqdm(total=len(to_compute) * n_tries)

for params in to_compute:
    model_params = tuple(
        (
            params["cache_4bit"],
            params["flash_attn"],
        )
    )

    if model_params != current_model_params:
        current_model_params = model_params
        load_model(
            cache_4bit=params["cache_4bit"],
            flash_attn=params["flash_attn"],
            model_name="bartowski_Meta-Llama-3-8B-Instruct-exl2_8_0",
            n_ctx=n_ctx,
        )
        postrequest_f("v1/completions", {**gen_json_f("Test"), "max_tokens": 1})

    times_tried, successful_gens = 0, 0
    max_gentps, max_prompttps = 0, 0

    while times_tried < max_retries and successful_gens < n_tries:
        times_tried += 1
        prompt = gen_random_prompt(params["promptlen"])
        gen_json = gen_json_f(prompt)
        gen_aborted = False

        # PROCESS PROMPT
        starttime = time()
        response = postrequest_f("v1/completions", {**gen_json, "max_tokens": 1})
        prompttime = time()

        # GEN TOKENS
        with postrequest_f(
            "v1/completions", {**gen_json, "stream": True}, stream=True
        ) as stream_response:
            client = sseclient.SSEClient(stream_response)

            completion_tokens = 0
            for event in client.events():
                if completion_tokens == 0:
                    firsttokentime = time()
                completion_tokens += 1
                if (
                    completion_tokens >= min_gen_tokens
                    and time() - firsttokentime > max_gen_time
                ):
                    gen_aborted = True
                    break

            endtime = time()

        # CLEANUP
        if gen_aborted:
            postrequest_f("v1/internal/stop-generation")
            prompt_tokens = params["promptlen"]
        else:
            completion_tokens = json.loads(event.data)["usage"]["completion_tokens"]
            prompt_tokens = json.loads(event.data)["usage"]["prompt_tokens"]
            if completion_tokens < min_tokens:
                print_f("Got short reply, retrying...")
                continue

        successful_gens += 1
        progressbar.update()

        gen_seconds = endtime - firsttokentime
        gentps = (completion_tokens - 1) / gen_seconds
        overhead_seconds = firsttokentime - prompttime - 1 / gentps
        prompt_seconds = prompttime - starttime - 1 / gentps
        prompttps = prompt_tokens / prompt_seconds

        if prompttps > max_prompttps:
            max_prompttps = prompttps
            max_prompt_seconds = prompt_seconds
            max_overhead_seconds = overhead_seconds
            max_prompt_tokens = prompt_tokens
        if gentps > max_gentps:
            max_gentps = gentps
            max_gen_seconds = gen_seconds
            max_completion_tokens = completion_tokens
            max_prompt_tokens = prompt_tokens

    if max_gentps == 0:
        continue

    results.append(
        dict(
            sorted(
                {
                    **params,
                    "prompt_tokens": max_prompt_tokens,
                    "completion_tokens": max_completion_tokens,
                    "overhead_seconds": max_overhead_seconds,
                    "prompt_seconds": max_prompt_seconds,
                    "gen_seconds": max_gen_seconds,
                    "prompttps": max_prompttps,
                    "gentps": max_gentps,
                }.items()
            )
        )
    )

    print_f(
        f"\nprompt {max_prompt_seconds:.2f} s, {max_prompttps:.3f} t/s, "
        f"gen {max_gentps:.3f} t/s, {1000/max_gentps:.1f} ms/t "
        f" - {list(params.values())}"
    )

    with open(results_file, "w") as f:
        json.dump(sorted(results, key=lambda x: tuple(sorted(x.items()))), f, indent=1)

progressbar.close()
postrequest_f("v1/internal/model/unload")
