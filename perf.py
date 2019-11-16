import time

total_latency = 0
inferences = 0
avg_inference_latency = 0

def calc_inference_latency(inference_start, inference_end):
    global total_latency
    global inferences
    global avg_inference_latency

    last_inference_latency = inference_end - inference_start
    total_latency += last_inference_latency

    inferences += 1
    avg_inference_latency = total_latency / inferences

    # return in milliseconds
    return avg_inference_latency * 1000
