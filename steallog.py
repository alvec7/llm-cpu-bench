"""Фоновый лог загрузки CPU: steal, busy, iowait, loadavg раз в секунду -> logs/steal.jsonl"""
import json, time
def cpu():
    return list(map(int, open("/proc/stat").readline().split()[1:9]))
f = open("steal.jsonl", "a")
a = cpu()
while True:
    time.sleep(1)
    b = cpu(); d = [y - x for x, y in zip(a, b)]; t = sum(d) or 1; a = b
    f.write(json.dumps({"t": round(time.time(), 1), "steal": round(100 * d[7] / t, 2),
                        "busy": round(100 * (t - d[3] - d[4]) / t, 1), "iowait": round(100 * d[4] / t, 2),
                        "load1": float(open("/proc/loadavg").read().split()[0])}) + "\n")
    f.flush()
