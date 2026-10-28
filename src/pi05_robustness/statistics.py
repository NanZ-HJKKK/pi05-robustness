"""Episode statistics; exploratory intervals do not measure training-seed variance."""
from collections import defaultdict
import math
import random
import statistics


def wilson(successes, total, z=1.959963984540054):
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    p = successes / total
    d = 1 + z*z/total
    center = (p+z*z/(2*total))/d
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0,center-half), min(1,center+half)]


def summarize(rows):
    if not rows:
        raise ValueError("No completed episodes; cannot report an empirical success rate")
    if any(r["status"] != "complete" or type(r["success"]) is not bool for r in rows):
        raise ValueError("Invalid episode status or success value")
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate episode")
    successes = sum(r["success"] for r in rows)
    tasks = defaultdict(list)
    for row in rows:
        tasks[(row["suite"],row["task_name"])].append(int(row["success"]))
    samples = [t for r in rows for t in r.get("inference",[]) if not t["warmup"]]
    result = {"episodes":len(rows), "tasks":len(tasks), "successes":successes,
        "success_rate":successes/len(rows), "wilson_95":wilson(successes,len(rows)),
        "task_macro_success_rate":statistics.mean(statistics.mean(x) for x in tasks.values()),
        "measured_requests_after_warmup":len(samples)}
    if samples:
        for key in ["sampling_ms","policy_wall_ms","request_wall_ms"]:
            values = [t[key] for t in samples]
            result["median_"+key] = statistics.median(values)
        result["amortized_action_throughput_hz"] = rows[0]["replan_steps"]*1000/result["median_sampling_ms"]
        result["request_amortized_action_throughput_hz"] = rows[0]["replan_steps"]*1000/result["median_request_wall_ms"]
    elapsed = sum(r.get("episode_wall_seconds",0) for r in rows)
    if elapsed > 0:
        result["simulation_actions_per_wall_second_including_setup_and_video"] = sum(r["environment_steps"] for r in rows)/elapsed
    periods=[]
    for row in rows:
        timing=row.get("inference",[])
        for current,next_sample in zip(timing,timing[1:]):
            if current["warmup"] or next_sample["warmup"]:
                continue
            if "request_start_elapsed_seconds" not in current or "request_start_elapsed_seconds" not in next_sample:
                continue
            interval=next_sample["request_start_elapsed_seconds"]-current["request_start_elapsed_seconds"]
            if not math.isfinite(interval) or interval<=0:
                raise ValueError("Invalid control feedback interval")
            periods.append(interval)
    if periods:
        result["feedback_intervals_after_warmup"]=len(periods)
        result["median_simulation_feedback_period_ms"]=statistics.median(periods)*1000
        result["median_simulation_feedback_hz"]=1/statistics.median(periods)
        result["feedback_scope"]="Wall-clock simulator request-to-request cycles including preprocessing, video append, inference and action execution; not physical robot frequency"
    timed=[r for r in rows if r.get("control_loop_wall_seconds",0)>0 and not any(t["warmup"] for t in r.get("inference",[]))]
    if timed:
        result["control_timed_episodes_without_warmup"]=len(timed)
        result["simulation_actions_per_control_second_without_warmup"]=sum(r["environment_steps"] for r in timed)/sum(r["control_loop_wall_seconds"] for r in timed)
    return result


def paired_comparison(control, treatment, seed=0, bootstraps=2000, rgb_exception=None):
    """Task-cluster paired bootstrap and exact two-sided McNemar diagnostic."""
    def index(rows):
        result = {}
        for row in rows:
            if row["pair_id"] in result:
                raise ValueError("Duplicate paired episode")
            result[row["pair_id"]] = row
        return result
    a, b = index(control), index(treatment)
    common = sorted(a.keys() & b.keys())
    if not common:
        raise ValueError("No matched episodes")
    groups = defaultdict(list)
    improved = regressed = 0
    applied=[]
    for key in common:
        x,y = a[key],b[key]
        for field in ["initial_state_sha256","settled_sim_state_sha256","initial_main_rgb_sha256","initial_wrist_rgb_sha256","bddl_sha256","prompt","seed"]:
            if x[field] != y[field]:
                if (rgb_exception and field=="initial_main_rgb_sha256" and key==rgb_exception["pair_id"]
                        and {x[field],y[field]}==set(rgb_exception["raw_hashes"])
                        and rgb_exception.get("status")=="audited_bounded_rgb_exception"):
                    applied.append({"pair_id":key,"field":field,"raw_hashes":rgb_exception["raw_hashes"]})
                    continue
                raise ValueError("Pair mismatch: " + field + " " + key)
        delta = int(y["success"]) - int(x["success"])
        improved += delta == 1
        regressed += delta == -1
        groups[(x["suite"],x["base_task"])].append(delta)
    discordant = improved+regressed
    pvalue = min(1, 2*sum(math.comb(discordant,k) for k in range(min(improved,regressed)+1))/2**discordant) if discordant else 1
    rng, clusters, draws = random.Random(seed),list(groups.values()),[]
    for _ in range(bootstraps):
        sample = [value for cluster in rng.choices(clusters,k=len(clusters)) for value in cluster]
        draws.append(statistics.mean(sample)*100)
    draws.sort()
    return {"paired_episodes":len(common),"control_unpaired":len(a)-len(common),"treatment_unpaired":len(b)-len(common),
            "treatment_minus_control_pp":100*(improved-regressed)/len(common),"improved":improved,"regressed":regressed,
            "base_task_clusters":len(clusters),"cluster_bootstrap_95_pp":[draws[int(.025*bootstraps)],draws[min(bootstraps-1,int(.975*bootstraps))]],
            "mcnemar_exact_p_diagnostic":pvalue,"strict_pixel_identity":not applied,"audited_rgb_exceptions":applied,
            "note":"Task-cluster bootstrap is exploratory; McNemar assumes independent pairs; neither measures training seed variance. Any bounded RGB exception is disclosed separately; it does not certify strict pixel identity."}
