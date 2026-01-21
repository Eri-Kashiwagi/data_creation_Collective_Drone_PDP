#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDPTW benchmark generator (Li&Lim-like) + energy-driven radial rings (y*=1/2/3)

你要的核心点：
1) 6 类：LC/LR/LRC × tight/loose  -> LC1 LC2 LR1 LR2 LRC1 LRC2
2) 每个实例里混合 y*=1/2/3（默认比例 50/30/20，可改）
3) y* 只由能耗决定；不允许超重：q <= single_Q (= CAP/group_size)
4) 输出 txt：9 列，与你 Instance.read_pdp_benchmark() 读取格式一致

标签定义（只看能耗）：
y* = min{y in {1,2,3} : depot->P->D->depot 的能耗 <= B}

实现方式：先按目标 y 采样半径环带，再用接受-拒绝校验 y* 是否匹配。
时间窗 tight/loose 在几何之后生成，并确保 direct route 不被 due 卡死（必要时只放宽 due）。
"""

from __future__ import annotations
import argparse, json, math, os, random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------
# Energy model (matches your common.calc_energy form)
# e = alpha * (w_base + load/y)^(3/2) * travel_time / (3600*1000)
# In Li&Lim-style instances, travel_time = distance (speed is a placeholder).
# -----------------------------

def calc_alpha(g: float, rho: float, blade_area: float, rotor_height: float) -> float:
    return math.sqrt((g ** 3) / (2.0 * rho * blade_area * rotor_height))

def seg_energy_kwh(alpha: float, w_base: float, load: float, y: int, t: float) -> float:
    return alpha * ((w_base + load / y) ** 1.5) * t / (3600.0 * 1000.0)

def route_energy_kwh(alpha: float, w_base: float, q: float, y: int, d0p: float, dpd: float, dd0: float) -> float:
    # depot->P empty + P->D loaded + D->depot empty
    return (
        seg_energy_kwh(alpha, w_base, 0.0, y, d0p) +
        seg_energy_kwh(alpha, w_base, q,   y, dpd) +
        seg_energy_kwh(alpha, w_base, 0.0, y, dd0)
    )

def min_feasible_y_energy(alpha: float, w_base: float, B: float, q: float, d0p: float, dpd: float, dd0: float) -> int:
    for y in (1, 2, 3):
        if route_energy_kwh(alpha, w_base, q, y, d0p, dpd, dd0) <= B + 1e-12:
            return y
    return 4

# -----------------------------
# Ring geometry
# Place P and D on same radius r, so you get a clean “ring” structure.
# Control dist(P,D) by pd_ratio: dist ≈ pd_ratio * r
# For same radius: dist = 2r sin(Δθ/2) => Δθ = 2 asin(pd_ratio/2)
# -----------------------------

def delta_theta(pd_ratio: float) -> float:
    pr = max(1e-6, min(1.999, pd_ratio))
    return 2.0 * math.asin(min(0.999999, pr / 2.0))

def ref_R(alpha: float, w_base: float, B: float, q: float, y: int, pd_ratio: float) -> float:
    # Assume d0p=r, dd0=r, dpd=pd_ratio*r => energy = K*r
    K = alpha * (2.0 * (w_base ** 1.5) + pd_ratio * ((w_base + q / y) ** 1.5)) / (3600.0 * 1000.0)
    if K <= 1e-18:
        return float("inf")
    return B / K

def ring_interval(alpha: float, w_base: float, B: float, q: float, pd_ratio: float, domain_max: float, target_y: int) -> Tuple[float, float, Dict[str, float]]:
    R1 = ref_R(alpha, w_base, B, q, 1, pd_ratio)
    R2 = ref_R(alpha, w_base, B, q, 2, pd_ratio)
    R3 = ref_R(alpha, w_base, B, q, 3, pd_ratio)

    R1e, R2e, R3e = min(R1, domain_max), min(R2, domain_max), min(R3, domain_max)

    # small safety margins; R2/R1 and R3/R2 can be close
    eps = 0.015
    if target_y == 1:
        lo, hi = 0.05 * R1e, (1.0 - eps) * R1e
    elif target_y == 2:
        lo, hi = (1.0 + eps) * R1e, (1.0 - eps) * R2e
    elif target_y == 3:
        lo, hi = (1.0 + eps) * R2e, (1.0 - eps/2.0) * R3e
    else:
        raise ValueError("target_y must be 1/2/3")

    dbg = {"R1": R1, "R2": R2, "R3": R3, "domain_max": domain_max}
    return lo, hi, dbg

def polar_to_xy(depot: Tuple[float, float], r: float, theta: float) -> Tuple[float, float]:
    return (depot[0] + r * math.cos(theta), depot[1] + r * math.sin(theta))

# -----------------------------
# Spatial patterns: clustered/random/mixed (Li&Lim-style)
# We implement clustering on ANGLE (theta). Radius is controlled by rings.
# This matches “radial diffusion” much better than Euclidean Gaussian clusters.
# -----------------------------

@dataclass
class SpatialCfg:
    pattern: str           # "C" / "R" / "RC"
    K: int = 4             # number of angle clusters
    sigma_theta: float = 0.35  # angle noise (radians)
    p_cluster: float = 0.5

def sample_theta(pattern: str, rng: random.Random, centers: List[float], sigma: float, p_cluster: float) -> float:
    if pattern == "R":
        return rng.uniform(0.0, 2.0 * math.pi)
    if pattern == "C":
        c = centers[rng.randrange(len(centers))]
        return (rng.gauss(c, sigma)) % (2.0 * math.pi)
    if pattern == "RC":
        if rng.random() < p_cluster:
            c = centers[rng.randrange(len(centers))]
            return (rng.gauss(c, sigma)) % (2.0 * math.pi)
        return rng.uniform(0.0, 2.0 * math.pi)
    raise ValueError("pattern must be C/R/RC")

# -----------------------------
# Time windows: tight vs loose
# Generate precedence-aware windows, then ensure direct route isn't killed by due time.
# (We only relax due, not geometry -> keep y* energy-driven.)
# -----------------------------

@dataclass
class TWCfg:
    name: str
    tau: float
    base_p: int
    base_d: int
    wait_pd_max: int
    horizon: int
    service: int = 90

def gen_time_windows(rng: random.Random, d0p: float, dpd: float, dd0: float, tw: TWCfg) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    Lp = max(30, int(tw.base_p * tw.tau))
    Ld = max(30, int(tw.base_d * tw.tau))

    buffer = int(d0p + dpd + dd0 + 2 * tw.service + 30)
    latest = max(0, tw.horizon - buffer)
    u_p = rng.randint(0, latest) if latest > 0 else 0

    wait = rng.randint(0, tw.wait_pd_max)
    u_d = int(u_p + dpd + wait + tw.service)

    e_p = int(u_p - 0.3 * Lp); l_p = int(u_p + 0.7 * Lp)
    e_d = int(u_d - 0.3 * Ld); l_d = int(u_d + 0.7 * Ld)

    e_p = max(0, min(tw.horizon, e_p)); l_p = max(0, min(tw.horizon, l_p))
    e_d = max(0, min(tw.horizon, e_d)); l_d = max(0, min(tw.horizon, l_d))
    if l_p < e_p: l_p = e_p
    if l_d < e_d: l_d = e_d
    return (e_p, l_p), (e_d, l_d)

def relax_due_if_needed(d0p: float, dpd: float, dd0: float, tw_p: Tuple[int,int], tw_d: Tuple[int,int], service: int) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    t = 0.0
    t += d0p
    if t > tw_p[1]:
        tw_p = (tw_p[0], int(math.ceil(t)))
    t = max(t, tw_p[0]) + service

    t += dpd
    if t > tw_d[1]:
        tw_d = (tw_d[0], int(math.ceil(t)))
    # waiting allowed
    return tw_p, tw_d

# -----------------------------
# Demand: not overweight (q <= single_Q), but keep it near the top so energy effect is visible.
# -----------------------------

def sample_demand(rng: random.Random, single_Q: float, step: int) -> int:
    q = rng.uniform(0.65 * single_Q, 1.0 * single_Q)
    q = max(step, int(round(q / step) * step))
    # ensure never exceed single_Q due to rounding
    q = min(q, int(math.floor(single_Q / step) * step))
    return max(step, q)

def choose_target_y(rng: random.Random, p1: float, p2: float, p3: float) -> int:
    s = p1 + p2 + p3
    p1, p2, p3 = p1/s, p2/s, p3/s
    r = rng.random()
    if r < p1: return 1
    if r < p1 + p2: return 2
    return 3

def generate_one_order(rng: random.Random,
                       alpha: float, w_base: float, B: float,
                       depot: Tuple[float,float], domain_max: float,
                       single_Q: float, step: int,
                       pd_ratio: float,
                       spatial: SpatialCfg, theta_centers: List[float],
                       tw_cfg: TWCfg,
                       target_y: int,
                       max_tries: int = 700) -> Dict:
    dtheta = delta_theta(pd_ratio)

    for _ in range(max_tries):
        q = sample_demand(rng, single_Q, step)

        r_lo, r_hi, dbg = ring_interval(alpha, w_base, B, q, pd_ratio, domain_max, target_y)
        if not (r_lo + 1e-6 < r_hi):
            continue

        r = rng.uniform(r_lo, r_hi)

        theta_p = sample_theta(spatial.pattern, rng, theta_centers, spatial.sigma_theta, spatial.p_cluster)
        P = polar_to_xy(depot, r, theta_p)

        # delivery angle = theta_p +/- dtheta (with small noise)
        noise = rng.uniform(-0.15, 0.15) * dtheta
        dt = dtheta + noise
        theta_d = theta_p + dt if rng.random() < 0.5 else theta_p - dt
        D = polar_to_xy(depot, r, theta_d)

        d0p = r
        dd0 = r
        dpd = 2.0 * r * math.sin(abs(dt) / 2.0)

        y_star = min_feasible_y_energy(alpha, w_base, B, q, d0p, dpd, dd0)
        if y_star != target_y:
            continue

        tw_p, tw_d = gen_time_windows(rng, d0p, dpd, dd0, tw_cfg)
        tw_p, tw_d = relax_due_if_needed(d0p, dpd, dd0, tw_p, tw_d, tw_cfg.service)

        return {"q": q, "P": P, "D": D, "tw_p": tw_p, "tw_d": tw_d, "y": target_y, "dbg": dbg}

    raise RuntimeError("Failed to generate order. Try --auto_grid or adjust B/pd_ratio.")

# -----------------------------
# TXT output (your 9-column format)
# -----------------------------

def write_txt(path: str, header_drone_num: int, CAP: int, speed: int, depot: Tuple[float,float], horizon: int, service: int, orders: List[Dict]) -> None:
    n = len(orders)
    lines = []
    lines.append(f"{header_drone_num}\t{CAP}\t{speed}")
    lines.append(f"0\t{int(round(depot[0]))}\t{int(round(depot[1]))}\t0\t0\t{horizon}\t0\t0\t0")

    # pickups 1..n
    for i, od in enumerate(orders, start=1):
        px, py = od["P"]; q = od["q"]; e, l = od["tw_p"]
        d_id = n + i
        lines.append(f"{i}\t{int(round(px))}\t{int(round(py))}\t{q}\t{e}\t{l}\t{service}\t0\t{d_id}")

    # deliveries n+1..2n
    for i, od in enumerate(orders, start=1):
        dx, dy = od["D"]; q = od["q"]; e, l = od["tw_d"]
        p_id = i; d_id = n + i
        lines.append(f"{d_id}\t{int(round(dx))}\t{int(round(dy))}\t{-q}\t{e}\t{l}\t{service}\t{p_id}\t0")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="bench_pdptw_energy_rings")
    ap.add_argument("--K", type=int, default=10, help="instances per class per scale")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--small", type=int, default=10)
    ap.add_argument("--medium", type=int, default=30)
    ap.add_argument("--large", type=int, default=50)

    ap.add_argument("--header_drone_num", type=int, default=50)
    ap.add_argument("--CAP", type=int, default=200)
    ap.add_argument("--group_size", type=int, default=6)
    ap.add_argument("--speed", type=int, default=1)
    ap.add_argument("--demand_step", type=int, default=10)

    ap.add_argument("--B", type=float, default=2.0)
    ap.add_argument("--w_drone", type=float, default=20.0)
    ap.add_argument("--w_battery", type=float, default=10.0)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--air_density", type=float, default=1.225)
    ap.add_argument("--blade_area", type=float, default=0.5)
    ap.add_argument("--rotor_height", type=float, default=0.2)

    ap.add_argument("--pd_ratio", type=float, default=1.15, help="bigger => harder energy => more y=2/3")
    ap.add_argument("--grid", type=float, default=140.0)
    ap.add_argument("--auto_grid", action="store_true")

    ap.add_argument("--p_y1", type=float, default=0.50)
    ap.add_argument("--p_y2", type=float, default=0.30)
    ap.add_argument("--p_y3", type=float, default=0.20)
    return ap.parse_args()

def auto_grid(args: argparse.Namespace, alpha: float, w_base: float, B: float, single_Q: float) -> float:
    if not args.auto_grid:
        return args.grid
    q_ref = 0.95 * single_Q
    R3 = ref_R(alpha, w_base, B, q_ref, 3, args.pd_ratio)
    # want domain_max ~= grid/2 - margin >= 1.1*R3
    need = 2.0 * (1.1 * R3 + 5.0)
    return float(max(args.grid, math.ceil(need)))

def main() -> None:
    args = parse_args()
    master_rng = random.Random(args.seed)

    single_Q = args.CAP / float(args.group_size)
    alpha = calc_alpha(args.gravity, args.air_density, args.blade_area, args.rotor_height)
    w_base = args.w_drone + args.w_battery

    grid = auto_grid(args, alpha, w_base, args.B, single_Q)
    depot = (grid / 2.0, grid / 2.0)
    domain_max = grid / 2.0 - 3.0

    # time-window configs
    tw_tight = TWCfg(name="tight", tau=0.65, base_p=220, base_d=280, wait_pd_max=30, horizon=1351, service=90)
    tw_loose = TWCfg(name="loose", tau=1.15, base_p=260, base_d=320, wait_pd_max=60, horizon=1351, service=90)

    # spatial configs
    spatial_C = SpatialCfg(pattern="C", K=4, sigma_theta=0.35, p_cluster=1.0)
    spatial_R = SpatialCfg(pattern="R", K=4, sigma_theta=0.35, p_cluster=0.0)
    spatial_RC = SpatialCfg(pattern="RC", K=4, sigma_theta=0.35, p_cluster=0.5)

    theta_centers = [2.0 * math.pi * k / spatial_C.K for k in range(spatial_C.K)]

    classes = [
        ("LC1", spatial_C, tw_tight),
        ("LC2", spatial_C, tw_loose),
        ("LR1", spatial_R, tw_tight),
        ("LR2", spatial_R, tw_loose),
        ("LRC1", spatial_RC, tw_tight),
        ("LRC2", spatial_RC, tw_loose),
    ]
    scales = [("small", args.small), ("medium", args.medium), ("large", args.large)]

    os.makedirs(args.out, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "grid": grid,
        "depot": [depot[0], depot[1]],
        "CAP": args.CAP,
        "group_size": args.group_size,
        "single_Q": single_Q,
        "energy": {"B_kWh": args.B, "alpha": alpha, "w_base": w_base, "pd_ratio": args.pd_ratio},
        "y_mix": {"p1": args.p_y1, "p2": args.p_y2, "p3": args.p_y3},
        "layout": "scale/{LC1..LRC2}/*.txt",
    }

    for scale_name, n_orders in scales:
        for cls_name, spatial, tw in classes:
            out_dir = os.path.join(args.out, scale_name, cls_name)
            os.makedirs(out_dir, exist_ok=True)

            for k in range(1, args.K + 1):
                inst_seed = args.seed + 100000 * k + (hash((scale_name, cls_name)) % 10007)
                rng = random.Random(inst_seed)

                orders = []
                hist = {1: 0, 2: 0, 3: 0}

                for _ in range(n_orders):
                    target_y = choose_target_y(rng, args.p_y1, args.p_y2, args.p_y3)
                    od = generate_one_order(
                        rng=rng,
                        alpha=alpha, w_base=w_base, B=args.B,
                        depot=depot, domain_max=domain_max,
                        single_Q=single_Q, step=args.demand_step,
                        pd_ratio=args.pd_ratio,
                        spatial=spatial, theta_centers=theta_centers,
                        tw_cfg=tw, target_y=target_y
                    )
                    orders.append(od)
                    hist[target_y] += 1

                txt_name = f"{cls_name}_{n_orders}_{k}.txt"
                txt_path = os.path.join(out_dir, txt_name)
                write_txt(txt_path, args.header_drone_num, args.CAP, args.speed, depot, tw.horizon, tw.service, orders)

                # debug json (optional)
                meta = {"seed": inst_seed, "class": cls_name, "scale": scale_name, "n_orders": n_orders,
                        "y_hist": hist, "grid": grid, "pd_ratio": args.pd_ratio}
                with open(txt_path.replace(".txt", ".json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                print(f"[OK] {scale_name}/{cls_name}/{txt_name} y_hist={hist}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved to: {args.out}")

if __name__ == "__main__":
    main()
