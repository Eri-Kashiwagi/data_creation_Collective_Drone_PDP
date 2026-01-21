#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

# 在无图形界面的环境中也能保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 计算能耗模型中的常数 alpha。
def calc_alpha(g: float, rho: float, blade_area: float, rotor_height: float) -> float:
    return math.sqrt((g**3) / (2.0 * rho * blade_area * rotor_height))


# 单段飞行能耗（kWh），与载荷和编队规模有关。
def seg_energy_kwh(alpha: float, w_base: float, load: float, y: int, t: float) -> float:
    return alpha * ((w_base + load / y) ** 1.5) * t / (3600.0 * 1000.0)


# 直达路线 depot->P->D->depot 的总能耗。
def route_energy_kwh(
    alpha: float, w_base: float, q: float, y: int, d0p: float, dpd: float, dd0: float
) -> float:

    return (
        seg_energy_kwh(alpha, w_base, 0.0, y, d0p)
        + seg_energy_kwh(alpha, w_base, q, y, dpd)
        + seg_energy_kwh(alpha, w_base, 0.0, y, dd0)
    )


# 找到满足能耗预算 B 的最小 y（1/2/3）。
def min_feasible_y(
    alpha: float, w_base: float, B: float, q: float, d0p: float, dpd: float, dd0: float
) -> int:
    for y in (1, 2, 3):
        if route_energy_kwh(alpha, w_base, q, y, d0p, dpd, dd0) <= B + 1e-12:
            return y
    return 4


# 参考半径：将能耗线性化后反推的 R（用于画环）。
def ref_R(
    alpha: float, w_base: float, B: float, q: float, y: int, pd_ratio: float
) -> float:

    K = (
        alpha
        * (2.0 * (w_base**1.5) + pd_ratio * ((w_base + q / y) ** 1.5))
        / (3600.0 * 1000.0)
    )
    if K <= 1e-18:
        return float("inf")
    return B / K


# 单个节点记录（depot/pickup/delivery）。
@dataclass
class Node:
    node_id: int
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float
    pick: int
    deli: int


# 解析后的实例对象（头信息 + 节点集合）。
@dataclass
class Instance:
    path: str
    header_drone_num: int
    CAP: float
    speed: float
    depot: Node
    pickups: Dict[int, Node]
    deliveries: Dict[int, Node]
    n: int


# 读取并解析 PDPTW 文本实例。
def parse_instance(txt_path: str) -> Instance:

    # 先过滤空行，避免格式混乱。
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f.readlines() if ln.strip()]

    # 至少应包含 header、depot 和一个节点。
    if len(raw) < 3:
        raise ValueError(f"File too short: {txt_path}")

    h = raw[0].split()

    # 允许 header 用浮点写法，先转 float 再转 int。
    header_drone_num = int(float(h[0]))
    CAP = float(h[1])
    speed = float(h[2])

    # 逐行解析节点，先汇总到列表。
    nodes: List[Node] = []
    for ln in raw[1:]:
        parts = ln.split()
        if len(parts) != 9:
            raise ValueError(f"Bad line (need 9 cols): {ln}")
        nodes.append(
            Node(
                node_id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                demand=float(parts[3]),
                ready=float(parts[4]),
                due=float(parts[5]),
                service=float(parts[6]),
                pick=int(parts[7]),
                deli=int(parts[8]),
            )
        )

    # 第一个节点必须是 depot。
    depot = nodes[0]
    if depot.node_id != 0:
        raise ValueError(f"First node after header should be depot id=0 in {txt_path}")

    # 总数必须满足 1 + 2n。
    if (len(nodes) - 1) % 2 != 0:
        raise ValueError(f"Node count not 1+2n in {txt_path}: got {len(nodes)} nodes")
    n = (len(nodes) - 1) // 2

    # 根据编号范围划分 pickup 与 delivery。
    pickups: Dict[int, Node] = {}
    deliveries: Dict[int, Node] = {}
    for node in nodes[1:]:
        if 1 <= node.node_id <= n:
            pickups[node.node_id] = node
        elif (n + 1) <= node.node_id <= 2 * n:
            deliveries[node.node_id] = node

    # pickup / delivery 数量必须一一对应。
    if len(pickups) != n or len(deliveries) != n:
        raise ValueError(
            f"Pickup/delivery mismatch in {txt_path}: {len(pickups)}/{len(deliveries)} expected {n}/{n}"
        )

    # 封装成 Instance 返回。
    return Instance(
        path=txt_path,
        header_drone_num=header_drone_num,
        CAP=CAP,
        speed=speed,
        depot=depot,
        pickups=pickups,
        deliveries=deliveries,
        n=n,
    )


# 欧氏距离，同时作为 travel_time 使用。
def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def plot_instance(
    inst: Instance,
    out_png: str,
    B: float,
    w_drone: float,
    w_battery: float,
    gravity: float,
    air_density: float,
    blade_area: float,
    rotor_height: float,
    group_size: float,
    pd_ratio_for_rings: float,
    draw_reference_rings: bool,
    draw_pair_lines: bool,
    annotate_y: bool,
) -> None:
    # 预计算能耗常量。
    alpha = calc_alpha(gravity, air_density, blade_area, rotor_height)

    # 基础重量 = 机体 + 电池。
    w_base = w_drone + w_battery

    # 参考载重（用于画环）。
    single_Q = inst.CAP / float(group_size)

    # depot 坐标提前取出。
    depot_xy = (inst.depot.x, inst.depot.y)

    y_star: Dict[int, int] = {}
    # 统计 y* 分布。
    hist = {1: 0, 2: 0, 3: 0, 4: 0}

    for i in range(1, inst.n + 1):
        # 取出一对 P-D 节点。
        P = inst.pickups[i]
        D = inst.deliveries[inst.n + i]
        # 订单需求量。
        q = P.demand

        # 三段距离：去程 / 中段 / 返程。
        d0p = dist(depot_xy, (P.x, P.y))
        dpd = dist((P.x, P.y), (D.x, D.y))
        dd0 = dist((D.x, D.y), depot_xy)

        # 计算最小可行 y。
        y = min_feasible_y(alpha, w_base, B, q, d0p, dpd, dd0)
        y_star[i] = y
        hist[y] = hist.get(y, 0) + 1

    # 建立正方形画布，避免坐标拉伸。
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")

    # 用星号突出 depot。
    ax.scatter([inst.depot.x], [inst.depot.y], marker="*", s=160, label="Depot")

    px = [inst.pickups[i].x for i in range(1, inst.n + 1)]
    py = [inst.pickups[i].y for i in range(1, inst.n + 1)]
    dx = [inst.deliveries[inst.n + i].x for i in range(1, inst.n + 1)]
    dy = [inst.deliveries[inst.n + i].y for i in range(1, inst.n + 1)]

    # 用不同形状区分 pickup / delivery。
    ax.scatter(px, py, marker="o", s=22, label="Pickups")
    ax.scatter(dx, dy, marker="x", s=22, label="Deliveries")

    if draw_pair_lines:
        # 连接 P-D 配对线。
        for i in range(1, inst.n + 1):
            P = inst.pickups[i]
            D = inst.deliveries[inst.n + i]
            ax.plot([P.x, D.x], [P.y, D.y], linewidth=0.8)

    if annotate_y:
        # 在 pickup 旁标出 y*。
        for i in range(1, inst.n + 1):
            P = inst.pickups[i]
            ax.text(P.x, P.y, str(y_star[i]), fontsize=7)

    if draw_reference_rings:
        # 参考载重略小于上限，环更稳健。
        q_ref = 0.95 * single_Q
        R1 = ref_R(alpha, w_base, B, q_ref, 1, pd_ratio_for_rings)
        R2 = ref_R(alpha, w_base, B, q_ref, 2, pd_ratio_for_rings)
        R3 = ref_R(alpha, w_base, B, q_ref, 3, pd_ratio_for_rings)
        for R, lab in [(R1, "R1 (y=1)"), (R2, "R2 (y=2)"), (R3, "R3 (y=3)")]:
            if math.isfinite(R) and R > 0:
                c = plt.Circle(depot_xy, R, fill=False, linewidth=1.2, label=lab)
                ax.add_patch(c)

    # 自动缩放边界并留白。
    xs = [inst.depot.x] + px + dx
    ys = [inst.depot.y] + py + dy
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 0.06 * max(maxx - minx, maxy - miny, 1.0)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    # 标题包含实例名与 y* 统计。
    title = os.path.basename(inst.path)
    subtitle = f"n={inst.n} | y*: {hist.get(1,0)}/{hist.get(2,0)}/{hist.get(3,0)} (1/2/3) | B={B}kWh | single_Q≈{single_Q:.2f}"
    ax.set_title(title + "\n" + subtitle)

    # 图例 + 网格，便于观察。
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linewidth=0.6)

    # 确保输出目录存在并保存图片。
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# 收集输入路径下的所有 .txt 文件。
def list_txt_files(inp: str) -> List[str]:

    if os.path.isfile(inp) and inp.lower().endswith(".txt"):
        return [inp]
    txts: List[str] = []

    for root, _, files in os.walk(inp):
        for fn in files:
            if fn.lower().endswith(".txt"):
                txts.append(os.path.join(root, fn))
    txts.sort()
    return txts


# CLI 入口：解析参数并生成图片。
def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="单个 .txt 文件，或包含 txt 的目录（递归）")
    ap.add_argument("--out", default="viz_out", help="输出 PNG 的目录")

    ap.add_argument("--B", type=float, default=2.0)
    ap.add_argument("--w_drone", type=float, default=20.0)
    ap.add_argument("--w_battery", type=float, default=10.0)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--air_density", type=float, default=1.225)
    ap.add_argument("--blade_area", type=float, default=0.5)
    ap.add_argument("--rotor_height", type=float, default=0.2)

    ap.add_argument("--group_size", type=float, default=6.0)

    ap.add_argument("--pd_ratio", type=float, default=1.15)

    ap.add_argument("--no_rings", action="store_true", help="不画参考环")
    ap.add_argument("--no_lines", action="store_true", help="不画 P-D 配对连线")
    ap.add_argument("--annotate_y", action="store_true", help="在 pickup 点旁标 y*（点多会挤）")
    ap.add_argument("--limit", type=int, default=0, help="限制处理的文件数量（0=不限制）")
    args = ap.parse_args()

    # 获取待处理的 txt 列表。
    files = list_txt_files(args.input)
    if not files:
        raise SystemExit("No .txt files found.")

    # 需要时限制处理数量。
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    for fpath in files:
        inst = parse_instance(fpath)

        if os.path.isdir(args.input):
            rel = os.path.relpath(fpath, args.input)
        else:
            rel = os.path.basename(fpath)
        rel_noext = os.path.splitext(rel)[0]
        out_png = os.path.join(args.out, rel_noext + ".png")

        # 调用绘图函数。
        plot_instance(
            inst=inst,
            out_png=out_png,
            B=args.B,
            w_drone=args.w_drone,
            w_battery=args.w_battery,
            gravity=args.gravity,
            air_density=args.air_density,
            blade_area=args.blade_area,
            rotor_height=args.rotor_height,
            group_size=args.group_size,
            pd_ratio_for_rings=args.pd_ratio,
            draw_reference_rings=(not args.no_rings),
            draw_pair_lines=(not args.no_lines),
            annotate_y=args.annotate_y,
        )
        print("[OK]", out_png)


if __name__ == "__main__":
    main()
