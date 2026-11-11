#!/usr/bin/env bash
# Start/stop the 2-node Ray cluster for RL (verl schedules all 16 GPUs through Ray).
#
#   node0:  bash scripts/ray_cluster.sh head
#   node1:  bash scripts/ray_cluster.sh worker <node0_ip>
#   any:    bash scripts/ray_cluster.sh stop
# The repo (and the same Python env) must be at the same path on both nodes (shared FS):
# verl does not ship code to workers (runtime_env.working_dir is None).
source "$(dirname "$0")/common.sh"
case "${1:-}" in
  head)   ray start --head --port=6379 --num-gpus 8 --disable-usage-stats ;;
  worker) ray start --address="${2:?head ip}:6379" --num-gpus 8 ;;
  stop)   ray stop --force ;;
  *) die "usage: $0 head | worker <head_ip> | stop" ;;
esac
