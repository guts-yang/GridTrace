# OpsWarden FAQ

> 运维值班常见问题与标准处置流程。基于真实工单整理。

## Q1. 服务出现 5xx 错误率飙升，如何快速定位？
Category: incident
Page: 1

1. 打开 Grafana 面板 `Service-5xx-Rate`，确认时间窗与告警一致。
2. 切换到对应服务 Dashboard 的 "Upstream" 面板，查看上游依赖错误率。
3. 若上游正常，检查 `app.log` 中 ERROR 级别堆栈定位异常类。
4. 若是发布后立即出现：`kubectl rollout undo deploy/<name> -n <ns>` 回滚。
5. 若是 DB 慢查询导致：`SELECT * FROM pg_stat_activity WHERE state='active' ORDER BY now()-query_start DESC LIMIT 10;` 找长事务。

## Q2. 容器 OOMKilled 怎么排查？
Category: container
Page: 2

1. `kubectl describe pod <pod>` 看到 `Last State: Terminated, Reason: OOMKilled`。
2. `kubectl top pod <pod> --containers` 看实际内存使用。
3. 检查 JVM/Python 应用 heap dump（`jmap -dump:format=b,file=heap.hprof <pid>`）。
4. 短期方案：调大 `resources.limits.memory`；长期方案：分析内存泄漏。

## Q3. PostgreSQL 复制延迟过大如何处理？
Category: database
Page: 3

1. 备机执行 `SELECT now() - pg_last_xact_replay_timestamp() AS lag;`
2. 若 lag > 30s：
   - 检查 `pg_stat_replication` 中 `replay_lag` 字段
   - 备机是否有长事务：`SELECT * FROM pg_stat_activity WHERE state='active' ON replica;`
   - 备机磁盘 IO：`iostat -x 1`
3. 必要时提升 `wal_receiver_timeout` 或重建备机。

## Q4. 如何重置一个被锁定的用户账号？
Category: account
Page: 4

```sql
UPDATE users
SET status = 'active',
    failed_login_count = 0,
    locked_until = NULL
WHERE user_id = :uid;
```
同时记录到审计表：
```sql
INSERT INTO account_unlock_log (user_id, unlocked_by, reason)
VALUES (:uid, :operator, :reason);
```

## Q5. Nginx 502 Bad Gateway 突然增多
Category: network
Page: 5

1. `systemctl status nginx` — 确认 worker 进程数。
2. `tail -f /var/log/nginx/error.log` — 查看 upstream 连接错误。
3. 检查 upstream 健康：`curl -I http://<upstream>/health`
4. 若 upstream 是 K8s service：检查 Endpoints 是否有 IP：
   `kubectl get endpoints <svc> -n <ns>`

## Q6. K8s Pod 一直处于 Pending 状态
Category: k8s
Page: 6

1. `kubectl describe pod <pod>` 查看 Events。
2. 常见原因：节点资源不足 → 检查 `kubectl describe nodes` 的 Allocatable。
3. PVC 未绑定：`kubectl get pvc` 看状态。
4. nodeSelector / toleration 不匹配：检查 `nodeName` 字段。
5. imagePullSecrets 缺失或镜像不存在。

## Q7. Redis 连接池耗尽报警
Category: cache
Page: 7

1. 检查应用日志中 Redis 超时异常。
2. `redis-cli -h <host> INFO clients` 查看 connected_clients。
3. `redis-cli CLIENT LIST | wc -l` 实时连接数。
4. 若确实高并发：调大应用侧连接池 `max-connections`。
5. 排查是否有 `KEYS *`、`FLUSHDB` 等慢命令阻塞。

## Q8. CI/CD 流水线卡在 "Waiting for runner"
Category: cicd
Page: 8

1. 检查 GitLab/GitHub runner 状态面板。
2. `gitlab-runner list` 查看注册状态。
3. 离线 runner：`gitlab-runner verify --delete` 后重新注册。
4. K8s executor runner：检查 `runner-*` pod 状态。
5. 共享 runner 抢占：考虑自建 runner。

## Q9. Prometheus 抓取失败 (context deadline exceeded)
Category: monitoring
Page: 9

1. 检查 target 是否 up：`/api/v1/targets`。
2. 网络连通性：`curl -v http://<target>:<port>/metrics`。
3. 抓取超时：prometheus.yml 中 `scrape_timeout: 15s` 调大。
4. target 端是否开启 basic auth / TLS：补 `basic_auth` / `tls_config`。
5. 大量 label cardinality：检查 target 端 `/metrics` 输出。

## Q10. 业务方反馈接口慢，如何快速诊断？
Category: performance
Page: 10

1. 拿到 traceId 后 APM（如 SkyWalking）查看整条调用链。
2. 找到耗时最长 span → 是 DB / RPC / 缓存？
3. DB 慢：`EXPLAIN ANALYZE` 对应 SQL，看是否有索引缺失。
4. RPC 慢：检查下游 provider 监控。
5. 缓存命中：检查 key 设计 & TTL。
