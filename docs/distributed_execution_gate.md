# Distributed execution gate

Status: blocked pending validated requirement. This project supports one local
coordinator and one SQLite study file. SQLite terminal trials remain source of
truth for local studies. Do not place a study file on NFS, SMB/CIFS,
cloud-synced, or other shared filesystems.

Distributed execution is not a scheduler switch. It requires a separately
operated database, durable ownership, and recovery semantics. No distributed
implementation may begin until every required field below has an approved
value.

## Requirement record

| Decision | Required value | Status |
| --- | --- | --- |
| Users/workload | Concrete workload and why one local coordinator is insufficient | Unset |
| Workers | Peak concurrent workers, processes per host, node count | Unset |
| Scale | Trials/day, expected trial duration, max queued/running trials | Unset |
| Network | Worker-to-database topology, latency, partitions, firewall/TLS | Unset |
| Database | Operated PostgreSQL owner, HA/backup/restore and retention policy | Unset |
| Security | AuthN/AuthZ, tenant isolation, secret delivery, audit requirements | Unset |
| Failure model | Worker crash, coordinator crash, DB outage, network partition behavior | Unset |
| Shutdown | Drain deadline, cancellation policy, orphaned-work policy | Unset |
| Operations | On-call owner, monitoring, alerting, migration and incident runbooks | Unset |
| Cost | PostgreSQL and operational cost explicitly accepted | Unset |

## Non-negotiable semantic contract

- External backend: Optuna RDB with PostgreSQL. Shared SQLite is unsupported.
- Trial state is durable: `WAITING`, `RUNNING`, then one terminal state.
- Claim/attempt has worker identity, lease expiry, heartbeat, attempt number.
- Execution is at-least-once. Recovered work may run again.
- Terminal commit is exactly-once/idempotent per attempt; duplicate reports do
  not create duplicate terminal records.
- Stale leases become recoverable only after configured expiry. Recovery emits
  auditable attempt history.
- Graceful shutdown stops new claims, heartbeats in-flight work until deadline,
  then leaves it for lease recovery.

## Gate evidence

Approve only when all requirement-record fields are set and both claims hold:

1. Measured or contractual workload exceeds single-coordinator capacity.
2. PostgreSQL operational cost is accepted by its owner.

Then create ADR with chosen topology and values, implement in this order:

1. PostgreSQL/Optuna RDB integration and explicit distributed API.
2. Lease/heartbeat/attempt schema plus idempotent terminal commit.
3. Worker lifecycle, stale-lease recovery, and graceful shutdown.
4. Multi-process and multi-node chaos tests: crash, partition, duplicate
   delivery, DB outage, lease expiry, and coordinator restart.

Until approval: retain current local scheduler and SQLite policy.
