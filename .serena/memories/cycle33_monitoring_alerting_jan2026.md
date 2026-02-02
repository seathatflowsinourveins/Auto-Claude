# Cycle 33: Monitoring & Alerting Patterns (January 25, 2026)

## Research Sources
- "FastAPI Observability Lab with Prometheus and Grafana" (Dec 2025)
- "Application Monitoring Best Practices: Complete Guide for 2026" (Jan 2026)
- "OpsGenie vs PagerDuty: Comparison for 2026" (Jan 2026)
- "5 Best AI-Powered Incident Management Platforms 2026" (Dec 2025)
- "Four Golden Signals for SRE Monitoring" (Mar 2025)
- Prometheus instrumentation documentation

---

## 1. SRE MONITORING FRAMEWORKS

### Four Golden Signals (Google SRE)

| Signal | Definition | Example Metric |
|--------|------------|----------------|
| **Latency** | Time to serve request | `http_request_duration_seconds` |
| **Traffic** | Demand on system | `http_requests_total` |
| **Errors** | Failed requests rate | `http_requests_total{status=~"5.."}` |
| **Saturation** | Resource utilization | `process_resident_memory_bytes` |

**Best For**: User-facing services, APIs, web applications

### RED Method (Microservices)

| Metric | Definition |
|--------|------------|
| **Rate** | Requests per second |
| **Errors** | Failed requests per second |
| **Duration** | Distribution of request latency |

```python
# RED metrics implementation
REQUEST_RATE = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_ERRORS = Counter('http_request_errors_total', 'Failed requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'Latency', ['method', 'endpoint'])
```

**Best For**: Microservices, request-driven workloads

### USE Method (Infrastructure)

| Metric | Definition |
|--------|------------|
| **Utilization** | % time resource busy |
| **Saturation** | Queue length, waiting work |
| **Errors** | Error count |

**Best For**: CPUs, memory, disks, network interfaces

---

## 2. PROMETHEUS INSTRUMENTATION

### Metric Types

```python
from prometheus_client import Counter, Gauge, Histogram, Summary

# Counter: Only goes up (requests, errors, bytes sent)
REQUESTS = Counter('http_requests_total', 'Total requests', ['method', 'status'])

# Gauge: Can go up or down (temperature, connections, queue size)
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Current active connections')

# Histogram: Bucketed latency distribution (p50, p90, p99)
LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Summary: Streaming quantiles (less accurate, lower memory)
RESPONSE_SIZE = Summary('response_size_bytes', 'Response size')
```

### FastAPI Integration

```python
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time

app = FastAPI()

# Define metrics
REQUEST_COUNT = Counter(
    'fastapi_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'fastapi_request_duration_seconds',
    'Request latency',
    ['method', 'endpoint']
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        endpoint = request.url.path
        
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        return response

app.add_middleware(PrometheusMiddleware)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### Flask Integration

```python
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Add /metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})
```

### Decorator Pattern

```python
from functools import wraps
from prometheus_client import Histogram
import time

FUNCTION_LATENCY = Histogram(
    'function_duration_seconds',
    'Function execution time',
    ['function_name']
)

def observe_latency(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            FUNCTION_LATENCY.labels(function_name=func.__name__).observe(
                time.perf_counter() - start
            )
    return wrapper

@observe_latency
async def process_data(data):
    ...
```

---

## 3. GRAFANA DASHBOARDS

### Essential Panels

```yaml
# Service Overview Dashboard
panels:
  - title: Request Rate
    type: graph
    query: rate(http_requests_total[5m])
    
  - title: Error Rate
    type: graph
    query: |
      sum(rate(http_requests_total{status=~"5.."}[5m])) 
      / sum(rate(http_requests_total[5m])) * 100
    
  - title: P99 Latency
    type: graph
    query: histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m]))
    
  - title: Active Connections
    type: stat
    query: active_connections
```

### Dashboard JSON Structure

```json
{
  "dashboard": {
    "title": "Service Health",
    "panels": [
      {
        "title": "Request Rate",
        "type": "timeseries",
        "targets": [{
          "expr": "sum(rate(http_requests_total[5m])) by (endpoint)",
          "legendFormat": "{{endpoint}}"
        }],
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
      }
    ]
  }
}
```

### SLO Dashboard Pattern

```promql
# Availability (% successful requests)
sum(rate(http_requests_total{status!~"5.."}[30d]))
/ sum(rate(http_requests_total[30d])) * 100

# Error Budget Remaining
1 - (
  sum(increase(http_requests_total{status=~"5.."}[30d]))
  / (sum(increase(http_requests_total[30d])) * (1 - 0.999))
)
```

---

## 4. ALERTING PATTERNS

### Alert Rule Structure

```yaml
# prometheus/rules/alerts.yml
groups:
  - name: service-alerts
    rules:
      # High Error Rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) 
          / sum(rate(http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      # High Latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, rate(request_latency_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency above 1s"
```

### Alert Severity Levels

| Severity | Response Time | Examples |
|----------|--------------|----------|
| **Critical** | Immediate (page) | Service down, data loss, security breach |
| **Warning** | Business hours | Degraded performance, approaching limits |
| **Info** | Next standup | Informational, capacity planning |

### Alert Fatigue Prevention

```yaml
# GOOD: Symptom-based alert (user impact)
- alert: APIErrorRateHigh
  expr: error_rate > 1%
  for: 5m

# BAD: Cause-based alert (noisy)
- alert: CPUHigh
  expr: cpu_usage > 80%
  # This fires constantly but may not affect users
```

**Rules to Reduce Fatigue**:
1. Alert on symptoms (user-facing), not causes
2. Set appropriate `for` duration (avoid flapping)
3. Use multi-window alerts for SLOs
4. Aggregate related alerts (don't page per-instance)
5. Auto-resolve alerts when condition clears

---

## 5. INCIDENT MANAGEMENT PLATFORMS (2026)

### Platform Comparison

| Platform | Best For | Key Features | Pricing |
|----------|----------|--------------|---------|
| **PagerDuty** | Enterprise | AI triage, event orchestration | $$$$ |
| **incident.io** | Slack-native | AI copilot, retrospectives | $$$ |
| **Rootly** | AI-first | Auto-remediation, runbooks | $$$ |
| **FireHydrant** | Process-focused | Statuspage, postmortems | $$ |
| **Squadcast** | SMB | Affordable, full-featured | $ |

**IMPORTANT**: OpsGenie is being deprecated (sunset by Oct 2025-2026). Migrate away.

### PagerDuty Integration

```python
import requests

def create_pagerduty_incident(
    routing_key: str,
    summary: str,
    severity: str = "critical",
    source: str = "python-app"
):
    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": summary,
            "severity": severity,
            "source": source,
            "custom_details": {}
        }
    }
    
    response = requests.post(
        "https://events.pagerduty.com/v2/enqueue",
        json=payload
    )
    return response.json()
```

### Slack Alerting

```python
import httpx

async def send_slack_alert(
    webhook_url: str,
    title: str,
    message: str,
    severity: str = "warning"
):
    colors = {
        "critical": "#FF0000",
        "warning": "#FFA500",
        "info": "#0000FF"
    }
    
    payload = {
        "attachments": [{
            "color": colors.get(severity, "#808080"),
            "title": title,
            "text": message,
            "footer": "Monitoring System",
            "ts": int(time.time())
        }]
    }
    
    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json=payload)
```

---

## 6. SLO/SLI IMPLEMENTATION

### Definitions

- **SLI (Service Level Indicator)**: Metric measuring service quality
- **SLO (Service Level Objective)**: Target value for an SLI
- **SLA (Service Level Agreement)**: Contract with consequences

### Common SLIs

```python
# Availability SLI
availability = successful_requests / total_requests

# Latency SLI (proportion under threshold)
latency_sli = requests_under_200ms / total_requests

# Freshness SLI (data staleness)
freshness = data_age_seconds < threshold
```

### Error Budget

```python
# 99.9% availability = 0.1% error budget
error_budget = 1 - slo_target  # 0.001 for 99.9%

# Monthly error budget (in minutes)
monthly_budget_minutes = 30 * 24 * 60 * error_budget  # 43.2 minutes

# Current consumption
errors_this_month = sum(errors)
budget_consumed = errors_this_month / (total_requests * error_budget)
```

### Multi-Window Alert (Burn Rate)

```yaml
# Fast burn: 14.4x in 1 hour (2% budget consumed)
- alert: SLOFastBurn
  expr: |
    (
      sum(rate(http_requests_total{status=~"5.."}[1h]))
      / sum(rate(http_requests_total[1h]))
    ) > 14.4 * 0.001

# Slow burn: 6x in 6 hours (3% budget consumed)  
- alert: SLOSlowBurn
  expr: |
    (
      sum(rate(http_requests_total{status=~"5.."}[6h]))
      / sum(rate(http_requests_total[6h]))
    ) > 6 * 0.001
```

---

## 7. PRODUCTION MONITORING CHECKLIST

### Essential Metrics

```yaml
Application:
  - [ ] Request rate (by endpoint)
  - [ ] Error rate (by type)
  - [ ] Latency percentiles (p50, p90, p99)
  - [ ] Active connections
  - [ ] Request queue depth

Infrastructure:
  - [ ] CPU utilization
  - [ ] Memory usage (RSS, heap)
  - [ ] Disk I/O and usage
  - [ ] Network throughput
  - [ ] Container restarts

Business:
  - [ ] Orders per minute
  - [ ] Payment success rate
  - [ ] User signups
  - [ ] Feature usage
```

### Dashboard Organization

```
Top Level: Executive Overview
├── Service Health (Golden Signals)
├── SLO Burn Down
└── Business Metrics

Per-Service: Detailed Dashboards
├── Request/Response metrics
├── Dependency health
├── Resource utilization
└── Error breakdown
```

---

## 8. ANTI-PATTERNS TO AVOID

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Alerting on metrics, not symptoms | Alert fatigue | Alert on user-facing issues |
| Missing `for` duration | Flapping alerts | Set 5-10 min threshold |
| No runbooks | Slow resolution | Link runbook in alert |
| Percent-based thresholds only | Miss low traffic issues | Use absolute + percentage |
| No correlation IDs | Hard to trace | Add request_id to all logs/metrics |
| Unbounded label cardinality | Memory explosion | Limit to known values |

---

## Summary: Monitoring Hierarchy

```
1. SLOs FIRST: Define what "healthy" means
   - Availability target (99.9%)
   - Latency target (p99 < 200ms)

2. GOLDEN SIGNALS: Alert on user impact
   - Latency, Traffic, Errors, Saturation

3. DASHBOARDS: Drill down for investigation
   - Overview → Service → Instance → Request

4. ALERTING: Page only for real incidents
   - Symptoms, not causes
   - Include runbooks

5. INCIDENT MANAGEMENT: Structured response
   - On-call rotation
   - Communication channels
   - Postmortems
```

---

*Cycle 33 Complete - Monitoring & Alerting Patterns*
*Research Date: January 25, 2026*
