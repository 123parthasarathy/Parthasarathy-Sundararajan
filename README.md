# ARTEMIS: Adaptive Resource and Temporal Emergency Management Intelligent System

Enhanced implementation with PAHO MQTT real-time data integration for emergency dispatch optimization.

## Features

- **Real-time MQTT Data Streaming**: Integration with Eclipse Paho MQTT for live emergency dispatch data
- **Message Transformation**: Support for JSON, CSV, and binary message formats with city-specific transformers
- **STQM Module**: Spatial-Temporal Queuing Model for zone clustering
- **DLRP Module**: Response time prediction with multiple ML models
- **PRO Module**: Multi-objective resource optimization with Pareto front generation
- **Validated Results**: Proper cross-validation and honest metric reporting

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Batch Analysis Mode
```bash
python artemis_realtime.py --mode batch
```

### Real-time Streaming Mode
```bash
python artemis_realtime.py --mode realtime --broker broker.hivemq.com --duration 300
```

### Demo Mode
```bash
python artemis_realtime.py --mode demo
```

## Data Sources

- San Francisco Fire Department Calls for Service
- NYC EMS Incident Dispatch Data
- Seattle Real-Time Fire 911 Calls

## Author

Department of Mathematics, SRM Institute of Science and Technology