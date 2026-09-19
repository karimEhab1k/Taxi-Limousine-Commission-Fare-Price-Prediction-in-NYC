# NYC TLC Taxi Fare Price Prediction

A machine learning pipeline designed to predict dynamic taxi fare prices across New York City. Instead of relying on static formulas tied solely to distance and duration, this solution captures real-world market dynamics to optimize pricing for fleet operations and passengers alike.

## Overview

Traditional fare calculation relies heavily on deterministic base rates alongside standard per-mile and per-minute charges. However, fixed equations fail to account for real-time market friction, demand elasticity, and operational surcharges.

This project implements supervised regression models to predict trip fares by incorporating contextual temporal, spatial, and demand-driven features from the NYC Taxi & Limousine Commission (TLC) dataset.

## Predictive Signals

Rather than treating every mile equally, the feature pipeline extracts critical variables that influence trip pricing:

* **Demand & Congestion:** Flags rush-hour windows to account for elevated demand, dense traffic, and surge conditions.
* **Airport Transit Surcharges:** Identifies airport pickups and drop-offs (e.g., JFK, LaGuardia, Newark) that carry flat-rate policies or access fees.
* **Temporal Patterns:** Models variations across weekends, weekdays, and late-night travel windows.
* **Spatio-Temporal Routing:** Combines pickup/drop-off zone identifiers with trip duration and distance metrics.

## Business Impact

* **Dynamic Fare Estimation:** Delivers realistic, upfront pricing tailored to live route conditions.
* **Revenue Optimization:** Aligns fares with driver opportunity costs, operating overhead, and localized peak demand.
* **Production-Ready Pipeline:** Structured to ingest raw TLC trip records, execute preprocessing, and serve inferences for downstream services.
