# Single-period Product Distribution Problem

## Business Context

The single-period product distribution problem is the foundational static
scenario in the thesis. It represents one planning period of routine retail
replenishment for a large electric-vehicle retailer.

Within this period, production and inventory are treated as fixed inputs. The
company must decide how to allocate available SKU inventory from plants to
dealers and how to pack those shipments into vehicles provided by a
third-party logistics provider (3PL). The decision directly affects short-term
logistics cost and the company's ability to respond to end-market demand.

The model deliberately excludes cross-period production planning and dynamic
inventory transfer. This isolates the core relationships among inventory
allocation, vehicle-type selection, and loading optimization.

## System Structure

The two-level supply chain contains:

- multiple production plants;
- multiple dealers;
- multiple SKU types;
- a heterogeneous 3PL vehicle fleet;
- end customers served by the dealers.

For every SKU, multiple plants may be able to supply the same SKU. A single
plant may serve multiple dealers, and a single dealer may receive shipments
from multiple plants. The supply network therefore has a many-to-many
structure.

Horizontal transfers between plants are prohibited because of geographic
separation and long-distance transport costs.

## Decision Problem

The company must jointly determine:

- how much of each SKU is shipped from each plant to each dealer;
- whether a vehicle is used;
- which vehicle type is assigned;
- which plant-dealer shipment is loaded onto each vehicle;
- how individual SKU quantities are distributed across vehicles.

The objective is to minimize the total fee paid by the retailer to the 3PL.
Vehicle-type selection is associated with a fixed activation cost.

## Core Constraints

1. **Demand coverage**

   The total quantity of each SKU delivered to each dealer must fully satisfy
   that dealer's demand.

2. **Supply capacity**

   The quantity of a SKU shipped from a plant cannot exceed the inventory held
   by that plant. No additional production is available during the period.

3. **Vehicle capacity**

   The total volume loaded into a vehicle cannot exceed the capacity of its
   vehicle type.

4. **Minimum shipment quantity**

   If the loaded volume of a vehicle is below the required threshold, a
   penalty cost is incurred. This reflects the commercial rules of the 3PL and
   encourages higher vehicle utilization.

## Modeling Assumptions

1. **Sufficient aggregate supply**

   The total inventory of each SKU across plants is at least the total dealer
   demand. This avoids global infeasibility caused by stockout.

2. **Production silence**

   Production plans are ignored and plant inventory is treated as a static
   parameter for the single-period decision.

3. **One-dimensional measurement**

   SKU volume and vehicle capacity use a linear volume scale. Weight,
   stacking, loading order, and complex geometric constraints are ignored.

4. **Unlimited vehicle availability**

   The 3PL can provide an unlimited number of vehicles. Vehicle-type selection
   is driven only by cost.

## Problem Diagram

```mermaid
flowchart LR
    subgraph Plants["Supply side: multiple plants"]
        P1["Plant 1<br/>SKU inventory"]
        P2["Plant 2<br/>SKU inventory"]
        P3["Plant i<br/>SKU inventory"]
    end

    subgraph Logistics["Third-party logistics"]
        FLEET["Integrated loading and delivery optimization<br/>Multiple vehicle types<br/>Fixed vehicle activation cost<br/>Vehicle capacity<br/>Minimum shipment quantity"]
    end

    subgraph Dealers["Demand side: multiple dealers"]
        D1["Dealer 1<br/>Multiple SKU demands"]
        D2["Dealer 2<br/>Multiple SKU demands"]
        D3["Dealer j<br/>Multiple SKU demands"]
    end

    CUSTOMERS["End customers"]

    P1 --> FLEET
    P2 --> FLEET
    P3 --> FLEET
    FLEET --> D1
    FLEET --> D2
    FLEET --> D3
    D1 --> CUSTOMERS
    D2 --> CUSTOMERS
    D3 --> CUSTOMERS

    P1 -. "No horizontal transfer between plants" .- P2
    P2 -. "No horizontal transfer between plants" .- P3
```

The diagram shows a many-to-many supply structure. Each plant may supply
multiple dealers, and each dealer may receive the same SKU from multiple
plants. All shipments pass through the integrated loading and delivery
decision managed by the 3PL.

## Code Relationship

- `src/single_period_delivery/monolithic/` solves the direct
  integer-programming model.
- `src/single_period_delivery/column-generation/` implements master and
  pricing models for the column-generation method.
- `src/single_period_delivery/heuristic/` implements the two-stage heuristic
  for fast response.
