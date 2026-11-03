# Kronkers · Deploying demand forecasts for a grocery business

Kronkers is a regional supermarket scenario from WGU. Fifteen analysts and data scientists work in several languages and share models through folders. With a 1.9% profit margin in the case, even a more accurate forecast has to justify its operating cost. A useful first question is whether it helps stores avoid stockouts or excess inventory in time for the next order.

## From a forecast to a store decision

The overnight forecast has a merchandising deadline. The design calls for validating incoming store data, tracking the model and dataset used for each forecast, and monitoring model quality alongside stockouts and overstock. A prediction service for product recommendations would instead need a response-time target and monitoring of purchase-through rates.

| Requirement | Proposed operation |
|---|---|
| Reproducible releases | Version the training data, code and model artifacts. Register approved versions centrally so teams can reproduce a result or roll back. |
| Detecting failures | Validate inputs, monitor drift by region and alert the operator when the data or forecasts change unexpectedly. |
| Keeping forecasts current | Retrain on a regular schedule. Evaluate a candidate against the deployed version before promoting it. |
| Scale and cost | Run batch forecasts within the overnight window and scale online capacity around demand. Track operating cost against the business benefit. |
| Governance | Restrict access to data and deployment controls and retain a record of model releases. |

The deliverable was a deployment design and set of requirements. An implementation would need operating targets agreed with store teams and measured business outcomes. The flight-delay pipeline in this repository is a separate implementation example.

*Source used in the coursework: Chip Huyen, Designing Machine Learning Systems (O’Reilly, 2022), pp. 26–33.*
