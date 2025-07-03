Implementing data/ Directory - Part 1
The data/ directory holds our application's static configuration files. We will define config.yml and introduce two new crucial files for our expanded RBAC and corporate-grade features: rbac_capabilities.yaml and tiers.yaml.

1. data/config.yml
This file contains general application settings, such as LLM defaults, RAG parameters, and web scraping configurations.
data/config.yml
Jul 2, 7:12 AM

Open

2. data/rbac_capabilities.yaml (NEW FILE)
This new file is critical for implementing granular, role-based capabilities. Instead of hardcoding what each tier or role can do, this YAML file will define which capabilities are enabled for different roles. This makes the system much more flexible and configurable without changing code.
data/rbac_capabilities.yaml
Jul 2, 7:12 AM

Open

3. data/tiers.yaml (NEW FILE)
This new file will define the tiers and their hierarchy. This is a more explicit way to manage the tier hierarchy than hardcoding it in main_app.py.
data/tiers.yaml
Jul 2, 7:12 AM

Open

Next Step: We need to update utils/user_manager.py to load and use these new rbac_capabilities.yaml and tiers.yaml files, making get_user_tier_capability more robust and flexible. We will also need to update main_app.py to load the tiers.yaml for its TIER_HIERARCHY.


Implementing Dynamic RBAC - Part 5: Update data/rbac_capabilities.yaml and data/tiers.yaml (Initial Seed Data)
These updates will serve as the initial default configurations that will be loaded into Firestore when the application first runs or when an admin explicitly seeds them.

1. data/rbac_capabilities.yaml (Updated with New Capabilities)
data/rbac_capabilities.yaml (Updated with New Capabilities - Seed Data)
Jul 3, 2:49 AM

Open

2. data/tiers.yaml (Updated with New Tiers and Capabilities)
data/tiers.yaml (Updated with New Tiers and Capabilities - Seed Data)
Jul 3, 2:49 AM
