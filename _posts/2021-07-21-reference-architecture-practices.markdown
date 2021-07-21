---
layout: post
title:  "Reference Architecture Best Practices"
date:   2021-07-21 10:00:00 +0530
categories: it
---
## Introduction

Enterprise Architecture and System Design are part and parcel of Enterprise Software Development Lifecycle. Reference Architecture is the terminology typically used to refer the repeatable architecture patterns that will be used to create solution specific architectures. You can simply think the reference architecture as a standard blueprint to create solution specific architectures. 

An organisation can create Reference Architecture which would serve as a blueprint to create solution specific reference architectures patterns.  Is it confusing ? To simplify further imagine Reference Architecture as your Out-of-Box Word Document Templates, like News Letter Template and you use that to create, say CNN Specific News Letter Template which will be used by CNN Employees. 

## Five Pillars & Four Dimensions

The Reference Architecture can also be called as Architecture Framework. In this article I want to share few best practices I learnt based on my experiences. The Reference Architecture centers around the following five pillars and four dimensions:

**Five Pillars**
1. Security & Compliance
2. Reliability & Availability
3. Efficient Governance / Operations
4. Efficient Performance & Monitoring
5. Efficient Cost 

**Four Dimensions**
1. Personas & Roles
2. Partners & Suppliers
3. Information & Technology
4. Processes & Practices

![Five Pillars and Four Dimensions]({{site.baseurl}}/assets/img/5PillarsAnd4Dimensions.png)

As seen very clearly that the five pillars and 4 dimensions is the foundation for the reference architecture or the framework. All this is good, one question you may have in mind will be "Can I get more details on the five pillars which is mentioned here?". Let me try answering that question to the best of my knowledge here:
**Security and Compliance**
* Identity and Access Management
* Encryption & Cryptography
* Auditability & Traceability
* Security Information and Event Monitoring (SIEM)
* Network Layer 4/7 Security 
    * Firewall - IP Firewall, Web Application Firewall, Hardware Firewall
    * Shield from Distributed Denial of Service Attack (DDoS)
    * Secure DNS
    * Intrusion Prevention & Detection System (IPS/IDS)
* Application Images/Runtime Security
    * Endpoint Detection and Response (EDR)
    * Static & Dynamic Vulnaerability Scans
* Zero Trust Model
* Regulatories & Controls
    * Data Privacy & Protection
    * Data Residency
    * Auditable Compliance: ISO, NIST, PCI-DSS, HIPPA, SOC

**Reliability**
* Business Continuity: Disaster Recovery
* High Availability
* Scalability
* Resource Limits
* Config Change Traceability
* Backup & Restore
* Log Management & Archival

**Operations / Governance**
* Automated Operations (ZeroOps)
* Design for Failure & Threats (DevSecOps)
* Agile Deployment (Small & Frequent)
* Automated On-Boarding & Off-Boarding

**Performance & Monitoring**
* Performance & Capacity Monitoring
* API Monitoring & Profiling
* Edge Caching & Routing

**Cost**
* Right Technology Selection
* Relinquish unused resources immediately
* Right License Plan Selection
* Resource Sharing

**Personas and Roles**
* Every Architecture should clearly define the personas & their intended role/responsibilities. This is useful to provide data classification & Controls
* The types of personas would vary based on the solutions. For e.g. for a SaaS kind of solution, there could be two broader types:
    * Operational Personas
    * Application Consumer Personas
* Lot of factors could influence Personas & Roles
    * Roles, Responsibility, Department, Location,  Geography, Data & Technology Controls

**Partners and Suppliers**
* Every Architecture should leverage partners & supplier technologies to the maximum extent. For e.g. a cloud provider is a supplier of hosting infrastructure
* Integrating with partner & supplier technologies provides agility & faster time to market for the solution

**Processes and Practices**
* A strong governance model should always be a part of the architecture
* Define set of process/practices for manage & maintain the solution
* Security & Compliance practices should be part of architecture
* Process & Practices should always strive to achieve automation to the maximum extent and reduce human intervention

**Information and Technology**
* Information here represents the data that the solution collect, process/consume and present
* Technology here represents the technologies that are required for the solution and also that supports the development cycle of the same. 

## Diagrams

Another key component of the reference architecture will be diagrams and following are the key diagrams that I prefer to provide samples. 
1. Component Model Diagram
2. System Context Diagram
3. Deployment Model Diagram
4. Data Flow Diagram
5. Sequence Model
6. Threat Model

Kindly note that the sample diagrams provided in the reference architecture is primarily for reference as the actual diagrams will be based on solution implementation. 

## Composition

When you create reference architecture or architecture frameworks its very important to ensure the document provide clear guidelines, for anyone to create solution specific architecture or create architecture patterns/blueprints. Here are few tips:
1. Reference architectures should provide a common vocabulary, reusable designs, and industry best practices. They are not solution architectures, nor implemented directly. Rather, they are used as a constraint for more concrete architectures. 
2. Reference architectures are standardized architectures that provide a frame of reference. Typically it includes common architecture principles, patterns, building blocks and standards which enables a more consistent approach to solutioning 
3. Reference architectures should represent configurations of reference model elements created to address specific requirements based on specific set of principles.

If you are creating an organisation level Reference Architecture or Architecture Framework, then it should provide a very high level view and addresses organisation specific requirements and principles. Also its very important to ensure its solution agnostic as the expectation is for others to inherit this template to create a solution specific reference architecture or architecture itself. Hence its important the document provide enough flexibility for adoption without compromising the core principles & practices. For instance an organisation can define a framework, which can look as follows:
![Organisational Reference Architecture]({{site.baseurl}}/assets/img/SampleReferenceArchitecture.png)

The following view provides you a Pyramid Representation of Reference Architecture which would help to clearly understand why Reference Architecture is important and how it will help to standardise the solution architectures.
![Reference Architecture Pyramid]({{site.baseurl}}/assets/img/ReferenceArchitecturePyramid.png)

I hope these tips would help you to create better reference architecture documents at any level. 