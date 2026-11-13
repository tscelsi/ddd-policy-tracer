What questions are we trying to answer:

tracing influence from ORGS to government members:

1. PERSON makes a CLAIM, are there any very similar/related CLAIM made by any other PERSON|ORGS?
2. What STANCE does MEMBER|ORG have on TOPIC
3. Given MEMBER has STANCE on TOPIC, do any ORGS have similar STANCE on TOPIC?
   1. e.g. Given Peter Dutton has a negative stance on net-zero ambitions, do any think-tanks have similar stances?
4. MEMBER opposes this bill on social housing, do any ORGS have recommendations about social housing that support the Member's stance?

## Ontology Definition

Entities

Person

Organization
DocumentChunk
Topic
Legislation


Document - [:DISCUSSES] -> Topic
Document - [:SUBMITTED_TO] -> Inquiry
Inquiry - [:ORGANISED_BY] -> Committee


Person - [:PUBLISHES] -> Document
Organisation - [:PUBLISHES] -> Document
Committee - [:PUBLISHES] -> Document

Person - [:MEMBER_OF] -> Committee
Person - [:MEMBER_OF] -> Organisation


Document types:

1. Speech
2. Report
3. Committee Submission
4. Committee Resolution