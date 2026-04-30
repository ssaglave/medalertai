# Pittsburgh and Allegheny County Public Dispatch Source Digest

This local RAG source summarizes public, open web sources for Pittsburgh EMS,
Allegheny County 911, MPDS use, fire dispatch context, and Pennsylvania NEMSIS
reporting. Some Pittsburgh and Allegheny County web servers block scripted PDF
downloads from the development environment, so this digest preserves the
official source URLs and the key non-proprietary facts needed for assistant
retrieval. Verify detailed operational claims against the linked public
documents.

## Official Pittsburgh EMS Performance Audit

Source URL:
https://www.pittsburghpa.gov/files/assets/city/v/1/controller/documents/performance-audits/22221_emergency_medical_service_final_audit_2023.pdf

Source title: Performance Audit, Department of Public Safety, Bureau of
Emergency Medical Services, Office of City Controller, August 2023.

Relevant facts for RAG:

- Pittsburgh EMS responds to emergency calls dispatched from the Allegheny
  County Emergency Services 911 Communications Center.
- The audit describes the combined County and City 911 center as using CAD
  software that records call received time, ambulance on-scene arrival,
  departure to hospital, hospital arrival, and clearing time.
- The audit states that the 911 center uses the Medical Priority Dispatch
  System, or MPDS, to help dispatch medical emergencies.
- The audit describes MPDS as using standardized caller questions to determine
  priority status and assign a code for medical emergencies.
- The audit states that MPDS helps remove human bias and supports appropriate
  EMS response selection.
- The audit identifies Pittsburgh EMS call and response-time analysis as using
  CAD data and priority-code groupings rather than a public MPDS determinant
  codebook.
- The audit discusses high-priority ALS response-time analysis for
  overdose/poisoning, trauma, cardiac emergencies, and stroke.
- The audit references NFPA response-time standards as a benchmark, while also
  noting that federal or Pennsylvania law does not impose EMS agency response
  time requirements in the way the benchmark is used by the audit.

RAG caution:

- This source supports answers that Allegheny County 911 uses MPDS for
  Pittsburgh EMS dispatch and that CAD records EMS response timestamps.
- This source does not provide the proprietary IAED MPDS determinant codebook.
  The assistant should not invent exact MPDS determinant meanings from this
  audit alone.

## Official Pittsburgh EMS Operations Summary

Source URL:
https://www.pittsburghpa.gov/files/assets/city/v/1/public-safety/documents/24716_pittsburgh_ems_2023_operations_summary_report.pdf.pdf

Source title: Pittsburgh EMS 2023 Operations Summary.

Relevant facts for RAG:

- Use this report for Pittsburgh EMS operational context, service overview,
  annual activity, programs, and staffing or unit context.
- Use the EMS performance audit when the question is specifically about MPDS,
  CAD, dispatch workflow, response-time methodology, or priority-code analysis.

## Official Pittsburgh EMS CARES Cardiac Arrest Report

Source URL:
https://www.pittsburghpa.gov/files/assets/city/v/1/public-safety/documents/24848_2023_cares_report_v.2.pdf

Source title: Pittsburgh EMS 2023 CARES Cardiac Arrest Report.

Relevant facts for RAG:

- Use this source for Pittsburgh cardiac arrest outcome context and CARES
  reporting.
- Do not use this source to infer MPDS determinant-code meanings.

## Official Pittsburgh 911 Response Times and Wellness Audit

Source URL:
https://www.pittsburghpa.gov/files/assets/city/v/1/controller/documents/department-of-public-safety-911-response-times-and-wellness.final.pdf

Source title: Department of Public Safety 911 Response Times and Wellness
Audit.

Relevant facts for RAG:

- Use this source for city 911 response-time and communications-center wellness
  context.
- Pair with the EMS performance audit for EMS-specific dispatch and MPDS
  questions.

## Official Pittsburgh Bureau of Fire Performance Audit

Source URL:
https://www.pittsburghpa.gov/files/assets/city/v/1/controller/documents/performance-audits/23382_2023_fire_audit_pittsburgh.pdf

Source title: Pittsburgh Bureau of Fire 2023 Performance Audit.

Relevant facts for RAG:

- Use this source for Pittsburgh fire operations context.
- Do not treat this as a full Emergency Fire Dispatch or FPDS protocol card set.

## Allegheny County EMS and Fire Communications

Allegheny County EMS source URL:
https://www.alleghenycounty.us/Government/Police-and-Emergency-Services/Emergency-Services/Emergency-Management/Emergency-Medical-Services-EMS

Allegheny County Fire Communications source URL:
https://www.alleghenycounty.us/Government/Police-and-Emergency-Services/911-Communications/Fire-Communications

Allegheny County 2025 Operating Budget source URL:
https://www.alleghenycounty.us/files/assets/county/v/1/government/budget-amp-finance/operating-budget/2025-operating-budget.pdf

Relevant facts for RAG:

- Use Allegheny County pages for county EMS coordination, 911 communications,
  fire communications, and emergency-services structure.
- Use the county budget for broader emergency-services operations and dispatch
  implementation context.
- Do not treat county public pages as a complete EFD, FPDS, or MPDS protocol
  card source.

## Pennsylvania NEMSIS Context

Primary PA source URL:
https://www.pa.gov/content/dam/copapwp-pagov/en/health/documents/topics/documents/ems/EMSIB%202023-15%20NEMSIS%20v3.5.0%20Data%20Collection.pdf

NEMSIS Pennsylvania page:
https://nemsis.org/state-data-managers/state-map-v3/pennsylvania/

NEMSIS v3.5.0 data dictionary:
https://nemsis.org/media/nemsis_v3/release-3.5.0/DataDictionary/PDFHTML/EMSDEMSTATE/NEMSISDataDictionary.pdf

Relevant facts for RAG:

- Pennsylvania uses NEMSIS v3.5.0 reporting for EMS data collection.
- Use the PA EMS bulletin for Pennsylvania-specific NEMSIS transition context.
- Use the NEMSIS v3.5.0 data dictionary for field definitions, element names,
  and standard data meanings.
- Use the NEMSIS Pennsylvania page for state implementation and state-data
  manager context.

## Proprietary Protocol Boundary

MPDS, EFD, and FPDS protocol card text and determinant-code definitions are
licensed materials. The assistant may answer from public documents that a
jurisdiction uses MPDS or fire dispatch protocols, but it must not fabricate
determinant-code meanings or full protocol instructions unless those exact
materials are present in the approved RAG corpus.
