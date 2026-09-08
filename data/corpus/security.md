# Security, Compliance, and Data Residency
## Encryption
Data in transit uses TLS 1.2 or newer. Data at rest uses AES-256. Customer-managed keys (CMK) are available on Enterprise via AWS KMS, GCP Cloud KMS, or Azure Key Vault in the workspace home region.
## Data residency
Starter and Pro workspaces may choose `us-east-1`, `eu-west-1`, or `ap-southeast-1` at creation time. The home region cannot be changed later. Enterprise can add `eu-central-1` and `ap-northeast-1`. Object storage replicas stay inside the same geopolitical area: US, EU, or APAC. Backups never leave that area.
## Compliance
SOC 2 Type II and ISO 27001 reports are available to Pro and Enterprise customers under NDA. HIPAA BAA is offered only on Enterprise. PCI-DSS is not in scope; do not store raw cardholder data in Nimbus databases or logs.
## Retention
Application logs are retained 14 days on Starter, 30 days on Pro, and 90 days on Enterprise. Audit logs are retained 1 year on Pro and 7 years on Enterprise. Deleted workspaces enter a 30-day recoverable tombstone, then are cryptographically erased.