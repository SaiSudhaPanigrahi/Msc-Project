from google.cloud import storage

SERVICE_ACCOUNT_FILE = 'c:/Users/USER/msc-project-owner.json'


storage_client = storage.Client.from_service_account_json( SERVICE_ACCOUNT_FILE)

# Make an authenticated API request
buckets = list(storage_client.list_buckets())
print(buckets)