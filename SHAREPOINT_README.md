# SharePoint Folder Listing Script

This script lists the contents of a SharePoint folder using Azure AD authentication.

## Prerequisites

- Python 3.x
- `requests` library (already installed)
- **Client Secret** for your Azure AD application

## Configuration

The script is pre-configured with the following details from your Azure AD application:

- **Application (client) ID**: `a54eaddf-654d-4f0e-9071-2b8c8ad26942`
- **Object ID**: `58b6d500-72e5-46a2-9020-be4ff8c31be3`
- **Directory (tenant) ID**: `0138a897-ff88-4dc2-933b-fad359609873`
- **SharePoint Site**: `anesthesiapartners.sharepoint.com`
- **Folder Path**: `/sites/CodedWork9/Shared Documents/IAS/2025/MOR`

## Important: Client Secret Required

⚠️ **You need to obtain the Client Secret from your Azure AD application.**

To get the client secret:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** → **App registrations**
3. Find your application: **Leon MD Token**
4. Go to **Certificates & secrets**
5. Create a new client secret or use an existing one
6. Copy the secret value (you'll need to enter this when running the script)

## Usage

Run the script:

```bash
python3 list_sharepoint_folder.py
```

When prompted, enter your client secret.

## Output

The script will:

1. Authenticate with Azure AD
2. Get the SharePoint site ID
3. List all folders and files in the specified location
4. Display:
   - Folder names with creation and modification dates
   - File names with size, creation, and modification dates
5. Optionally save the results to a JSON file

## Example Output

```
SharePoint Folder Listing Script
================================================================================

1. Obtaining access token...
✅ Access token obtained successfully.

2. Getting SharePoint site ID...
✅ Site ID: anesthesiapartners.sharepoint.com,abc123...

3. Listing folder contents...

================================================================================
Folder Contents: /sites/CodedWork9/Shared Documents/IAS/2025/MOR
================================================================================

📁 FOLDERS (2):
--------------------------------------------------------------------------------
  📁 Subfolder1
     Created: 2025-01-15T10:30:00Z
     Modified: 2025-01-20T14:22:00Z

📄 FILES (5):
--------------------------------------------------------------------------------
  📄 document.xlsx
     Size: 45.23 KB
     Created: 2025-01-10T09:15:00Z
     Modified: 2025-01-18T16:45:00Z

================================================================================
Total: 2 folder(s), 5 file(s)
================================================================================
```

## Permissions Required

Your Azure AD application needs the following Microsoft Graph API permissions:

- `Sites.Read.All` or `Sites.ReadWrite.All`
- `Files.Read.All` or `Files.ReadWrite.All`

Make sure these permissions are granted and admin consent is provided.

## Troubleshooting

### Authentication Errors

- Verify your client secret is correct
- Check that your Azure AD application has the required permissions
- Ensure admin consent has been granted for the permissions

### Access Denied Errors

- Verify the application has access to the SharePoint site
- Check that the folder path is correct
- Ensure the service account has permissions to the folder

### Folder Not Found

- Double-check the folder path: `/sites/CodedWork9/Shared Documents/IAS/2025/MOR`
- Verify the site path: `/sites/CodedWork9`
