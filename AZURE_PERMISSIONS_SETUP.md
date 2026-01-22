# Azure AD API Permissions Setup Guide

## Problem
The script is getting a 401 Unauthorized error because your Azure AD application doesn't have the required Microsoft Graph API permissions to access SharePoint.

## Solution: Add API Permissions

Follow these steps in the Azure Portal:

### 1. Navigate to API Permissions
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** → **App registrations**
3. Click on **Leon MD Token** (your application)
4. Click **API permissions** in the left sidebar

### 2. Add Microsoft Graph Permissions
1. Click **+ Add a permission**
2. Select **Microsoft Graph**
3. Select **Application permissions** (not Delegated)
4. Search for and add these permissions:
   - `Sites.Read.All` - Read items in all site collections
   - `Files.Read.All` - Read files in all site collections

### 3. Grant Admin Consent
⚠️ **CRITICAL STEP**: After adding the permissions:
1. Click the **Grant admin consent for [Your Organization]** button
2. Click **Yes** to confirm
3. Wait for the status to show green checkmarks

### 4. Verify Permissions
You should see:
- ✅ Sites.Read.All (Application) - Granted for [Your Organization]
- ✅ Files.Read.All (Application) - Granted for [Your Organization]

## After Setup
Once the permissions are granted, run the script again:
```bash
python3 list_sharepoint_folder.py
```

## Alternative: Use SharePoint-Specific Permissions
If you prefer to use SharePoint REST API instead of Microsoft Graph:
1. Add permission for **SharePoint** (not Microsoft Graph)
2. Select **Application permissions**
3. Add `Sites.Read.All`
4. Grant admin consent

---

**Note**: Permission changes may take a few minutes to propagate. If you still get errors after granting consent, wait 2-3 minutes and try again.
